import streamlit as st
from streamlit.logger import get_logger
import numpy as np
import pandas as pd
import plotly.express as px
import joblib as jb
from sklearn.preprocessing import LabelEncoder
from datetime import datetime
import folium
from branca.colormap import linear


# ...

LOGGER = get_logger(__name__)


def run():
    st.set_page_config(
        page_title="Hello",
        page_icon="👋",
    )

    st.write("# Welcome to MandelaAir Guard! 👋")

    st.markdown(
        """
        This web page is created to monitor the air quality of locations in Nigeria. The application provides the overall Air Quality Index derived from different pollutants, namely - CO, NO, NO3, O3, SO2, NH3, PM2.5, PM10, present in the air.

        👈 Enter the longitude and latitude of a state in the sidebar to see what the air quality feels like!
    """
    )

    # Read data from CSV file
    file_path = 'nigerian_states.csv'
    state_coordinates_df = pd.read_csv(file_path)

    # Display Nigerian states and coordinates
    st.write("## Nigerian States and Coordinates")
    show_coordinates_table = st.checkbox("Show Nigerian States and Coordinates")
    if show_coordinates_table:
        # Convert DataFrame to markdown table without index
        markdown_table = state_coordinates_df.to_markdown(index=False)
        st.markdown(markdown_table, unsafe_allow_html=True)

    global scaler_fitted 
    scaler_fitted = False  # Flag to check if the scaler is fitted

    # Load the trained model and scaler
    try:
        with open('mgbr_model.joblib', 'rb') as model_file:
            model = jb.load(model_file)
            print(type(model))  
    except FileNotFoundError:
        st.warning("Model not found. Please retrain the model.")
    try:
        with open('scaler.pkl', 'rb') as scaler_file:
            scaler = jb.load(scaler_file)
        scaler_fitted = True
    except FileNotFoundError:
        st.warning("Scaler not found. Please retrain the model to generate a fitted scaler.")
        scaler_fitted = False


    # Placeholder DataFrame for state labels
    state_labels_df = pd.DataFrame({
        'state_label': list(range(37)),
        'state_name': ['Abia', 'Adamawa', 'Akwa Ibom', 'Anambra', 'Bauchi', 'Bayelsa', 'Benue', 'Borno', 'Cross River',
                    'Delta', 'Ebonyi', 'Edo', 'Ekiti', 'Enugu', 'Federal Capital Territory', 'Gombe', 'Imo', 'Jigawa',
                    'Kaduna', 'Kano', 'Katsina', 'Kebbi', 'Kogi', 'Kwara', 'Lagos', 'Nasarawa', 'Niger', 'Ogun', 'Ondo',
                    'Osun', 'Oyo', 'Plateau', 'Rivers', 'Sokoto', 'Taraba', 'Yobe', 'Zamfara']
    })

    # Create a label encoder and fit_transform the state names
    label_encoder = LabelEncoder()
    state_labels_df['state_label'] = label_encoder.fit_transform(state_labels_df['state_name'])

    # Function to preprocess input and make predictions
    def make_predictions(lon, lat, date, state_name):
        if not scaler_fitted:
            st.warning("Scaler is not fitted. Please retrain the model to generate a fitted scaler.")
            return np.zeros(9)  # Return zeros or any default values as needed

        # Feature engineering based on date
        date = datetime.strptime(str(date), "%Y-%m-%d")
        day = date.day
        month = date.month
        hour = date.hour

        # Label encode state_name
        state_label = label_encoder.transform([state_name])[0]

        # Normalize the input features using Min-Max scaling
        input_features = pd.DataFrame({
            'lon': [lon],
            'lat': [lat],
            'month': [month],
            'day': [day],
            'hour': [hour],
            'state_label': [state_label]
        })

        columns_to_normalize = ['lon', 'lat', 'month', 'day', 'hour','state_label']

        # Normalize the input features using the fitted scaler
        normalized_features = scaler.transform(input_features)

        # Combine the normalized features with other non-normalized columns
        input_data = pd.DataFrame(normalized_features, columns=columns_to_normalize)

        # Combine the normalized features with other non-normalized columns
        #new_data = pd.DataFrame([[7.5153071,5.454095299, 12,15,21, 14]], columns=['lon', 'lat', 'month', 'day', 'hour','state_label'])

        # Make predictions using the trained model
        predictions = model.predict(input_data)
        #predictions = np.random.rand(9)  # Dummy values for demonstration

        #return predictions
        return predictions.flatten()  # Flatten the predictions to a 1D array

    # Function to create a folium map
    def create_map(lon, lat, predictions):
        # Create a colormap for AQI values
        colormap = linear.RdYlGn_11.scale(min(predictions), max(predictions))

        # Function to assign colors based on AQI category
        def assign_color(aqi_value):
            if 4 < aqi_value <= 5:
                return '#88001e'  # Red for very poor
            elif 3 < aqi_value <= 4:
                return '#ff0000'  # Dark Orange for poor
            elif 2 < aqi_value <= 3:
                return '#ff7602'  # Yellow for moderate
            elif 1 < aqi_value <= 2:
                return '#fec201'  # Light Green for satisfactory
            else:
                return '#8cc63e'  # Green for good
        
        # Get the overall AQI value (assuming it's in the first column of predictions)
        overall_aqi = predictions[0]

        # Get the assigned color for the AQI value
        aqi_color = assign_color(overall_aqi)
        m = folium.Map(location=[lat, lon], zoom_start=6)

        # Add a Marker with a custom icon and tooltip
        tooltip = f"""
            <div style="font-family: 'Arial', sans-serif; font-size: 14px; padding: 10px; background-color: {aqi_color}; color: white; border-radius: 5px;">
                <b>AQI:</b> {predictions[0]}<br>
                <b>CO:</b> {predictions[1]}<br>
                <b>NO:</b> {predictions[2]}<br>
                <b>NO2:</b> {predictions[3]}<br>
                <b>O3:</b> {predictions[4]}<br>
                <b>SO2:</b> {predictions[5]}<br>
                <b>PM2.5:</b> {predictions[6]}<br>
                <b>PM10:</b> {predictions[7]}<br>
                <b>NH3:</b> {predictions[8]}<br>
            </div>
        """
        folium.Marker(
            location=[lat, lon],
            popup=folium.Popup(tooltip, max_width=300),
            icon=folium.Icon(color=aqi_color, prefix='fa', icon='info-circle')
        ).add_to(m)


        return m


    # Streamlit UI
    # Sidebar with user input
    st.sidebar.header("Enter Location Details")
    lon = st.sidebar.number_input("Longitude", min_value=-180.0, max_value=180.0, value=8.6753)
    lat = st.sidebar.number_input("Latitude", min_value=-90.0, max_value=90.0, value=9.0820)
    date = st.sidebar.date_input("Date")
    state_name = st.sidebar.selectbox("State Name", state_labels_df['state_name'].tolist())

    if st.sidebar.button("Make Predictions"):
        # Make predictions
        predictions = make_predictions(lon, lat, date, state_name)

        # Display predictions
        st.write("### Predicted Air Quality Values for {}: ".format(state_name))
        st.write(pd.DataFrame({"Air Pollutant": ['AQI', 'CO', 'NO', 'NO2', 'O3', 'SO2', 'PM2.5', 'PM10', 'NH3'],
                                "Prediction": predictions}))
        # Create and display folium map
        st.write("### Air Quality Map:")
        folium_map = create_map(lon, lat, predictions)
        st.write(folium_map)

if __name__ == "__main__":
    run()
