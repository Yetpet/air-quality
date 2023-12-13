import streamlit as st
from streamlit.logger import get_logger
import numpy as np
import pandas as pd
import joblib as jb
from sklearn.preprocessing import LabelEncoder
from datetime import datetime
import folium
from folium.plugins import MarkerCluster
from branca.colormap import linear
import pickle as pk

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

    # Assuming 'scaler' is already fitted on the training data
    global scaler_fitted 
    scaler_fitted = False  # Flag to check if the scaler is fitted

    # Load the trained model and scaler
    try:
        with open('mgbr_model.joblib', 'rb') as model_file:
            model = jb.load(model_file)
            print(type(model))  # Add this line to check the type of 'model'
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
        # global scaler_fitted  # Declare it as global to modify the global variable

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

        # Make predictions using your model
        # Replace this with actual prediction logic
        # predictions = model.predict([[lat, lon, day, month, hour, state_label]])
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

        # Extract the features to be normalized
        # features_to_normalize = nigeria_air_pollution[columns_to_normalize]
        
        # Normalize the input features using the fitted scaler
        #input_features = np.array([[lon, lat, month, day, hour, state_label]])  # Convert to NumPy array
        #normalized_features = pd.DataFrame(scaler.fit_transform(input_features), columns=columns_to_normalize)
        normalized_features = scaler.transform(input_features)
        # normalized_features = pd.DataFrame(scaler.transform(input_features), columns=columns_to_normalize).to_numpy()

        # Combine the normalized features with other non-normalized columns
        input_data = pd.DataFrame(normalized_features, columns=columns_to_normalize)


        # Combine the normalized features with other non-normalized columns
        # input_data = pd.concat([input_features.drop(columns=columns_to_normalize), normalized_features], axis=1)
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
            if aqi_value == 5:
                return 'red'
            elif aqi_value == 4:
                return 'darkorange'
            elif aqi_value == 3:
                return 'yellow'
            elif aqi_value == 2:
                return 'lightgreen'
            else:
                return 'green'

        m = folium.Map(location=[lon, lat], zoom_start=8)

        # Add markers with custom icons and tooltips
        # for lat, lon, aqi_value in zip([9.0820, 9.5, 10], [8.6753, 8.8, 9], predictions):
        #     folium.Marker(
        #         location=[lat, lon],
        #         popup=f"AQI: {aqi_value}",
        #         icon=folium.Icon(color=assign_color(aqi_value))
        #     ).add_to(m)
        # Add marker with custom icon and tooltip
        tooltip = f"AQI: {predictions[0]}, CO: {predictions[1]}, NO: {predictions[2]}, NO2: {predictions[3]}, O3: {predictions[4]}, SO2: {predictions[5]}, PM2.5: {predictions[6]}, PM10: {predictions[7]}, NH3: {predictions[8]}"
        folium.Marker(
            location=[lat, lon],
            popup=tooltip,
            icon=folium.Icon(color=assign_color(predictions[0]))
        ).add_to(m)

        # # Add marker with tooltip containing predicted values
        # tooltip = f"AQI: {predictions[0]}, CO: {predictions[1]}, NO: {predictions[2]}, NO2: {predictions[3]}, O3: {predictions[4]}, SO2: {predictions[5]}, PM2.5: {predictions[6]}, PM10: {predictions[7]}, NH3: {predictions[8]}"
        # folium.Marker([lon, lat], popup=tooltip).add_to(m)

        return m

    # Streamlit UI
    #st.title("Air Quality Prediction for Nigerian States")

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
        
        # st.write(pd.DataFrame({"Air Pollutant": ['AQI', 'CO', 'NO', 'NO2', 'O3', 'SO2', 'PM2.5', 'PM10', 'NH3'],
        #                "Prediction": predictions.flatten()}))


        # Create and display folium map
        st.write("### Air Quality Map:")
        folium_map = create_map(lon, lat, predictions)
        st.write(folium_map)


if __name__ == "__main__":
    run()
