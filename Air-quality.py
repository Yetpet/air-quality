import streamlit as st
from streamlit.logger import get_logger
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import joblib as jb
from sklearn.preprocessing import LabelEncoder
from datetime import datetime
import folium
from folium.plugins import MarkerCluster
from branca.colormap import linear
import pickle as pk
import bokeh
from bokeh.plotting import figure, show
from bokeh.plotting import figure, show
from bokeh.tile_providers import get_provider, Vendors
from bokeh.io import output_notebook, push_notebook
from bokeh.models import HoverTool, ColumnDataSource

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

    # Display the table of Nigerian states and coordinates
    st.write("## Nigerian States and Coordinates")
    # Display the table of Nigerian states and coordinates
    show_coordinates_table = st.checkbox("Show Nigerian States and Coordinates")
    if show_coordinates_table:
        # Convert DataFrame to markdown table without index
        markdown_table = state_coordinates_df.to_markdown(index=False)

        # Display the markdown table using st.markdown
        st.markdown(markdown_table, unsafe_allow_html=True)

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
                if 4 < aqi_value < 5:
                    return 'red'
                elif 3 < aqi_value < 4:
                    return 'darkorange'
                elif 2 < aqi_value < 3:
                    return 'yellow'
                elif 1 < aqi_value < 2:
                    return 'lightgreen'
                else:
                    return 'green'
        
        # Get the overall AQI value (assuming it's in the first column of predictions)
        overall_aqi = predictions[0]

        # Get the assigned color for the AQI value
        aqi_color = assign_color(overall_aqi)
        m = folium.Map(location=[lon, lat], zoom_start=6)

        # Add markers with custom icons and tooltips
        # for lat, lon, aqi_value in zip([9.0820, 9.5, 10], [8.6753, 8.8, 9], predictions):
        #     folium.Marker(
        #         location=[lat, lon],
        #         popup=f"AQI: {aqi_value}",
        #         icon=folium.Icon(color=assign_color(aqi_value))
        #     ).add_to(m)
        # # Add marker with custom icon and tooltip
        # tooltip = f"AQI: {predictions[0]}, CO: {predictions[1]}, NO: {predictions[2]}, NO2: {predictions[3]}, O3: {predictions[4]}, SO2: {predictions[5]}, PM2.5: {predictions[6]}, PM10: {predictions[7]}, NH3: {predictions[8]}"
        # folium.Marker(
        #     location=[lat, lon],
        #     # popup=tooltip,
        #     # icon=folium.Icon(color=assign_color(predictions[0]))
        #     popup=folium.Popup(tooltip, max_width=300),
        #     icon=folium.Icon(color=assign_color(predictions[0]), prefix='fa', icon='info-circle')
        # ).add_to(m)

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
            icon=folium.Icon(color=assign_color(overall_aqi), prefix='fa', icon='info-circle')
        ).add_to(m)

        # # Add marker with tooltip containing predicted values
        # tooltip = f"AQI: {predictions[0]}, CO: {predictions[1]}, NO: {predictions[2]}, NO2: {predictions[3]}, O3: {predictions[4]}, SO2: {predictions[5]}, PM2.5: {predictions[6]}, PM10: {predictions[7]}, NH3: {predictions[8]}"
        # folium.Marker([lon, lat], popup=tooltip).add_to(m)

        return m

    # # Function to create a Plotly Express scatter map
    # def create_plotly_map(lat, lon, predictions):
    #     hover_data = ["AQI", "CO", "NO", "NO2", "O3", "SO2", "PM2.5", "PM10", "NH3"]
    #     # color = predictions["AQI"]

    #     # Create a DataFrame with user-input latitude and longitude
    #     user_location_df = pd.DataFrame({"lon": [lon], "lat": [lat]})

    #     fig = px.scatter_mapbox(
    #         predictions,
    #         lon=lon,  # Use the user-input longitude
    #         lat=lat,  # Use the user-input latitude
    #         hover_data=hover_data,
    #         # color=color,
    #         color_continuous_scale=px.colors.sequential.Viridis,
    #         title=f"Air Quality Map for Location ({lat}, {lon})",
    #     )

    #     # Add user location as a scatter point
    #     fig.add_trace(px.scatter_mapbox(user_location_df, lon="lon", lat="lat").data[0])


    #     return fig

    # # Function to create a Plotly Express scatter map
    # def create_plotly_map(lon, lat, predictions):
    #     #hover_data = ["lon","lat","AQI", "CO", "NO", "NO2", "O3", "SO2", "PM2.5", "PM10", "NH3"]
    #     hover_data = ["lon","lat"]
    #     # color = predictions[:, 0]  # Assuming AQI is the first column, adjust if needed

    #     # Create a DataFrame with user-input latitude and longitude
    #     user_location_df = pd.DataFrame({"lon": [lon], "lat": [lat]})

    #     fig = px.scatter_mapbox(
    #         user_location_df,  # Update here, assuming lon and lat are correctly formatted
    #         lon="lon",
    #         lat="lat",
    #         hover_data=hover_data,
    #         color_discrete_sequence=["red"],  # Color for user location
    #         title=f"Air Quality Map for Location ({lat}, {lon})"
    #     )

    #     # Add scatter map for predictions
    #     fig.add_trace(px.scatter_mapbox(
    #         predictions,
    #         # lon=lon,  # Use the user-input longitude
    #         # lat=lat,  # Use the user-input latitude
    #         hover_data=hover_data,
    #         # color=color,  # Assign colors based on AQI value
    #         color_continuous_scale=px.colors.sequential.Viridis,
    #     ).data[0])

    #     return fig



    # # Function to create a Bokeh map
    # def create_bokeh_map(lon, lat, predictions):
    #     # Assuming predictions has AQI values
    #     p = figure(x_range=(lon - 1, lon + 1),
    #             y_range=(lat - 1, lat + 1),
    #             # plot_width=800,
    #             title="Air Quality Map",
    #             tools="pan,box_zoom,wheel_zoom,reset,save",
    #             x_axis_label='Longitude',
    #             y_axis_label='Latitude',
    #             toolbar_location="below",
    #             tooltips=[("Longitude", f"{lon}"), ("Latitude", f"{lat}"), ("AQI", "@aqi")]
    #             )

    #     tile_provider = get_provider(Vendors.CARTODBPOSITRON)
    #     p.add_tile(tile_provider)

    #     # Add a circle to represent the user input location
    #     p.circle(x=[lon], y=[lat], size=10, color="blue", alpha=0.7)

    #     # Add circles to represent other locations with colors based on AQI values
    #     # Assuming AQI is the only prediction value
    #     colors = ["red" if aqi >= 5 else "darkorange" if aqi == 4 else "yellow" if aqi == 3 else "lightgreen" if aqi == 2 else "green" for aqi in predictions]

    #     # Assuming you want to add some latitude and longitude values from predictions
    #     latitudes = [lat + 0.01 * i for i in range(len(predictions))]
    #     longitudes = [lon + 0.01 * i for i in range(len(predictions))]

    #     p.circle(x=longitudes, y=latitudes, size=10, color=colors, alpha=0.5)

    #     return p



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

        # # Convert predictions to DataFrame
        # predictions_df = pd.DataFrame({
        #     'Air Pollutant': ['AQI', 'CO', 'NO', 'NO2', 'O3', 'SO2', 'PM2.5', 'PM10', 'NH3'],
        #     'Prediction': predictions.flatten()  # Flatten the predictions to a 1D array
        # })

        # # Display predictions
        # st.write("### Predicted Air Quality Values for {}: ".format(state_name))
        # st.write(predictions_df)


        # Create and display folium map
        st.write("### Air Quality Map:")
        folium_map = create_map(lon, lat, predictions)
        st.write(folium_map)

        # # Create and display Plotly Express map
        # st.write("### Air Quality Map:")
        # plotly_map = create_plotly_map(lon, lat, predictions)  # Make sure to pass the correct DataFrame
        # st.plotly_chart(plotly_map)

        # # Create and display Bokeh map
        # st.write("### Air Quality Map:")
        # bokeh_map = create_bokeh_map(lon, lat, predictions)
        # st.bokeh_chart(bokeh_map)

        # # Map the AQI values to intensity levels
        # aqi_intensity_mapping = {
        #     1: 'Good',
        #     2: 'Fair',
        #     3: 'Moderate',
        #     4: 'Poor',
        #     5: 'Very Poor'
        # }

        # # Add AQI Intensity column
        # predictions_df['AQI Intensity'] = predictions_df['Prediction'].apply(lambda x: min(int(x // 1), 5))

        # # Map AQI Intensity to description
        # predictions_df['AQI Desc'] = predictions_df['AQI Intensity'].map(aqi_intensity_mapping)

        # # Create Plotly map
        # fig = px.scatter_mapbox(
        #     predictions_df,
        #     lon=lon,
        #     lat=lat,
        #     hover_data=['Air Pollutant', 'Prediction', 'AQI Desc'],
        #     color='AQI Intensity',
        #     color_continuous_scale=px.colors.sequential.Viridis,
        #     title=f"Air Quality Map for Location ({lat}, {lon})"
        # )

        # # Add user location as a scatter point
        # user_location_df = pd.DataFrame({"lon": [lon], "lat": [lat]})
        # fig.add_trace(px.scatter_mapbox(user_location_df, lon="lon", lat="lat").data[0])

        # # Display the Plotly map
        # st.write("### Air Quality Map:")
        # st.plotly_chart(fig)

        # ...

        # # Display predictions
        # st.write("### Predicted Air Quality Values for {}: ".format(state_name))
        # st.write(pd.DataFrame({"Air Pollutant": ['AQI', 'CO', 'NO', 'NO2', 'O3', 'SO2', 'PM2.5', 'PM10', 'NH3'],
        #                     "Prediction": predictions,
        #                     "AQI Intensity": predictions['AQI'].apply(lambda x: min(int(x // 1), 5)),
        #                     "AQI Desc": pd.cut(predictions['AQI'], bins=[0, 1, 2, 3, 4, 5],
        #                                         labels=['Good', 'Fair', 'Moderate', 'Poor', 'Very Poor'])}))

        # # Create DataFrame for Plotly map
        # predictions_df = pd.DataFrame({
        #     "Air Pollutant": ['AQI', 'CO', 'NO', 'NO2', 'O3', 'SO2', 'PM2.5', 'PM10', 'NH3'],
        #     "Prediction": predictions,
        #     "AQI Intensity": predictions['AQI'].apply(lambda x: min(int(x // 1), 5)),
        #     "AQI Desc": pd.cut(predictions['AQI'], bins=[0, 1, 2, 3, 4, 5],
        #                     labels=['Good', 'Fair', 'Moderate', 'Poor', 'Very Poor']),
        #     "lat": [lat] * 9,  # Repeat lat value for each pollutant
        #     "lon": [lon] * 9   # Repeat lon value for each pollutant
        # })

        # # Display the Plotly map
        # st.write("### Air Quality Map:")
        # fig = px.scatter_mapbox(
        #     predictions_df,
        #     lat="lat",  # Use the correct column name for latitude
        #     lon="lon",  # Use the correct column name for longitude
        #     hover_name="Air Pollutant",
        #     hover_data=["Prediction", "AQI Intensity", "AQI Desc"],
        #     color="AQI Intensity",
        #     color_discrete_map={
        #         1: 'green',  # Good
        #         2: 'yellow',  # Fair
        #         3: 'lightblue',  # Moderate
        #         4: 'orange',  # Poor
        #         5: 'red'  # Very Poor
        #     },
        #     title=f"Air Quality Map for Location ({lat}, {lon})",
        # )

        # st.plotly_chart(fig)
        # # Create a scatter mapbox
        # fig = px.scatter_mapbox(
        #     predictions,
        #     # lat="lat",
        #     # lon="lon",
        #     # hover_data=["AQI", "CO", "NO", "NO2", "O3", "SO2", "PM2.5", "PM10", "NH3"],
        #     # color="AQI",
        #     # color_continuous_scale=px.colors.sequential.Viridis,
        #     title="Air Quality Map",
        # )

        # # Show the map
        # fig.show()

        # # Create DataFrame with all predicted values
        # predictions_df = pd.DataFrame({
        #     "Air Pollutant": ['AQI', 'CO', 'NO', 'NO2', 'O3', 'SO2', 'PM2.5', 'PM10', 'NH3'],
        #     "Prediction": predictions.flatten()  # Assuming 'predictions' is a 2D array
        # })

        # # Add 'AQI Intensity' column
        # predictions_df['AQI Intensity'] = predictions_df['Prediction'].apply(lambda x: min(int(x // 1), 5))

        # # Display the Plotly map
        # st.write("### Air Quality Map:")
        # fig = px.scatter_mapbox(
        #     predictions_df,
        #     lat=[lat] * len(predictions_df),  # Repeat lat value for each pollutant
        #     lon=[lon] * len(predictions_df),  # Repeat lon value for each pollutant
        #     hover_name="Air Pollutant",
        #     hover_data=["Prediction", "AQI Intensity"],
        #     color="AQI Intensity",
        #     color_discrete_map={
        #         1: 'green',  # Good
        #         2: 'yellow',  # Fair
        #         3: 'lightblue',  # Moderate
        #         4: 'orange',  # Poor
        #         5: 'red'  # Very Poor
        #     },
        #     title=f"Air Quality Map for Location ({lat}, {lon})",
        # )

        # st.plotly_chart(fig)


        # # Sample predictions DataFrame
        # predictions_data = pd.DataFrame({
        #     "AQI": [3.63],
        #     "CO": [725.5887],
        #     "NO": [-0.087],
        #     "NO2": [3.4732],
        #     "O3": [19.5887],
        #     "SO2": [0.7506],
        #     "PM2.5": [55.2077],
        #     "PM10": [173.1008],
        #     "NH3": [3.3307],
        #     "Latitude": [12.971598],  # Updated latitude for Bangalore, India
        #     "Longitude": [77.594562],  # Updated longitude for Bangalore, India
        # })

        # # Display the air pollution map using px.scatter_mapbox
        # st.write("### Air Pollution Map:")
        # fig = px.scatter_mapbox(
        #     predictions_data,
        #     lat="Latitude",
        #     lon="Longitude",
        #     hover_name="AQI",  # Use AQI as the identifier for the tooltip
        #     hover_data={
        #         "Latitude": False,
        #         "Longitude": False,
        #         "AQI": ":.2f",
        #         "CO": ":.2f",
        #         "NO": ":.3f",
        #         "NO2": ":.4f",
        #         "O3": ":.2f",
        #         "SO2": ":.4f",
        #         "PM2.5": ":.2f",
        #         "PM10": ":.2f",
        #         "NH3": ":.4f",
        #     },
        #     color="AQI",
        #     color_continuous_scale="Viridis",
        #     title="Air Pollution Map",
        # )
        # st.plotly_chart(fig)



if __name__ == "__main__":
    run()
