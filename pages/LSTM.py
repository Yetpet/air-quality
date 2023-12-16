# -*- coding: utf-8 -*-
"""

"""

# Importing necessary libraries
import requests
import streamlit as st
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import LSTM, Dense

# Function to get current weather data
def get_current_weather(api_key, latitude, longitude):
    base_url = "https://api.openweathermap.org/data/3.0/onecall"
    params = {
        "lat": latitude,
        "lon": longitude,
        "exclude": "minutely,daily",
        "appid": api_key,
        "units": "metric",  
    }

    try:
        response = requests.get(base_url, params=params)
        response.raise_for_status()
        weather_data = response.json()
        return weather_data
    except requests.exceptions.RequestException as e:
        st.error(f"Error fetching weather data: {e}")
        return None

# Streamlit App
st.title('Anomaly Detection with LSTM')

# User Input
latitude = st.number_input('Enter Latitude:')
longitude = st.number_input('Enter Longitude:')

# OpenWeatherMap API key
api_key = ''  

# Check if the API key is provided
if api_key:
    # Get current weather data
    current_weather_data = get_current_weather(api_key, latitude, longitude)

    if current_weather_data:
        # Display fetched weather data
        st.subheader("Current Weather Data:")
        st.write(f"Temperature: {current_weather_data['current']['temp']}°C")
        st.write(f"Humidity: {current_weather_data['current']['humidity']}%")
        st.write(f"Pressure: {current_weather_data['current']['pressure']} hPa")

        # Check if humidity is greater than 50%
        if current_weather_data['current']['humidity'] > 50:
            st.warning("High humidity detected! Please be cautious.")

        
        # ...

        # Assuming 'hourly' is your relevant data
        hourly_data = current_weather_data['hourly']

        # Extracting relevant features (temperature, humidity, pressure)
        temperature_data = [entry['temp'] for entry in hourly_data]
        humidity_data = [entry['humidity'] for entry in hourly_data]
        pressure_data = [entry['pressure'] for entry in hourly_data]

        # Convert to DataFrame
        df = pd.DataFrame({'temperature': temperature_data, 'humidity': humidity_data, 'pressure': pressure_data})

        # Normalize the data
        scaler = StandardScaler()
        df[['temperature', 'humidity', 'pressure']] = scaler.fit_transform(df[['temperature', 'humidity', 'pressure']])

        # Define sequence length (number of time steps to look back)
        sequence_length = 10  

        # Create sequences and labels
        sequences = []
        labels = []

        for i in range(len(df) - sequence_length):
            seq = df.iloc[i:i+sequence_length].values
            label = df.iloc[i+sequence_length]['temperature']  
            sequences.append(seq)
            labels.append(label)

        # Convert to numpy arrays
        X = np.array(sequences)
        y = np.array(labels)

        # Split the data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Build the LSTM model
        model = Sequential()
        model.add(LSTM(50, activation='relu', input_shape=(sequence_length, 3)))  
        model.add(Dense(1))
        model.compile(optimizer='adam', loss='mse')

        # Reshape the data for LSTM input
        X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 3))  
        X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 3))  

        # Train the model
        model.fit(X_train, y_train, epochs=40, batch_size=22, validation_data=(X_test, y_test), verbose=2)

        # Make predictions on the test set
        y_pred = model.predict(X_test)
# Evaluate the model and calculate the mean squared error 
mse = np.mean((y_pred - y_test)**2)
st.write(f'Mean Squared Error: {mse}')

# Set a threshold for anomaly detection using mean and standard deviation 
k = 2  
threshold = np.mean(mse) + k * np.std(mse)

# Detect anomalies based on the threshold
anomalies = np.where(mse > threshold)

# Print the indices of anomalies
st.write('Anomalies indices:', anomalies)

# Check if anomalies are detected and show a warning
if np.atleast_1d(anomalies)[0].size > 0:
    st.warning("Anomalies detected! Please be cautious.")
else:
    st.success("No anomalies detected. Weather conditions are stable.")

