import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go
from datetime import datetime

# Load the trained model
model_filename = "rf_aqi_model.pkl"
rf_model = joblib.load(model_filename)

# Load the dataset used for training to get feature names
sample_data = pd.read_csv("delhi_aqi.csv")  # Use your original dataset file
sample_data['date'] = pd.to_datetime(sample_data['date'])

# Feature Engineering (same as training phase)
sample_data['hour'] = sample_data['date'].dt.hour
sample_data['day'] = sample_data['date'].dt.day
sample_data['month'] = sample_data['date'].dt.month
sample_data['season'] = sample_data['month'] % 12 // 3 + 1

# Creating Lag & Rolling Mean Features
for col in ['co', 'no', 'no2', 'o3', 'so2', 'pm2_5', 'pm10', 'nh3']:
    sample_data[f'{col}_lag1'] = sample_data[col].shift(1)
for col in ['pm2_5', 'pm10']:
    sample_data[f'{col}_rolling24'] = sample_data[col].rolling(window=24, min_periods=1).mean()
sample_data.dropna(inplace=True)

# Get feature names
features = [col for col in sample_data.columns if col not in ['date', 'pm2_5']]

# Streamlit App Title
st.title("Air Quality Prediction App 🌍")
st.subheader("Predict PM2.5 Levels Based on PM10 and Other Factors")

# User input for PM10 level
pm10_input = st.number_input("Enter current PM10 level:", min_value=0.0, step=1.0)

# Generate features based on current timestamp
current_time = datetime.now()

# Create input data with all necessary features
input_df = pd.DataFrame(columns=features)
input_df.loc[0] = sample_data[features].mean()  # Initialize with mean values
input_df['pm10'] = pm10_input  # Set user input value

# Add current datetime-based features
input_df['hour'] = current_time.hour
input_df['day'] = current_time.day
input_df['month'] = current_time.month
input_df['season'] = (current_time.month % 12) // 3 + 1

# Prediction Button
if st.button("Predict AQI"):
    predicted_aqi = rf_model.predict(input_df)[0]
    
    # AQI Classification
    def classify_aqi(value):
        if value <= 50:
            return "Good", "green"
        elif value <= 100:
            return "Moderate", "orange"
        else:
            return "Bad", "red"
    
    aqi_category, color = classify_aqi(predicted_aqi)
    
    # Display the result
    st.markdown(f"### Predicted AQI: *{predicted_aqi:.2f}*")
    st.markdown(f"#### Air Quality: *{aqi_category}*")
    st.markdown(f'<p style="color:{color}; font-size:20px;">Air Quality Status: {aqi_category}</p>', unsafe_allow_html=True)

    # AQI Meter
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=predicted_aqi,
        title={"text": "AQI Meter"},
        gauge={
            "axis": {"range": [0, 300]},
            "steps": [
                {"range": [0, 50], "color": "green"},
                {"range": [51, 100], "color": "orange"},
                {"range": [101, 300], "color": "red"}
            ],
            "bar": {"color": color}
        }
    ))
    st.plotly_chart(fig)

    # Simulated AQI Predictions for Next Few Hours (for Graph)
    future_hours = list(range(1, 6))
    future_aqi = [predicted_aqi + np.random.uniform(-10, 10) for _ in future_hours]  # Adding randomness for demo
    
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=future_hours, y=future_aqi, mode='lines+markers', name='Predicted AQI'))
    fig2.update_layout(title='Predicted AQI for Next Few Hours', xaxis_title='Hours Ahead', yaxis_title='AQI Level')
    st.plotly_chart(fig2)

# Run the app using: streamlit run app.py