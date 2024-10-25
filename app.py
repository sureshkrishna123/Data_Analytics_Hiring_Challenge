import streamlit as st
import joblib
import pandas as pd

# Load the trained model
model = joblib.load('linear_regression_model.joblib')

# Function to make predictions
def predict_effective_soc(features):
    # Convert features into DataFrame for prediction
    features_df = pd.DataFrame(features, index=[0])
    prediction = model.predict(features_df)
    return prediction[0]

# Streamlit UI
st.title("Effective SOC Predictor")
st.write("Enter the following values:")

# User input for features
fixed_battery_voltage = st.number_input("Fixed Battery Voltage (V)", min_value=0.0)
portable_battery_voltage = st.number_input("Portable Battery Voltage (V)", min_value=0.0)
portable_battery_current = st.number_input("Portable Battery Current (A)", min_value=0.0)
fixed_battery_current = st.number_input("Fixed Battery Current (A)", min_value=0.0)
motor_status = st.selectbox("Motor Status (On/Off)", ["0", "1"])  # 0 for Off, 1 for On
bcm_battery_selected = st.selectbox("BCM Battery Selected (0/1)", ["0", "1"])  # 0 for Not Selected, 1 for Selected
portable_battery_temperatures = st.number_input("Portable Battery Temperatures (°C)", min_value=-40.0, max_value=100.0)
fixed_battery_temperatures = st.number_input("Fixed Battery Temperatures (°C)", min_value=-40.0, max_value=100.0)

# Calculate 'Voltage Difference'
voltage_difference = fixed_battery_voltage - portable_battery_voltage

# Calculate 'Normalized Portable Temp'
normalized_portable_temp = (
    portable_battery_temperatures - portable_battery_temperatures.mean()
) / portable_battery_temperatures.std()

# Button to predict
if st.button("Predict Effective SOC"):
    features = {
        "Fixed Battery Voltage": fixed_battery_voltage,
        "Portable Battery Voltage": portable_battery_voltage,
        "Portable Battery Current": portable_battery_current,
        "Fixed Battery Current": fixed_battery_current,
        "Motor Status (On/Off)": int(motor_status),  # Convert string to int
        "BCM Battery Selected": int(bcm_battery_selected),  # Convert string to int
        "Portable Battery Temperatures": portable_battery_temperatures,
        "Fixed Battery Temperatures": fixed_battery_temperatures,
        "Voltage Difference": voltage_difference,
        "Normalized Portable Temp": normalized_portable_temp,
    }
    
    # Ensure that the features dictionary has the same keys as the model was trained with
    try:
        predicted_soc = predict_effective_soc(features)
        st.success(f"Predicted Effective SOC: {predicted_soc:.2f}")
    except Exception as e:
        st.error(f"Error in prediction: {str(e)}")
