import streamlit as st
import pandas as pd
import joblib

# Load the trained model
model = joblib.load(r"linear_regression_model.joblib")

# Streamlit app setup
st.title("Effective SOC Prediction App")
st.write("Enter the required inputs to predict the Effective SOC.")


# Input fields for all required features
fixed_voltage = st.number_input("Fixed Battery Voltage", value=3.7, format="%.2f")
portable_voltage = st.number_input("Portable Battery Voltage", value=3.6, format="%.2f")
portable_temp = st.number_input("Portable Battery Temperatures", value=25.0, format="%.2f")
fixed_temp = st.number_input("Fixed Battery Temperatures", value=25.0, format="%.2f")
fixed_current = st.number_input("Fixed Battery Current", value=1.0, format="%.2f")
portable_current = st.number_input("Portable Battery Current", value=1.0, format="%.2f")
bcm_selected = st.selectbox("BCM Battery Selected (1 for Yes, 0 for No)", [0, 1])
motor_status = st.selectbox("Motor Status (On/Off - 1 for On, 0 for Off)", [0, 1])

# Calculate derived features
voltage_diff = fixed_voltage - portable_voltage
normalized_portable_temp = (portable_temp - 25) / 5  # Assuming mean=25, std=5

# Prepare the input data for prediction
input_data = pd.DataFrame(
    [[fixed_voltage, portable_voltage, portable_temp, fixed_temp, 
      fixed_current, portable_current, bcm_selected, motor_status, 
      voltage_diff, normalized_portable_temp]],
    columns=[
        'Fixed Battery Voltage', 
        'Portable Battery Voltage', 
        'Portable Battery Temperatures', 
        'Fixed Battery Temperatures', 
        'Fixed Battery Current', 
        'Portable Battery Current', 
        'BCM Battery Selected', 
        'Motor Status (On/Off)', 
        'Voltage Difference', 
        'Normalized Portable Temp'
    ]
)




# Perform prediction when the button is clicked
if st.button("Predict Effective SOC"):
    try:
        # Make prediction
        soc_prediction = model.predict(input_data)
        st.success(f"Predicted Effective SOC: {soc_prediction[0]:.2f}")
    except Exception as e:
        st.error(f"An error occurred during prediction: {e}")
