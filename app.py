import streamlit as st
import pandas as pd
import joblib

# Load the trained linear regression model
model = joblib.load(r"linear_regression_model.joblib")

# Streamlit app setup
st.title("Effective SOC Prediction App")
st.write("Enter the input values for all required features to predict the Effective SOC.")

# Input fields for all 8 features
fixed_voltage = st.number_input("Fixed Battery Voltage", value=3.7, format="%.2f")
portable_voltage = st.number_input("Portable Battery Voltage", value=3.6, format="%.2f")
portable_temp = st.number_input("Portable Battery Temperatures", value=25.0, format="%.2f")
fixed_temp = st.number_input("Fixed Battery Temperatures", value=25.0, format="%.2f")
bcm_selected = st.selectbox("BCM Battery Selected (1 for Yes, 0 for No)", [0, 1])
motor_status = st.selectbox("Motor Status (On/Off - 1 for On, 0 for Off)", [0, 1])

# Calculate 'Voltage Difference' as a derived feature
voltage_diff = fixed_voltage - portable_voltage

# Display the derived feature value for user's reference
st.write(f"Voltage Difference (Fixed - Portable): {voltage_diff:.2f}")

# Prepare the input data with all features
input_data = pd.DataFrame(
    [[fixed_voltage, portable_voltage, portable_temp, fixed_temp, 
      bcm_selected, motor_status, voltage_diff]],
    columns=[
        'Fixed Battery Voltage', 
        'Portable Battery Voltage', 
        'Portable Battery Temperatures', 
        'Fixed Battery Temperatures', 
        'BCM Battery Selected', 
        'Motor Status (On/Off)', 
        'Voltage Difference'
    ]
)

# Show the input data in the app for verification
st.write("Input Data Preview:")
st.dataframe(input_data)

# Make a prediction when the button is clicked
if st.button("Predict Effective SOC"):
    try:
        # Perform the prediction
        soc_prediction = model.predict(input_data)
        st.success(f"Predicted Effective SOC: {soc_prediction[0]:.2f}")
    except Exception as e:
        st.error(f"An error occurred during prediction: {e}")
