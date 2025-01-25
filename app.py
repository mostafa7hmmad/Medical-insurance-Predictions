import streamlit as st
import numpy as np
import joblib

# Load the trained model
model = joblib.load("model.pkl")

# Streamlit app title
st.title("Medical Insurance Charges Prediction")

# Sidebar for user input
st.sidebar.header("User Input Parameters")

def user_input_features():
    age = st.sidebar.slider("Age", 18, 100, 30)  # Age slider
    smoker = st.sidebar.selectbox("Smoker", ("Yes", "No"))  # Smoker dropdown
    bmi = st.sidebar.slider("BMI", 10.0, 100.0, 25.0, step=0.1)  # BMI slider (range 10-100)
    
    # Convert smoker input to binary (0/1)
    smoker_binary = 1 if smoker == "Yes" else 0
    
    # Create feature array
    features = np.array([[age, smoker_binary, bmi]])
    return features

# Get user inputs
input_data = user_input_features()

# Make predictions using the model
if st.sidebar.button("Predict"):
    prediction = model.predict(input_data)
    st.write(f"### Predicted Charges: ${prediction[0]:,.2f}")

# Display model details and about section
st.sidebar.write("----")
st.sidebar.write("This model predicts medical insurance charges based on Age, Smoking Status, and BMI.")
st.sidebar.write("Created with a RandomForestRegressor.")
