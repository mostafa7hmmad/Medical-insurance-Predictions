import streamlit as st
import numpy as np
import joblib
from streamlit.components.v1 import html

# Function to render the initial animation
def render_animation():
    animation_code = """
    <div style="display:flex;justify-content:center;align-items:center;margin-bottom:20px;">
        <div style="width:100px;height:100px;border-radius:50%;background-color:#ff5722;animation:bounce 2s infinite;"></div>
        <style>
            @keyframes bounce {
                0%, 100% { transform: translateY(0); }
                50% { transform: translateY(-20px); }
            }
        </style>
    </div>
    """
    html(animation_code, height=200)

# Load the trained model
model = joblib.load("model.pkl")

# Streamlit app title
st.title("Medical Insurance Charges Prediction")
render_animation()

# Centered user input form
def user_input_features():
    with st.form(key="user_input_form"):
        st.markdown("### Enter Your Details")
        st.markdown("---")
        col1, col2 = st.columns(2)

        with col1:
            age = st.text_input("Age", "30")  # Age input
            bmi = st.text_input("BMI", "25.0")  # BMI input

        with col2:
            smoker = st.selectbox("Smoker", ("Yes", "No"))  # Smoker dropdown

        submit_button = st.form_submit_button(label="Predict")

        # Convert smoker input to binary (0/1)
        smoker_binary = 1 if smoker == "Yes" else 0

        # Create feature array
        if submit_button:
            return np.array([[int(age), smoker_binary, float(bmi)]]), True
        else:
            return None, False

input_data, is_submitted = user_input_features()

# Make predictions using the model
if is_submitted:
    try:
        prediction = model.predict(input_data)
        st.write(f"### Predicted Charges: ${prediction[0]:,.2f}")
    except Exception as e:
        st.error(f"An error occurred during prediction: {e}")

# About section
st.markdown("----")
st.markdown(
    "This model predicts medical insurance charges based on Age, Smoking Status, and BMI.\n"
    "Created with a RandomForestRegressor."
)
