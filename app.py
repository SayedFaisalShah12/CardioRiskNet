import streamlit as st
import numpy as np
import pandas as pd
import tensorflow as tf
import joblib
import os

# Page configuration
st.set_page_config(
    page_title="CardioRiskNet - CVD Prediction",
    page_icon="❤️",
    layout="wide"
)

# Custom CSS for a premium look
st.markdown("""
<style>
    .main {
        background-color: #f5f7f9;
    }
    .stButton>button {
        width: 100%;
        border-radius: 5px;
        height: 3em;
        background-color: #ff4b4b;
        color: white;
        font-weight: bold;
    }
    .result-card {
        padding: 2rem;
        border-radius: 10px;
        background-color: white;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        text-align: center;
    }
    .disclaimer {
        font-size: 0.8rem;
        color: #666;
        padding: 1rem;
        border-top: 1px solid #ddd;
        margin-top: 2rem;
    }
</style>
""", unsafe_allow_html=True)

# Load resources
@st.cache_resource
def load_assets():
    model = tf.keras.models.load_model('models/cardiorisknet_model.h5')
    scaler = joblib.load('models/scaler.joblib')
    return model, scaler

# Sidebar information
st.sidebar.title("❤️ CardioRiskNet")
st.sidebar.info("""
**Project CardioRiskNet**
A deep learning system for predicting cardiovascular disease risk based on patient clinical data.
""")

st.sidebar.subheader("Features Used")
st.sidebar.markdown("""
- Age & Sex
- Chest Pain Type (CP)
- Resting Blood Pressure
- Serum Cholesterol
- Fasting Blood Sugar
- ECG Results
- Max Heart Rate
- Exercise Induced Angina
- ST Depression (Oldpeak)
- ST Segment Slope
- Major Vessels (CA)
- Thalassemia
""")

# Main UI
st.title("Cardiovascular Disease Risk Prediction")
st.markdown("Enter the patient's clinical parameters to calculate the risk score using our deep neural network.")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Demographics & Vitals")
    age = st.number_input("Age", 1, 120, 50)
    sex = st.selectbox("Sex", options=[0, 1], format_func=lambda x: "Male" if x == 1 else "Female")
    trestbps = st.number_input("Resting Blood Pressure (mm Hg)", 50, 250, 120)
    chol = st.number_input("Serum Cholesterol (mg/dl)", 100, 600, 200)
    fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl", options=[0, 1], format_func=lambda x: "True" if x == 1 else "False")

with col2:
    st.subheader("Clinical Findings")
    cp = st.selectbox("Chest Pain Type", options=[0, 1, 2, 3], 
                    help="0: typical angina, 1: atypical angina, 2: non-anginal pain, 3: asymptomatic")
    restecg = st.selectbox("Resting ECG Results", options=[0, 1, 2])
    thalach = st.number_input("Max Heart Rate Achieved", 50, 250, 150)
    exang = st.selectbox("Exercise Induced Angina", options=[0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
    oldpeak = st.number_input("ST Depression (Oldpeak)", 0.0, 10.0, 1.0, step=0.1)
    slope = st.selectbox("ST Segment Slope", options=[0, 1, 2])
    ca = st.selectbox("Number of Major Vessels (0-4)", options=[0, 1, 2, 3, 4])
    thal = st.selectbox("Thalassemia", options=[0, 1, 2, 3], help="0: null, 1: fixed defect, 2: normal, 3: reversible defect")

# Prediction logic
if st.button("Predict Cardiac Risk"):
    model, scaler = load_assets()
    
    # Prepare input data
    input_data = np.array([[
        age, sex, cp, trestbps, chol, fbs, restecg, 
        thalach, exang, oldpeak, slope, ca, thal
    ]])
    
    # Scale input
    input_scaled = scaler.transform(input_data)
    
    # Predict
    prediction_prob = model.predict(input_scaled)[0][0]
    risk_level = "High" if prediction_prob > 0.5 else "Low"
    color = "#ff4b4b" if risk_level == "High" else "#28a745"

    # Display Result
    st.markdown("---")
    st.markdown(f"""
    <div class="result-card">
        <h2 style="color: {color};">Risk Level: {risk_level}</h2>
        <h3>Probability: {prediction_prob*100:.2f}%</h3>
    </div>
    """, unsafe_allow_html=True)
    
    if risk_level == "High":
        st.warning("The model indicates a high probability of cardiovascular disease. Please consult a medical professional.")
    else:
        st.success("The model indicates a lower probability of cardiovascular disease based on these features.")

# Footer
st.markdown("""
<div class="disclaimer">
    <strong>Medical Disclaimer:</strong> This application is for educational and research purposes only. 
    It is not a substitute for professional medical advice, diagnosis, or treatment. 
    Never disregard professional medical advice or delay in seeking it because of something you have read on this website.
    The predictions are based on a dataset and should be validated by health professionals.
</div>
""", unsafe_allow_html=True)
