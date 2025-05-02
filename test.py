import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import joblib

# عنوان التطبيق
st.title('📊 Stress Level Prediction for Individuals')
st.write("""
Enter your physiological data to predict your stress level (0-4).
""")

# تحميل النموذج والمقياس
@st.cache_resource
def load_model():
    try:
        model = joblib.load('XGB.pkl')
        scaler = joblib.load('scaler.pkl')
        return model, scaler
    except FileNotFoundError:
        st.error("Model not found. Please train the model first.")
        return None, None

model, scaler = load_model()

if model is not None:
    # إدخال البيانات يدويًا
    st.sidebar.header("Enter Your Data")

    snoring = st.sidebar.number_input("Snoring Rate")
    respiration = st.sidebar.number_input("Respiration Rate")
    temp = st.sidebar.number_input("Body Temperature ")
    limb_movement = st.sidebar.number_input("Limb Movement")
    oxygen = st.sidebar.number_input("Blood Oxygen (%)")
    eye_movement = st.sidebar.number_input("Eye Movement")
    sleep_hours = st.sidebar.number_input("Sleeping Hours")
    heart_rate = st.sidebar.number_input("Heart Rate (bpm)")

    if st.sidebar.button("Predict My Stress Level"):
        input_data = np.array([[snoring, respiration, temp, limb_movement, oxygen, eye_movement, sleep_hours, heart_rate]])
        input_scaled = scaler.transform(input_data)
        prediction = model.predict(input_scaled)[0]

        # عرض النتيجة بطريقة جذابة
        st.success(f"### 🔍 Your Predicted Stress Level: **{prediction}**")
        
        # تفسير النتيجة
        stress_levels = {
            0: "NoStress 😊",
            1: "Stress 😵"
        }
        st.write(f"**Interpretation:** {stress_levels.get(prediction, 'Unknown')}")
else:
    st.warning("Please ensure 'stress_model.pkl' and 'scaler.pkl' exist in the directory.")