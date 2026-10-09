import os

import requests
import streamlit as st

API_URL = os.getenv("API_URL", "http://127.0.0.1:8000")

st.title("Diabetes Risk Prediction")

col1, col2 = st.columns(2)
with col1:
    glucose = st.number_input("Glucose (mg/dL)", 1.0, 399.0, 120.0)
    blood_pressure = st.number_input("Blood Pressure (mm Hg)", 1.0, 249.0, 70.0)
    skin = st.number_input("Skin Thickness (mm)", 1.0, 99.0, 25.0)
    insulin = st.number_input("Insulin (mu U/ml)", 1.0, 999.0, 100.0)
with col2:
    bmi = st.number_input("BMI (kg/m²)", 1.0, 99.0, 30.0)
    dpf = st.number_input("Diabetes Pedigree Function", 0.01, 4.99, 0.5)
    age = st.number_input("Age (years)", 1.0, 119.0, 40.0)

if st.button("Predict risk"):
    patient = {
        "Glucose": glucose, "BloodPressure": blood_pressure, "SkinThickness": skin,
        "Insulin": insulin, "BMI": bmi, "DiabetesPedigreeFunction": dpf, "Age": age,
    }
    try:
        response = requests.post(f"{API_URL}/predict", json=patient, timeout=10)
        response.raise_for_status()
        risk = response.json()["risk_level"]
    except requests.exceptions.RequestException as e:
        st.error(f"Could not get a prediction: {e}")
        st.stop()

    if risk == "high":
        st.markdown(
            "<div style='background:#d62728;color:white;padding:16px;border-radius:8px;"
            "font-size:24px;text-align:center'>HIGH RISK</div>", unsafe_allow_html=True)
    else:
        st.markdown(
            "<div style='background:#2ca02c;color:white;padding:16px;border-radius:8px;"
            "font-size:24px;text-align:center'>LOW RISK</div>", unsafe_allow_html=True)

