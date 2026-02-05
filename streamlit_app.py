"""
CardioScreen - Cardiovascular Disease Risk Prediction
Streamlit Web Application

Author: Matthias
Model: Gradient Boosting (scikit-learn), tuned via RandomizedSearchCV
"""

import streamlit as st
import numpy as np
import pandas as pd
import joblib
import os

# ─────────────────────────── Page Configuration ───────────────────────────
st.set_page_config(
    page_title="CardioScreen - CVD Risk Predictor",
    page_icon="❤️",
    layout="centered",
)

# ─────────────────────────── Load Model ───────────────────────────
@st.cache_resource
def load_model():
    model_path = os.path.join(os.path.dirname(__file__), "cardio_pipeline.pkl")
    features_path = os.path.join(os.path.dirname(__file__), "feature_names.pkl")
    pipeline = joblib.load(model_path)
    feature_names = joblib.load(features_path)
    return pipeline, feature_names

try:
    pipeline, feature_names = load_model()
    model_loaded = True
except Exception as e:
    model_loaded = False
    st.error(f"⚠️ Could not load model: {e}")

# ─────────────────────────── Header ───────────────────────────
st.title("❤️ CardioScreen")
st.subheader("Cardiovascular Disease Risk Prediction")
st.markdown(
    "Enter your health screening data below to receive an **instant CVD risk estimate**. "
    "This tool is designed for primary-care clinics and health-screening programmes."
)
st.divider()

# ─────────────────────────── Sidebar Inputs ───────────────────────────
st.sidebar.header("🩺 Patient Information")

age = st.sidebar.slider("Age (years)", min_value=30, max_value=80, value=50, step=1)
gender = st.sidebar.selectbox("Gender", options=[1, 2], format_func=lambda x: "Female" if x == 1 else "Male")

st.sidebar.header("📏 Body Measurements")
height = st.sidebar.slider("Height (cm)", min_value=120, max_value=220, value=165, step=1)
weight = st.sidebar.slider("Weight (kg)", min_value=30, max_value=200, value=70, step=1)

st.sidebar.header("🩸 Blood Pressure")
ap_hi = st.sidebar.slider("Systolic BP (mmHg)", min_value=60, max_value=250, value=120, step=1)
ap_lo = st.sidebar.slider("Diastolic BP (mmHg)", min_value=30, max_value=150, value=80, step=1)

# Validate BP
if ap_lo >= ap_hi:
    st.sidebar.error("⚠️ Diastolic BP must be lower than Systolic BP. Please correct.")

st.sidebar.header("🔬 Lab Results")
cholesterol = st.sidebar.selectbox(
    "Cholesterol Level",
    options=[1, 2, 3],
    format_func=lambda x: {1: "Normal", 2: "Above Normal", 3: "Well Above Normal"}[x]
)
gluc = st.sidebar.selectbox(
    "Glucose Level",
    options=[1, 2, 3],
    format_func=lambda x: {1: "Normal", 2: "Above Normal", 3: "Well Above Normal"}[x]
)

st.sidebar.header("🚬 Lifestyle")
smoke = st.sidebar.selectbox("Smoking", options=[0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
alco = st.sidebar.selectbox("Alcohol Intake", options=[0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
active = st.sidebar.selectbox("Physically Active", options=[0, 1], format_func=lambda x: "No" if x == 0 else "Yes")

# ─────────────────────────── Feature Engineering ───────────────────────────
bmi = round(weight / ((height / 100) ** 2), 1)
pulse_pressure = ap_hi - ap_lo
high_bp_flag = int(ap_hi >= 140 or ap_lo >= 90)
map_val = round(ap_lo + (pulse_pressure / 3), 1)

# ─────────────────────────── Display Patient Summary ───────────────────────────
col1, col2, col3 = st.columns(3)
with col1:
    st.metric("BMI", f"{bmi:.1f}")
with col2:
    st.metric("Pulse Pressure", f"{pulse_pressure} mmHg")
with col3:
    st.metric("MAP", f"{map_val:.1f} mmHg")

st.divider()

# ─────────────────────────── Prediction ───────────────────────────
if model_loaded and ap_lo < ap_hi:
    # Build input DataFrame matching training feature order
    input_data = pd.DataFrame([{
        'age': age,
        'gender': gender,
        'height': height,
        'weight': weight,
        'ap_hi': ap_hi,
        'ap_lo': ap_lo,
        'cholesterol': cholesterol,
        'gluc': gluc,
        'smoke': smoke,
        'alco': alco,
        'active': active,
        'bmi': bmi,
        'pulse_pressure': pulse_pressure,
        'high_bp_flag': high_bp_flag,
        'map': map_val,
    }])

    # Ensure column order matches training
    input_data = input_data[feature_names]

    prediction = pipeline.predict(input_data)[0]
    probability = pipeline.predict_proba(input_data)[0]

    prob_cvd = probability[1] * 100
    prob_no_cvd = probability[0] * 100

    # ─────────────────────── Result Display ───────────────────────
    if prediction == 1:
        st.error(f"## 🔴 HIGH RISK — CVD Detected")
        st.markdown(f"**Probability of CVD:** {prob_cvd:.1f}%")
        st.progress(prob_cvd / 100)
        st.markdown(
            "**Interpretation:** Based on the provided health data, this patient has a "
            f"**{prob_cvd:.1f}% estimated probability** of having cardiovascular disease. "
            "We recommend follow-up with a cardiologist for further evaluation including "
            "ECG, lipid panel, and cardiac imaging."
        )
    else:
        st.success(f"## 🟢 LOW RISK — No CVD Detected")
        st.markdown(f"**Probability of No CVD:** {prob_no_cvd:.1f}%")
        st.progress(prob_no_cvd / 100)
        st.markdown(
            "**Interpretation:** Based on the provided health data, this patient has a "
            f"**{prob_no_cvd:.1f}% estimated probability** of being free from cardiovascular disease. "
            "Continue routine health monitoring and maintain a healthy lifestyle."
        )

    # Risk factor breakdown
    st.divider()
    st.subheader("📊 Key Risk Indicators")
    risk_col1, risk_col2 = st.columns(2)
    with risk_col1:
        bp_status = "⚠️ High" if high_bp_flag else "✅ Normal"
        bmi_status = "⚠️ Overweight/Obese" if bmi >= 25 else "✅ Normal"
        st.markdown(f"- **Blood Pressure:** {bp_status}")
        st.markdown(f"- **BMI Category:** {bmi_status}")
        st.markdown(f"- **Cholesterol:** {'⚠️ Elevated' if cholesterol > 1 else '✅ Normal'}")
    with risk_col2:
        st.markdown(f"- **Glucose:** {'⚠️ Elevated' if gluc > 1 else '✅ Normal'}")
        st.markdown(f"- **Smoking:** {'⚠️ Smoker' if smoke else '✅ Non-smoker'}")
        st.markdown(f"- **Physical Activity:** {'✅ Active' if active else '⚠️ Inactive'}")

elif not model_loaded:
    st.warning("Model could not be loaded. Please ensure `cardio_pipeline.pkl` and `feature_names.pkl` exist.")

# ─────────────────────────── Disclaimer ───────────────────────────
st.divider()
st.caption(
    "⚕️ **Medical Disclaimer:** This tool is for screening purposes only and does NOT replace "
    "professional medical advice, diagnosis, or treatment. Always consult a qualified healthcare "
    "provider for medical decisions. Model accuracy is approximately 73-74% and should be used "
    "as one input among many clinical considerations."
)
st.caption("Built with Streamlit | Model: Gradient Boosting (scikit-learn) | Dataset: Kaggle Cardiovascular Disease")
