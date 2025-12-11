import streamlit as st
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import load_model
import requests
import io

st.set_page_config(page_title="Diabetes Predictor", layout="centered")

st.title("🩺 Diabetes Prediction App (CNN-LSTM + LSTM Models)")
st.write("Enter your values below to get prediction.")

# -------------------------
# DOWNLOAD MODELS FROM GITHUB RELEASE
# -------------------------
@st.cache_resource
def load_models():
    cnn_url = "https://github.com/keshu76/diabetes-predictor/releases/download/v1.0/cnn_lstm_model.h5"
    lstm_url = "https://github.com/keshu76/diabetes-predictor/releases/download/v1.0/lstm_model.h5"

    # Download CNN-LSTM
    cnn_bytes = io.BytesIO(requests.get(cnn_url).content)
    cnn_model = load_model(cnn_bytes)

    # Download LSTM
    lstm_bytes = io.BytesIO(requests.get(lstm_url).content)
    lstm_model = load_model(lstm_bytes)

    return cnn_model, lstm_model


cnn_model, lstm_model = load_models()

st.success("Models loaded successfully ✔")

# -------------------------
# INPUT FORM
# -------------------------

preg = st.number_input("Pregnancies", 0, 20, 2)
glucose = st.number_input("Glucose Level", 50, 250, 120)
bp = st.number_input("Blood Pressure", 40, 150, 70)
skin = st.number_input("Skin Thickness", 5, 99, 20)
insulin = st.number_input("Insulin Level", 0, 900, 80)
bmi = st.number_input("BMI", 10.0, 60.0, 25.0)
pedigree = st.number_input("Diabetes Pedigree", 0.0, 5.0, 0.5)
age = st.number_input("Age", 10, 100, 30)

if st.button("Predict"):
    X_tab = np.array([[preg, glucose, bp, skin, insulin, bmi, pedigree, age]])

    # Simple LSTM prediction (no sequence needed)
    pred = lstm_model.predict(X_tab)[0][0]

    st.subheader("🔍 Prediction Result")
    if pred > 0.5:
        st.error("⚠ High chance of Diabetes")
    else:
        st.success("✔ Low chance of Diabetes")
