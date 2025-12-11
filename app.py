import streamlit as st
import numpy as np
import pandas as pd
import tensorflow as tf

# ======================
# Load Saved Models
# ======================
cnn_lstm_model = tf.keras.models.load_model("cnn_lstm_model.h5", compile=False)
lstm_model = tf.keras.models.load_model("lstm_model.h5", compile=False)


st.set_page_config(page_title="Diabetes Prediction", layout="centered")

st.title("🔮 Diabetes Prediction App")
st.write("This app predicts diabetes using CNN-LSTM and LSTM AI models.")

st.subheader("📥 Enter Patient Details")

# ======================
# INPUT FIELDS
# ======================
preg = st.number_input("Pregnancies", 0, 20, 2)
glucose = st.number_input("Glucose Level", 50, 250, 120)
bp = st.number_input("Blood Pressure", 40, 150, 70)
skin = st.number_input("Skin Thickness", 5, 99, 20)
insulin = st.number_input("Insulin", 0, 900, 80)
bmi = st.number_input("BMI", 10.0, 70.0, 25.0)
pedigree = st.number_input("Diabetes Pedigree Function", 0.0, 3.0, 0.5)
age = st.number_input("Age", 10, 100, 30)

# Convert input to array
user_data = np.array([[preg, glucose, bp, skin, insulin, bmi, pedigree, age]], dtype=np.float32)

# ================
# Generate 100-step synthetic sequence for CNN-LSTM (same method used in training)
# ================
def gen_seq(row, T=100):
    base_hr = 60 + (row[5] - 20)*0.5 + (row[7] - 30)*0.2
    risk_factor = (row[1] / 180.0) + (row[0] / 10.0)

    t = np.linspace(0, 6*np.pi, T)
    hr = base_hr + np.sin(t)*(2+3*risk_factor) + np.random.normal(0,3+2*risk_factor,T)
    hr = np.clip(hr, 40, 200)

    act = np.abs(np.random.normal(0.2,0.5,T)) + (np.sin(np.linspace(0,4*np.pi,T))*0.5 + 1.0)
    act = np.clip(act, 0, 10)

    return np.stack([hr, act], axis=1)

sequence = gen_seq(user_data[0])
sequence = np.expand_dims(sequence, axis=0)  # shape (1,100,2)

# ======================
# Predict
# ======================
if st.button("🔍 Predict Diabetes"):
    cnn_prob = cnn_lstm_model.predict([sequence, user_data])[0][0]
    lstm_prob = lstm_model.predict(sequence)[0][0]

    st.subheader("📊 Results")
    st.write(f"**CNN-LSTM Probability:** `{cnn_prob:.3f}`")
    st.write(f"**LSTM Probability:** `{lstm_prob:.3f}`")

    final_prob = (cnn_prob + lstm_prob) / 2
    st.write(f"**Final Combined Score:** `{final_prob:.3f}`")

    if final_prob > 0.5:
        st.error("🚨 High Risk of Diabetes")
    else:
        st.success("✅ Low Risk of Diabetes")
