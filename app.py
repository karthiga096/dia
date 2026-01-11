import streamlit as st
import pickle
import numpy as np
import os

# ---------------------------
# Load the Diabetes model
# ---------------------------
MODEL_PATH = "diabetes_model.pkl"  # Place your .pkl here (same folder as app.py)

if os.path.exists(MODEL_PATH):
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
else:
    st.error(f"Model file not found! Please upload {MODEL_PATH} in the project folder.")
    st.stop()

# ---------------------------
# Streamlit UI
# ---------------------------
st.set_page_config(page_title="Diabetes Prediction App", layout="centered")
st.title("🍀 Diabetes Prediction App")
st.write("Predict Diabetes easily using user inputs!")

st.subheader("Enter your details:")

# ---------------------------
# Input fields (based on Pima Indians Diabetes Dataset)
# ---------------------------
pregnancies = st.number_input("Pregnancies", 0, 20, 0)
glucose = st.number_input("Glucose Level", 0, 200, 120)
blood_pressure = st.number_input("Blood Pressure (mm Hg)", 0, 140, 70)
skin_thickness = st.number_input("Skin Thickness (mm)", 0, 100, 20)
insulin = st.number_input("Insulin Level (mu U/ml)", 0.0, 900.0, 79.0)
bmi = st.number_input("BMI", 0.0, 70.0, 20.0)
dpf = st.number_input("Diabetes Pedigree Function", 0.0, 2.5, 0.5)
age = st.number_input("Age", 1, 120, 30)

# ---------------------------
# Prepare input for model
# ---------------------------
user_input = np.array([[pregnancies, glucose, blood_pressure, skin_thickness,
                        insulin, bmi, dpf, age]])

# ---------------------------
# Predict button
# ---------------------------
if st.button("Predict"):
    prediction = model.predict(user_input)[0]

    if prediction == 0:
        st.success("No Diabetes detected ✅")
    else:
        st.error("Diabetes detected ⚠️")

st.markdown("---")
st.write("App developed with Streamlit and Python 🍀")
