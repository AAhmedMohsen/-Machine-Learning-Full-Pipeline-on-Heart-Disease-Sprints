import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns 
import matplotlib.pyplot as plt
import joblib


model = joblib.load("heart_disease_svm_pipeline.pkl")

st.title(" Heart Disease Prediction App")
st.write("Enter your health data below to predict the likelihood of heart disease.")


with st.form("input_form"):
    age = st.number_input("Age", min_value=20, max_value=100, value=50)
    sex = st.selectbox("Sex", options=["Male", "Female"])
    cp = st.selectbox("Chest Pain Type", options=[0, 1, 2, 3])
    trestbps = st.number_input("Resting Blood Pressure (mm Hg)", min_value=80, max_value=200, value=120)
    chol = st.number_input("Cholesterol (mg/dl)", min_value=100, max_value=600, value=200)
    fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl", options=[0, 1])
    restecg = st.selectbox("Resting ECG Results", options=[0, 1, 2])
    thalach = st.number_input("Max Heart Rate Achieved", min_value=60, max_value=220, value=150)
    exang = st.selectbox("Exercise Induced Angina", options=[0, 1])
    oldpeak = st.number_input("ST Depression Induced by Exercise", min_value=0.0, max_value=10.0, value=1.0, step=0.1)
    slope = st.selectbox("Slope of Peak Exercise ST Segment", options=[0, 1, 2])
    ca = st.number_input("Number of Major Vessels (0-3)", min_value=0, max_value=3, value=0)
    thal = st.selectbox("Thalassemia", options=[0, 1, 2, 3])

    submitted = st.form_submit_button("Predict")

if submitted:
    # Prepare input for model
    input_df = pd.DataFrame([[age, 1 if sex == "Male" else 0, cp, trestbps, chol, fbs,
                              restecg, thalach, exang, oldpeak, slope, ca, thal]],
                            columns=["age", "sex", "cp", "trestbps", "chol", "fbs",
                                     "restecg", "thalch", "exang", "oldpeak", "slope", "ca", "thal"])
    
    prediction = model.predict(input_df)[0]
    prob = model.predict_proba(input_df)[0][1] * 100

    st.subheader("Prediction Result:")
    if prediction == 1:
        st.error(f"⚠ High Risk of Heart Disease ({prob:.2f}% probability)")
    else:
        st.success(f"✅ Low Risk of Heart Disease ({prob:.2f}% probability)")
