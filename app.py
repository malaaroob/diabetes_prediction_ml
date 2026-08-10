import streamlit as st
import joblib
import pandas as pd

model = joblib.load('diabetes_model.pkl')
scaler = joblib.load('scaler.pkl')
le = joblib.load('label_encoder.pkl')
columns = joblib.load('columns.pkl')
st.title("Diabetes Prediction App")

age = st.number_input("Age", min_value=1, max_value=120, value=30, step=1)
gender = st.selectbox("Gender", ['Male', 'Female'])
bmi = st.number_input("BMI", min_value=10.0, max_value=100.0, value=25.0, step=0.5)
systolic_bp = st.number_input("Systolic BP", min_value=50.0, max_value=250.0, value=120.0, step=1.0)
diastolic_bp = st.number_input("Diastolic BP", min_value=30.0, max_value=150.0, value=80.0, step=1.0)
hba1c = st.number_input("HbA1c Level", min_value=3.0, max_value=20.0, value=5.5, step=0.1)

if st.button("Predict"):
    gender_encoded = le.transform([gender])[0]
    input_data = pd.DataFrame([[age, gender_encoded, bmi, systolic_bp, diastolic_bp, hba1c]], columns=columns)
    input_scaled = scaler.transform(input_data)
    prediction = model.predict(input_scaled)[0]
    if prediction == 0:
        st.success("Result: No Diabetes")
    elif prediction == 1:
        st.warning("Result: Prediabetes")
    else:
        st.error("Result: Diabetes")



