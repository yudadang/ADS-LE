import streamlit as st
import pandas as pd
import joblib

# -----------------------------
# Load saved model + preprocessor
# -----------------------------
st.set_page_config(page_title="Heart Disease Prediction", layout="centered")

st.title("❤️ Heart Disease Prediction App")
st.write("Enter patient information below to estimate the likelihood of heart disease.")

# Load model files
preprocessor = joblib.load("preprocessor.pkl")
model = joblib.load("model.pkl")

# -----------------------------
# Input Fields
# -----------------------------
st.subheader("Patient Information")

age = st.slider("Age", 20, 100, 50)
trestbps = st.slider("Resting Blood Pressure (trestbps)", 80, 200, 120)
chol = st.slider("Cholesterol (chol)", 100, 600, 250)
thalch = st.slider("Maximum Heart Rate (thalch)", 70, 220, 150)
oldpeak = st.slider("ST Depression (oldpeak)", 0.0, 6.0, 1.0, step=0.1)

sex = st.selectbox("Sex", ["Male", "Female"])
cp = st.selectbox("Chest Pain Type (cp)", [
    "typical angina",
    "atypical angina",
    "non-anginal",
    "asymptomatic"
])
restecg = st.selectbox("Resting ECG", [
    "normal",
    "lv hypertrophy",
    "st-t abnormality"
])
slope = st.selectbox("Slope", ["upsloping", "flat", "downsloping"])
thal = st.selectbox("Thal", ["normal", "fixed defect", "reversable defect"])
ca = st.selectbox("Number of Major Vessels (ca)", [0, 1, 2, 3])
fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl (fbs)", [0, 1])
exang = st.selectbox("Exercise-induced Angina (exang)", [0, 1])

# -----------------------------
# Prediction Button
# -----------------------------
if st.button("Predict Heart Disease Risk"):
    
    # Prepare input data
    input_data = pd.DataFrame([{
        "age": age,
        "trestbps": trestbps,
        "chol": chol,
        "thalch": thalch,
        "oldpeak": oldpeak,
        "sex": sex,
        "cp": cp,
        "restecg": restecg,
        "slope": slope,
        "thal": thal,
        "ca": ca,
        "fbs": fbs,
        "exang": exang
    }])

    # Preprocess and predict
    transformed_data = preprocessor.transform(input_data)
    probability = model.predict_proba(transformed_data)[0][1]
    prediction = model.predict(transformed_data)[0]

    st.subheader("🩺 Prediction Result")

    st.write(f"**Estimated Probability of Heart Disease: {probability*100:.2f}%**")

    if prediction == 1:
        st.error("⚠️ High Risk — The patient is likely to have heart disease. Clinical evaluation recommended.")
    else:
        st.success("✅ Low Risk — The patient is unlikely to have heart disease.")


# -----------------------------
# Footer
# -----------------------------
st.write("---")
st.caption("Machine Learning model based on UCI Heart Disease dataset.")
