import streamlit as st
import pandas as pd
import joblib

# -----------------------------
# Streamlit Config
# -----------------------------
st.set_page_config(page_title="Heart Disease Prediction", layout="centered")

st.title("❤️ Heart Disease Prediction App")
st.write("Enter patient information to estimate the likelihood of heart disease.")

# Load model files
preprocessor = joblib.load("preprocessor.pkl")
model = joblib.load("model.pkl")

# -----------------------------
# Input Fields
# -----------------------------
st.subheader("Patient Information")

# ---- NUMERIC FEATURES (must come first) ----
age = st.slider("Age", 20, 100, 50)
trestbps = st.slider("Resting Blood Pressure (trestbps)", 80, 200, 120)
chol = st.slider("Cholesterol (chol)", 100, 600, 250)
thalch = st.slider("Maximum Heart Rate (thalch)", 70, 220, 150)
oldpeak = st.slider("ST Depression (oldpeak)", 0.0, 6.0, 1.0, step=0.1)

# ---- CATEGORICAL FEATURES (in exact notebook order) ----

# sex
sex = 1 if st.selectbox("Sex", ["Male", "Female"]) == "Male" else 0

# cp
cp_label = st.selectbox("Chest Pain Type (cp)", [
    "typical angina",
    "atypical angina",
    "non-anginal",
    "asymptomatic"
])
cp_map = {
    "typical angina": 0,
    "atypical angina": 1,
    "non-anginal": 2,
    "asymptomatic": 3
}
cp = cp_map[cp_label]

# restecg
restecg_label = st.selectbox("Resting ECG (restecg)", [
    "normal",
    "st-t abnormality",
    "lv hypertrophy"
])
restecg_map = {"normal": 0, "st-t abnormality": 1, "lv hypertrophy": 2}
restecg = restecg_map[restecg_label]

# slope
slope_label = st.selectbox("Slope", ["upsloping", "flat", "downsloping"])
slope_map = {"upsloping": 0, "flat": 1, "downsloping": 2}
slope = slope_map[slope_label]

# thal
thal_label = st.selectbox("Thal", ["normal", "fixed defect", "reversable defect"])
thal_map = {"normal": 0, "fixed defect": 1, "reversable defect": 2}
thal = thal_map[thal_label]

# other categorical ints
ca = st.selectbox("Number of Major Vessels (ca)", [0, 1, 2, 3])
fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl (fbs)", [0, 1])
exang = st.selectbox("Exercise-induced Angina (exang)", [0, 1])

# -----------------------------
# PREDICTION BUTTON
# -----------------------------
if st.button("Predict Heart Disease Risk"):

    # EXACT TRAINING COLUMN ORDER
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

    transformed = preprocessor.transform(input_data)
    probability = model.predict_proba(transformed)[0][1]
    prediction = model.predict(transformed)[0]

    st.subheader("🩺 Prediction Result")
    st.write(f"**Estimated Probability of Heart Disease: {probability*100:.2f}%**")

    if prediction == 1:
        st.error("⚠️ High Risk — The patient is likely to have heart disease.")
    else:
        st.success("✅ Low Risk — The patient is unlikely to have heart disease.")

# -----------------------------
# Footer
# -----------------------------
st.write("---")
st.caption("Machine Learning model based on UCI Heart Disease dataset.")
