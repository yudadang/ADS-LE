import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import joblib

st.set_page_config(page_title="Heart Disease Dashboard", layout="wide")

st.title("❤️ Heart Disease Analysis & Prediction App")

# Load model
model = joblib.load("model.pkl")

# Load dataset for charts
df = pd.read_csv("heart_disease_uci.csv")

# Sidebar Navigation
page = st.sidebar.selectbox("Navigation", ["📊 Dashboard", "🩺 Prediction"])


# ============================================================
# 📊 DASHBOARD PAGE
# ============================================================
if page == "📊 Dashboard":
    st.header("📈 Heart Disease Dataset Overview")

    # Summary metrics
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Patients", len(df))
    col2.metric("Average Age", round(df["age"].mean(), 1))
    col3.metric("Heart Disease Cases (%)", 
        f"{round((df['num'] > 0).mean()*100, 1)}%"
    )

    st.subheader("🔍 Dataset Preview")
    st.dataframe(df.head())

    # Target distribution
    st.subheader("❤️ Heart Disease Distribution")
    fig, ax = plt.subplots()
    sns.countplot(data=df, x=(df["num"] > 0), ax=ax)
    ax.set_xticklabels(["No Disease", "Disease"])
    st.pyplot(fig)

    # Age distribution
    st.subheader("📌 Age Distribution")
    fig, ax = plt.subplots()
    sns.histplot(df["age"], kde=True, ax=ax)
    st.pyplot(fig)

    # Correlation heatmap
    st.subheader("📉 Correlation Heatmap")
    numeric_df = df.select_dtypes(include=['int64','float64'])
    fig, ax = plt.subplots(figsize=(10,6))
    sns.heatmap(numeric_df.corr(), cmap="coolwarm", annot=False, ax=ax)
    st.pyplot(fig)

    # Feature importances (from model)
    st.subheader("🔥 Feature Importances (Model)")

    rf = model.named_steps["rf"]
    pre = model.named_steps["preprocess"]

    ohe = pre.named_transformers_["cat"]["encoder"]
    ohe_cols = list(ohe.get_feature_names_out(
        ["sex", "cp", "restecg", "slope", "thal", "ca", "fbs", "exang"]
    ))

    all_features = ["age", "trestbps", "chol", "thalch", "oldpeak"] + ohe_cols
    importances = rf.feature_importances_

    importance_df = pd.DataFrame({
        "Feature": all_features,
        "Importance": importances
    }).sort_values("Importance", ascending=False)

    st.dataframe(importance_df)

    fig, ax = plt.subplots(figsize=(10,5))
    sns.barplot(data=importance_df.head(10), x="Importance", y="Feature", ax=ax)
    st.pyplot(fig)




# ============================================================
# 🩺 PREDICTION PAGE
# ============================================================
elif page == "🩺 Prediction":

    st.header("🩺 Heart Disease Prediction")

    st.write("Enter patient information to estimate heart disease risk:")

    # ------------ Input Fields -------------
    age = st.slider("Age", 20, 100, 50)
    trestbps = st.slider("Resting Blood Pressure", 80, 200, 120)
    chol = st.slider("Cholesterol", 100, 600, 250)
    thalch = st.slider("Max Heart Rate", 70, 220, 150)
    oldpeak = st.slider("ST Depression", 0.0, 6.0, 1.0, step=0.1)

    sex = st.selectbox("Sex", ["Male", "Female"])
    sex = 1 if sex == "Male" else 0

    cp_map = {"typical angina":0, "atypical angina":1, "non-anginal":2, "asymptomatic":3}
    cp = cp_map[st.selectbox("Chest Pain Type", list(cp_map.keys()))]

    restecg_map = {"normal":0, "st-t abnormality":1, "lv hypertrophy":2}
    restecg = restecg_map[st.selectbox("Resting ECG", list(restecg_map.keys()))]

    slope_map = {"upsloping":0, "flat":1, "downsloping":2}
    slope = slope_map[st.selectbox("Slope", list(slope_map.keys()))]

    thal_map = {"normal":0, "fixed defect":1, "reversable defect":2}
    thal = thal_map[st.selectbox("Thal", list(thal_map.keys()))]

    ca = st.selectbox("Major Vessels (ca)", [0, 1, 2, 3])
    fbs = st.selectbox("Fasting Blood Sugar > 120", [0, 1])
    exang = st.selectbox("Exercise-induced Angina", [0, 1])

    if st.button("Predict"):

        input_data = pd.DataFrame([{
            "age": age,
            "trestbps": trestbps,
            "chol": chol,
            "thalch": thalch,
            "oldpeak": oldpeak,
            "sex": sex,
            "cp": cp,
            "fbs": fbs,
            "restecg": restecg,
            "exang": exang,
            "slope": slope,
            "ca": ca,
            "thal": thal
        }])

        probability = model.predict_proba(input_data)[0][1]
        prediction = model.predict(input_data)[0]

        st.subheader("🔬 Prediction Result")
        st.write(f"**Heart Disease Probability: {probability*100:.2f}%**")

        if prediction == 1:
            st.error("⚠️ High risk of heart disease detected!")
        else:
            st.success("✅ Low risk of heart disease detected.")

# Footer
st.write("---")
st.caption("Machine Learning Dashboard using UCI Heart Disease Dataset")
