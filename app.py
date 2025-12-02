import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt

# Try seaborn
try:
    import seaborn as sns
except:
    sns = None
    st.warning("⚠️ Seaborn is not installed. Some visualizations may be limited.")

# Interactive charts fallback
try:
    import plotly.express as px
except:
    px = None


# =======================================
# PAGE CONFIG
# =======================================
st.set_page_config(
    page_title="Heart Disease Dashboard",
    page_icon="❤️",
    layout="wide"
)

st.title("❤️ Heart Disease Analysis & Prediction App")

# =======================================
# LOAD MODEL + PREPROCESSOR + DATA
# =======================================
preprocessor = joblib.load("preprocessor.pkl")
model = joblib.load("model.pkl")
df = pd.read_csv("heart_disease_uci.csv")  # CSV must be in your repo


# =======================================
# SIDEBAR NAVIGATION
# =======================================
st.sidebar.title("📌 Navigation")
page = st.sidebar.radio("", ["📊 Dashboard", "📈 Model Metrics", "🩺 Prediction"])

st.sidebar.write("---")
st.sidebar.info("Built for Applied Data Science 🧠")
st.sidebar.write("GitHub: yudadang/ADS-LE")


# =======================================
# 📊 DASHBOARD PAGE
# =======================================
if page == "📊 Dashboard":

    st.header("📈 Heart Disease Dataset Overview")

    # =========================
    # KPI Summary
    # =========================
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Patients", len(df))
    col2.metric("Average Age", round(df["age"].mean(), 1))
    col3.metric("Heart Disease (%)", f"{round((df['num']>0).mean()*100, 1)}%")

    st.write("---")

    # =========================
    # Dataset Preview
    # =========================
    with st.expander("📌 View Dataset"):
        st.dataframe(df.head())

    # =========================
    # Target Distribution
    # =========================
    st.markdown("### ❤️ Heart Disease Distribution")

    if px:
        fig = px.histogram(df, x=(df["num"] > 0),
                           color=(df["num"] > 0),
                           labels={'x': 'Disease Presence'},
                           color_discrete_sequence=["green", "red"])
        fig.update_layout(xaxis_ticktext=["No Disease", "Disease"],
                          xaxis_tickvals=[0, 1])
        st.plotly_chart(fig, use_container_width=True)
    else:
        fig, ax = plt.subplots()
        sns.countplot(data=df, x=(df["num"] > 0), ax=ax)
        ax.set_xticklabels(["No Disease", "Disease"])
        st.pyplot(fig)

    st.write("---")

    # =========================
    # Age Distribution
    # =========================
    st.markdown("### 👥 Age Distribution")

    if px:
        fig = px.histogram(df, x="age", nbins=30, marginal="box",
                           color_discrete_sequence=["blue"])
        st.plotly_chart(fig, use_container_width=True)
    else:
        fig, ax = plt.subplots()
        sns.histplot(df["age"], kde=True, ax=ax)
        st.pyplot(fig)

    st.write("---")

    # =========================
    # Correlation Heatmap
    # =========================
    st.markdown("### 🔥 Correlation Heatmap")
    numeric_df = df.select_dtypes(include=['int64', 'float64'])

    if px:
        fig = px.imshow(numeric_df.corr(), color_continuous_scale="RdBu_r")
        st.plotly_chart(fig, use_container_width=True)
    else:
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.heatmap(numeric_df.corr(), cmap="coolwarm", ax=ax)
        st.pyplot(fig)

    st.write("---")

    # =========================
    # Feature Importances
    # =========================
    st.subheader("✨ Feature Importances (Model)")

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

    if px:
        fig = px.bar(importance_df.head(10),
                     x="Importance", y="Feature",
                     orientation="h",
                     color="Importance",
                     color_continuous_scale="Inferno")
        st.plotly_chart(fig, use_container_width=True)

# =======================================
# 📈 MODEL METRICS PAGE
# =======================================
elif page == "📈 Model Metrics":

    st.title("📈 Model Performance Metrics")

    # Clean dataset EXACTLY like training
    df_metrics = df.copy()
    df_metrics["target"] = (df_metrics["num"] > 0).astype(int)

    X = df_metrics[[
        "age", "sex", "cp", "trestbps", "chol",
        "fbs", "restecg", "thalch", "exang", "oldpeak",
        "slope", "ca", "thal"
    ]]

    y = df_metrics["target"]

    # ❗ NO PREPROCESSOR HERE - model already contains it
    y_pred = model.predict(X)
    y_proba = model.predict_proba(X)[:, 1]

    # Metrics Display
    from sklearn.metrics import (
        accuracy_score, precision_score, recall_score,
        f1_score, roc_auc_score
    )

    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Accuracy", round(accuracy_score(y, y_pred), 3))
    col2.metric("Precision", round(precision_score(y, y_pred), 3))
    col3.metric("Recall", round(recall_score(y, y_pred), 3))
    col4.metric("F1 Score", round(f1_score(y, y_pred), 3))
    col5.metric("ROC AUC", round(roc_auc_score(y, y_proba), 3))

    st.write("---")

    # ROC Curve
    try:
        from sklearn.metrics import RocCurveDisplay
        fig, ax = plt.subplots()
        RocCurveDisplay.from_predictions(y, y_proba, ax=ax)
        st.pyplot(fig)
    except:
        st.info("⚠️ ROC Curve could not be displayed.")

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

        # 🚫 NO MORE TRANSFORM — model is already a full pipeline
        probability = model.predict_proba(input_data)[0][1]
        prediction = model.predict(input_data)[0]

        st.subheader("🔬 Prediction Result")
        st.write(f"**Heart Disease Probability: {probability*100:.2f}%**")

        if prediction == 1:
            st.error("⚠️ High risk of heart disease detected!")
        else:
            st.success("✅ Low risk of heart disease detected.")

    # -----------------------------------
    # Predict Button
    # -----------------------------------
    if st.button("Predict Risk"):

        input_data = pd.DataFrame([{
            "age": age, "trestbps": trestbps, "chol": chol,
            "thalch": thalch, "oldpeak": oldpeak,
            "sex": sex, "cp": cp, "fbs": fbs, "restecg": restecg,
            "exang": exang, "slope": slope, "ca": ca, "thal": thal
        }])

        transformed = preprocessor.transform(input_data)
        probability = model.predict_proba(transformed)[0][1]
        prediction = model.predict(transformed)[0]

        st.subheader("🔬 Prediction Result")
        st.write(f"**Estimated Heart Disease Risk: {probability*100:.2f}%**")

        # Progress bar risk meter
        st.progress(int(probability * 100))

        if prediction == 1:
            st.error("⚠️ HIGH RISK — Possible heart disease detected.")
        else:
            st.success("✅ LOW RISK — Heart condition appears normal.")


# =======================================
st.write("---")
st.caption("Machine Learning Dashboard using UCI Heart Disease Dataset ❤️")


