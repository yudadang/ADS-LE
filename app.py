import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt

# Try seaborn
try:
    import seaborn as sns
except:
    sns = None
    st.warning("⚠️ Seaborn is not installed. Some visualizations will be limited.")

# Plotly fallback for interactive charts
try:
    import plotly.express as px
except:
    px = None


# ================================
# PAGE CONFIG
# ================================
st.set_page_config(
    page_title="Heart Disease Dashboard",
    layout="wide",
    page_icon="❤️"
)

# Load model components
preprocessor = joblib.load("preprocessor.pkl")
model = joblib.load("model.pkl")

df = pd.read_csv("heart_disease_uci.csv")

# ================================
# SIDEBAR
# ================================
st.sidebar.title("📌 Navigation")
page = st.sidebar.radio("", ["📊 Dashboard", "📈 Model Metrics", "🩺 Prediction"])

st.sidebar.write("---")
st.sidebar.info("Built for Applied Data Science 🧠")
st.sidebar.write("GitHub: yudadang/ADS-LE")


# ================================
# 📊 DASHBOARD PAGE
# ================================
if page == "📊 Dashboard":
    st.title("❤️ Heart Disease Analysis Dashboard")

    # ================================
    # Summary Cards
    # ================================
    st.subheader("📘 Summary Statistics")
    col1, col2, col3 = st.columns(3)

    col1.metric("Total Patients", len(df))
    col2.metric("Average Age", round(df["age"].mean(), 1))
    col3.metric("Heart Disease (%)", f"{round((df['num']>0).mean()*100,1)}%")

    st.write("---")

    # ================================
    # Dataset Preview
    # ================================
    with st.expander("📌 Dataset Preview"):
        st.dataframe(df.head())

    # ================================
    # Charts Section
    # ================================
    st.subheader("📈 Visual Insights")

    # TARGET DISTRIBUTION
    st.markdown("### ❤️ Heart Disease Distribution")
    if px:
        fig = px.histogram(df, x=(df["num"]>0), color=(df["num"]>0),
                           labels={'x':'Disease Presence'},
                           color_discrete_sequence=["green", "red"])
        fig.update_layout(xaxis_ticktext=["No Disease", "Disease"],
                          xaxis_tickvals=[0,1])
        st.plotly_chart(fig, use_container_width=True)
    elif sns:
        fig, ax = plt.subplots()
        sns.countplot(data=df, x=(df["num"] > 0), ax=ax)
        ax.set_xticklabels(["No Disease", "Disease"])
        st.pyplot(fig)

    st.write("---")

    # AGE DISTRIBUTION
    st.markdown("### 👥 Age Distribution")
    if px:
        fig = px.histogram(df, x="age", nbins=30, marginal="box", color_discrete_sequence=["blue"])
        st.plotly_chart(fig, use_container_width=True)
    elif sns:
        fig, ax = plt.subplots()
        sns.histplot(df["age"], kde=True, ax=ax)
        st.pyplot(fig)

    st.write("---")

    # HEATMAP
    st.markdown("### 🔥 Correlation Heatmap")
    numeric_df = df.select_dtypes(include=['int64','float64'])
    if px:
        fig = px.imshow(numeric_df.corr(), color_continuous_scale="RdBu_r")
        st.plotly_chart(fig, use_container_width=True)
    elif sns:
        fig, ax = plt.subplots(figsize=(10,6))
        sns.heatmap(numeric_df.corr(), cmap="coolwarm", annot=False, ax=ax)
        st.pyplot(fig)

    st.write("---")

    # FEATURE IMPORTANCE
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
        fig = px.bar(
            importance_df.head(10),
            x="Importance", y="Feature",
            color="Importance", orientation="h",
            color_continuous_scale="Inferno"
        )
        st.plotly_chart(fig, use_container_width=True)


# ================================
# 📈 MODEL METRICS PAGE
# ================================
elif page == "📈 Model Metrics":
    st.title("📈 Model Performance Metrics")

    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

    # Compute test predictions
    X = df.drop(columns=["id", "dataset", "num"])
    X["target"] = (df["num"] > 0).astype(int)
    y = X["target"]
    X = X.drop("target", axis=1)

    X_transformed = preprocessor.transform(X)
    y_pred = model.predict(X_transformed)
    y_proba = model.predict_proba(X_transformed)[:,1]

    # Display metrics
    st.metric("Accuracy", round(accuracy_score(y, y_pred), 3))
    st.metric("Precision", round(precision_score(y, y_pred), 3))
    st.metric("Recall", round(recall_score(y, y_pred), 3))
    st.metric("F1 Score", round(f1_score(y, y_pred), 3))
    st.metric("ROC AUC", round(roc_auc_score(y, y_proba), 3))

    st.write("---")

    # ROC Curve
    try:
        from sklearn.metrics import RocCurveDisplay
        fig, ax = plt.subplots()
        RocCurveDisplay.from_predictions(y, y_proba, ax=ax)
        st.pyplot(fig)
    except:
        st.info("Plotly/Matplotlib ROC display unavailable in this environment.")

    st.write("---")

    st.subheader("💡 Interpretation Notes")
    st.info("""
    • **Higher precision** → fewer false positives  
    • **Higher recall** → fewer missed disease cases  
    • **AUC close to 1.0** → strong predictive ability  
    """)


# ================================
# 🩺 PREDICTION PAGE
# ================================
elif page == "🩺 Prediction":
    st.title("🩺 Heart Disease Prediction Tool")
    st.write("Enter patient information to estimate heart disease risk.")

    # ---------------- INPUTS ----------------
    age = st.slider("Age", 20, 100, 50)
    trestbps = st.slider("Resting Blood Pressure", 80, 200, 120)
    chol = st.slider("Cholesterol", 100, 600, 250)
    thalch = st.slider("Max Heart Rate", 70, 220, 150)
    oldpeak = st.slider("ST Depression", 0.0, 6.0, 1.0)

    sex = 1 if st.selectbox("Sex", ["Male", "Female"]) == "Male" else 0
    cp_map = {"typical angina":0, "atypical angina":1, "non-anginal":2, "asymptomatic":3}
    cp = cp_map[st.selectbox("Chest Pain Type", list(cp_map.keys()))]
    restecg_map = {"normal":0, "st-t abnormality":1, "lv hypertrophy":2}
    restecg = restecg_map[st.selectbox("Resting ECG", list(restecg_map.keys()))]
    slope_map = {"upsloping":0, "flat":1, "downsloping":2}
    slope = slope_map[st.selectbox("Slope", list(slope_map.keys()))]
    thal_map = {"normal":0, "fixed defect":1, "reversable defect":2}
    thal = thal_map[st.selectbox("Thal", list(thal_map.keys()))]

    ca = st.selectbox("Major Vessels", [0,1,2,3])
    fbs = st.selectbox("Fasting Blood Sugar >120", [0,1])
    exang = st.selectbox("Exercise-induced Angina", [0,1])

    if st.button("Predict"):
        input_data = pd.DataFrame([{
            "age": age, "trestbps": trestbps, "chol": chol,
            "thalch": thalch, "oldpeak": oldpeak, "sex": sex, "cp": cp,
            "fbs": fbs, "restecg": restecg, "exang": exang, "slope": slope,
            "ca": ca, "thal": thal
        }])

        transformed = preprocessor.transform(input_data)
        probability = model.predict_proba(transformed)[0][1]
        prediction = model.predict(transformed)[0]

        st.subheader("🔬 Prediction Result")
        st.write(f"**Heart Disease Probability: {probability*100:.2f}%**")

        # gauge-style indicator
        st.progress(int(probability*100))

        if prediction == 1:
            st.error("⚠️ HIGH RISK — consult a doctor immediately.")
        else:
            st.success("✅ LOW RISK — condition appears normal.")

st.write("---")
st.caption("Machine Learning Dashboard using UCI Heart Disease Dataset ❤️")
