import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# ─────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────
st.set_page_config(
    page_title="Customer Churn Predictor",
    page_icon="📊",
    layout="wide"
)

# ─────────────────────────────────────────
# LOAD MODEL
# ─────────────────────────────────────────
@st.cache_resource
def load_model():
    model = joblib.load('models/xgboost_model.pkl')
    feature_names = joblib.load('models/feature_names.pkl')
    return model, feature_names

model, feature_names = load_model()

# ─────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────
st.title("📊 Customer Churn Prediction System")
st.markdown("Enter customer details to predict whether they are likely to churn.")
st.markdown("---")

# ─────────────────────────────────────────
# SIDEBAR — INPUT FORM
# ─────────────────────────────────────────
st.sidebar.header("Customer Details")

def user_input():
    gender = st.sidebar.selectbox("Gender", ["Male", "Female"])
    senior = st.sidebar.selectbox("Senior Citizen", ["No", "Yes"])
    partner = st.sidebar.selectbox("Has Partner", ["Yes", "No"])
    dependents = st.sidebar.selectbox("Has Dependents", ["Yes", "No"])
    tenure = st.sidebar.slider("Tenure (months)", 0, 72, 12)
    phone_service = st.sidebar.selectbox("Phone Service", ["Yes", "No"])
    multiple_lines = st.sidebar.selectbox("Multiple Lines", ["No", "Yes", "No phone service"])
    internet_service = st.sidebar.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
    online_security = st.sidebar.selectbox("Online Security", ["Yes", "No", "No internet service"])
    online_backup = st.sidebar.selectbox("Online Backup", ["Yes", "No", "No internet service"])
    device_protection = st.sidebar.selectbox("Device Protection", ["Yes", "No", "No internet service"])
    tech_support = st.sidebar.selectbox("Tech Support", ["Yes", "No", "No internet service"])
    streaming_tv = st.sidebar.selectbox("Streaming TV", ["Yes", "No", "No internet service"])
    streaming_movies = st.sidebar.selectbox("Streaming Movies", ["Yes", "No", "No internet service"])
    contract = st.sidebar.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
    paperless_billing = st.sidebar.selectbox("Paperless Billing", ["Yes", "No"])
    payment_method = st.sidebar.selectbox("Payment Method", [
        "Electronic check", "Mailed check",
        "Bank transfer (automatic)", "Credit card (automatic)"
    ])
    monthly_charges = st.sidebar.slider("Monthly Charges ($)", 18.0, 120.0, 65.0)
    total_charges = st.sidebar.slider("Total Charges ($)", 0.0, 9000.0, 1500.0)

    # Encode inputs same way as training
    data = {
        'gender': 1 if gender == "Male" else 0,
        'SeniorCitizen': 1 if senior == "Yes" else 0,
        'Partner': 1 if partner == "Yes" else 0,
        'Dependents': 1 if dependents == "Yes" else 0,
        'tenure': tenure,
        'PhoneService': 1 if phone_service == "Yes" else 0,
        'MultipleLines': {"No": 0, "Yes": 1, "No phone service": 2}[multiple_lines],
        'InternetService': {"DSL": 0, "Fiber optic": 1, "No": 2}[internet_service],
        'OnlineSecurity': {"No": 0, "Yes": 1, "No internet service": 2}[online_security],
        'OnlineBackup': {"No": 0, "Yes": 1, "No internet service": 2}[online_backup],
        'DeviceProtection': {"No": 0, "Yes": 1, "No internet service": 2}[device_protection],
        'TechSupport': {"No": 0, "Yes": 1, "No internet service": 2}[tech_support],
        'StreamingTV': {"No": 0, "Yes": 1, "No internet service": 2}[streaming_tv],
        'StreamingMovies': {"No": 0, "Yes": 1, "No internet service": 2}[streaming_movies],
        'Contract': {"Month-to-month": 0, "One year": 1, "Two year": 2}[contract],
        'PaperlessBilling': 1 if paperless_billing == "Yes" else 0,
        'PaymentMethod': {
            "Bank transfer (automatic)": 0,
            "Credit card (automatic)": 1,
            "Electronic check": 2,
            "Mailed check": 3
        }[payment_method],
        'MonthlyCharges': monthly_charges,
        'TotalCharges': total_charges
    }
    return pd.DataFrame([data])

input_df = user_input()

# ─────────────────────────────────────────
# PREDICTION
# ─────────────────────────────────────────
prediction = model.predict(input_df)[0]
probability = model.predict_proba(input_df)[0][1]

# ─────────────────────────────────────────
# RESULTS SECTION
# ─────────────────────────────────────────
col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Churn Prediction", "⚠️ Will Churn" if prediction == 1 else "✅ Will Stay")

with col2:
    st.metric("Churn Probability", f"{probability*100:.1f}%")

with col3:
    risk = "🔴 High Risk" if probability > 0.7 else "🟡 Medium Risk" if probability > 0.4 else "🟢 Low Risk"
    st.metric("Risk Level", risk)

st.markdown("---")

# ─────────────────────────────────────────
# PROGRESS BAR
# ─────────────────────────────────────────
st.subheader("Churn Probability")
st.progress(float(probability))
st.markdown(f"**{probability*100:.1f}%** chance this customer will churn.")

st.markdown("---")

# ─────────────────────────────────────────
# SHAP EXPLANATION
# ─────────────────────────────────────────
st.subheader("🔍 Why this prediction? (SHAP Explanation)")
st.markdown("The chart below shows which features pushed the prediction toward churning or staying.")

try:
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(input_df)

    fig, ax = plt.subplots(figsize=(10, 4))
    shap.waterfall_plot(
        shap.Explanation(
            values=shap_values[0],
            base_values=explainer.expected_value,
            data=input_df.iloc[0],
            feature_names=feature_names
        ),
        show=False
    )
    st.pyplot(fig)
    plt.close()
except Exception as e:
    st.info("SHAP explanation not available for this input.")

st.markdown("---")

# ─────────────────────────────────────────
# MODEL PERFORMANCE CHARTS
# ─────────────────────────────────────────
st.subheader("📈 Model Performance")

tab1, tab2, tab3, tab4 = st.tabs([
    "ROC Curve", "Confusion Matrix",
    "Feature Importance", "Churn Distribution"
])

with tab1:
    st.image("assets/roc_curve.png", caption="ROC Curve — All 3 Models")

with tab2:
    st.image("assets/confusion_matrix.png", caption="Confusion Matrix — XGBoost")

with tab3:
    st.image("assets/feature_importance.png", caption="Top 15 Features — XGBoost")

with tab4:
    st.image("assets/churn_distribution.png", caption="Churn Distribution in Dataset")

st.markdown("---")

# ─────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────
st.markdown("""
<div style='text-align: left;'>
    Developed by <strong><a href='https://www.linkedin.com/in/azmain-abir'
    target='_blank' style='text-decoration: none; color: #1f77b4;'>
    Azmain Tahmid Abir</a></strong> — CSE Student @ Daffodil International University
</div>
""", unsafe_allow_html=True)