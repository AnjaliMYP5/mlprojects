import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# --- Sidebar Navigation ---
st.sidebar.title("🔎 Fraud Detection App")
section = st.sidebar.radio(
    "Navigate",
    [
        "About the Project",
        "Data",
        "Prediction",
        "Insights"
    ]
)

# --- About the Project Section ---
if section == "About the Project":
    st.markdown("""
    # 🛡️ Fraud Detection Web App
    ## ℹ️ About the Project
    **Problem Statement:**  
    Detecting fraudulent transactions is challenging due to class imbalance and evolving fraud patterns.

    **Methodology:**  
    - Data preprocessing (handling missing values, scaling, encoding)
    - Model selection (Logistic Regression, Random Forest, XGBoost)
    - Evaluation metrics: Precision, Recall, F1-Score, ROC-AUC

    **Technology Stack:**  
    Python, Pandas, Scikit-learn, Streamlit, Matplotlib
    ---
    """)

# --- Data Section ---
elif section == "Data":
    st.markdown("## 📁 Data Upload & Preview")
    uploaded_file = st.file_uploader("Upload CSV or Excel file", type=["csv", "xlsx"])
    show_df = st.toggle("Show DataFrame Preview")
    if uploaded_file:
        df = pd.read_csv(uploaded_file) if uploaded_file.name.endswith('.csv') else pd.read_excel(uploaded_file)
        if show_df:
            st.dataframe(df.head())
        with st.expander("📊 Basic Statistics", expanded=False):
            st.write("Rows:", df.shape[0])
            st.write("Columns:", df.shape[1])
            st.write("Missing Values:", df.isnull().sum().sum())
            st.write("Column Types:", df.dtypes)
    else:
        st.info("Upload a file to preview data.")

# --- Prediction Section ---
elif section == "Prediction":
    st.markdown("## 📝 Interactive Input Form")
    st.write("Enter transaction details for real-time fraud prediction.")

    # Example feature inputs
    amount = st.number_input("Amount", min_value=0.0, max_value=10000.0, value=100.0)
    time = st.number_input("Time (seconds since first transaction)", min_value=0, max_value=172800, value=50000)
    st.markdown("#### V1-V28 (Anonymized Features)")
    v_features = {}
    cols = st.columns(7)
    for i in range(1, 29):
        col = cols[(i-1)%7]
        v_features[f"V{i}"] = col.slider(f"V{i}", min_value=-20.0, max_value=20.0, value=0.0)

    # Button to randomly generate values
    if st.button("Random Example"):
        amount = np.random.uniform(0, 10000)
        time = np.random.randint(0, 172800)
        for i in range(1, 29):
            v_features[f"V{i}"] = np.random.uniform(-20, 20)
        st.experimental_rerun()

    # Placeholder for model prediction
    if st.button("Predict"):
        # Dummy prediction logic (replace with your model)
        prob_fraud = np.clip((amount/10000 + abs(v_features['V1'])/20)/2, 0, 1)
        verdict = "Fraudulent" if prob_fraud > 0.5 else "Legitimate"
        st.success(f"Prediction: **{verdict}**")
        st.info(f"Probability Score: {prob_fraud*100:.2f}% {'fraudulent' if verdict=='Fraudulent' else 'legitimate'}")

        # --- Feature Importance (Dummy) ---
        st.markdown("### 📊 Feature Importance")
        importance = {k: abs(v) for k, v in v_features.items()}
        imp_df = pd.DataFrame(list(importance.items()), columns=["Feature", "Importance"])
        imp_df = imp_df.sort_values("Importance", ascending=False)
        fig, ax = plt.subplots()
        ax.barh(imp_df["Feature"], imp_df["Importance"])
        ax.invert_yaxis()
        st.pyplot(fig)

        # --- Data Distribution Plots (Dummy) ---
        st.markdown("### 📈 Data Distribution")
        fig, axs = plt.subplots(1, 2, figsize=(10, 3))
        # Amount distribution
        axs[0].hist(np.random.normal(1000, 2000, 1000), bins=30, alpha=0.7, label='Legitimate')
        axs[0].axvline(amount, color='red', linestyle='--', label='Your Input')
        axs[0].set_title("Amount Distribution")
        axs[0].legend()
        # Time distribution
        axs[1].hist(np.random.normal(50000, 30000, 1000), bins=30, alpha=0.7, label='Legitimate')
        axs[1].axvline(time, color='red', linestyle='--', label='Your Input')
        axs[1].set_title("Time Distribution")
        axs[1].legend()
        st.pyplot(fig)

# --- Insights Section ---
elif section == "Insights":
    st.markdown("## 📊 Insights & Explanations")
    with st.expander("Feature Importance", expanded=False):
        st.write("Placeholder for global feature importance (e.g., SHAP/LIME).")
    with st.expander("Fraud vs. Non-Fraud Distribution", expanded=False):
        st.write("Placeholder for fraud distribution chart.")
    with st.expander("Correlation Heatmap", expanded=False):
        st.write("Placeholder for correlation heatmap.")
    with st.expander("ROC & Precision-Recall Curve", expanded=False):
        st.write("Placeholder for ROC and PR curves.")
    with st.expander("Confusion Matrix", expanded=False):
        st.write("Placeholder for confusion matrix and threshold slider.")

# --- Footer ---
st.markdown("---")
st.caption("© 2025 Fraud Detection")


