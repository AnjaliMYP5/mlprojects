import streamlit as st
import pandas as pd
import numpy as np
import random

# --- Streamlit App Configuration ---
st.set_page_config(
    page_title="Loan Prediction App",
    page_icon="💵",
    layout="wide",
)

st.title("Loan Prediction App 💵")
st.write("This app predicts whether a loan will be approved based on an applicant's details.")

# --- Mock Prediction Model ---
def mock_predict_loan_status(features):
    """
    A mock prediction function that simulates a machine learning model.
    In a real-world scenario, you would load a trained model here (e.g., from scikit-learn).
    For example:
    import joblib
    model = joblib.load('your_trained_model.pkl')
    prediction = model.predict([features_as_list])
    
    This mock model uses simple rules for demonstration purposes.
    - A credit history of 1 (yes) greatly increases the chance of approval.
    - High income and low loan amount also contribute to approval.
    """
    
    credit_history = features['Credit_History']
    applicant_income = features['ApplicantIncome']
    coapplicant_income = features['CoapplicantIncome']
    loan_amount = features['LoanAmount']
    
    score = 0
    
    # Rule 1: Credit History is the most important factor
    if credit_history == 1:
        score += 50
    else:
        score -= 50
        
    # Rule 2: Check for high income
    total_income = applicant_income + coapplicant_income
    if total_income > 6000:
        score += 20
    elif total_income > 3000:
        score += 10
    else:
        score -= 10
        
    # Rule 3: Check loan amount vs. income
    if loan_amount and total_income:
        debt_to_income = loan_amount / total_income
        if debt_to_income < 0.2:
            score += 20
        elif debt_to_income < 0.4:
            score += 10
        else:
            score -= 10
    
    # Randomness for a more dynamic mock result
    score += random.randint(-10, 10)
    
    # Final decision threshold
    if score >= 30:
        return "Y", "Approved"
    else:
        return "N", "Rejected"

# --- User Input Section ---
st.sidebar.header("Applicant Details")

col1, col2 = st.sidebar.columns(2)
with col1:
    gender = st.selectbox("Gender", ("Male", "Female"))
    married = st.selectbox("Marital Status", ("Married", "Unmarried"))
    education = st.selectbox("Education", ("Graduate", "Not Graduate"))
    
with col2:
    self_employed = st.selectbox("Self Employed?", ("Yes", "No"))
    credit_history = st.selectbox("Credit History (1=Yes, 0=No)", (1, 0))
    property_area = st.selectbox("Property Area", ("Urban", "Rural", "Semiurban"))

st.sidebar.markdown("---")
st.sidebar.header("Financial Details")

col3, col4 = st.sidebar.columns(2)
with col3:
    applicant_income = st.number_input("Applicant Income", min_value=0, value=5000)
    coapplicant_income = st.number_input("Co-applicant Income", min_value=0, value=2000)
with col4:
    loan_amount = st.number_input("Loan Amount", min_value=0, value=150)
    loan_amount_term = st.number_input("Loan Amount Term (in months)", min_value=12, value=360)

# --- Prediction Button and Logic ---
st.markdown("---")

if st.button("Predict Loan Status"):
    
    # Create a dictionary of features
    features = {
        'Gender': gender,
        'Married': married,
        'Education': education,
        'Self_Employed': self_employed,
        'ApplicantIncome': applicant_income,
        'CoapplicantIncome': coapplicant_income,
        'LoanAmount': loan_amount,
        'Loan_Amount_Term': loan_amount_term,
        'Credit_History': credit_history,
        'Property_Area': property_area,
    }
    
    with st.spinner("Predicting..."):
        prediction, status = mock_predict_loan_status(features)
    
    # Display the result
    if prediction == "Y":
        st.balloons()
        st.success(f"**Loan Status:** {status}")
        st.markdown("Congratulations! Based on the provided data, the loan is likely to be **approved**.")
        st.markdown(
            "The model's decision was primarily influenced by a positive credit history and a healthy income-to-loan ratio."
        )
    else:
        st.error(f"**Loan Status:** {status}")
        st.markdown("Based on the provided data, the loan is likely to be **rejected**.")
        st.markdown(
            "The model's decision was likely influenced by a negative credit history, lower income, or a high loan amount relative to income."
        )
