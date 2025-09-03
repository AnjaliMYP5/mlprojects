import streamlit as st
import pickle
import pandas as pd
import numpy as np

# Load the trained model and preprocessor (e.g., a scaler or SMOTE)
# The file names should match what you saved in your model training step.
try:
    with open('model.pkl', 'rb') as model_file:
        model = pickle.load(model_file)
    # If you used a scaler for preprocessing, load it here as well
    # with open('scaler.pkl', 'rb') as scaler_file:
    #     scaler = pickle.load(scaler_file)
except FileNotFoundError:
    st.error("Model or preprocessor file not found. Please ensure 'model.pkl' is in the same directory.")
    st.stop()
st.title("Credit Risk Predictor")
st.markdown("A simple app to predict the likelihood of credit card default based on financial and demographic data.")

with st.form("user_input_form"):
    st.header("Financial and Demographic Information")

    # Use a slider for numerical input with a range
    limit_bal = st.slider("Credit Limit (in NT dollars)", 10000, 1000000, 50000)

    # Use a selectbox for categorical data
    sex = st.selectbox("Gender", options=["Male", "Female"])
    education = st.selectbox("Education Level", options=["Graduate School", "University", "High School", "Other"])
    marital_status = st.selectbox("Marital Status", options=["Married", "Single", "Divorced", "Other"])
    age = st.number_input("Age", min_value=18, max_value=80, value=30)
    
    # Use number inputs for payment history
    pay_0 = st.number_input("Payment Status (Sep)", min_value=-2, max_value=8, value=0, help="-2: no consumption, -1: paid, 0: revolving credit, 1: 1-month delay, etc.")
    pay_2 = st.number_input("Payment Status (Aug)", min_value=-2, max_value=8, value=0)
    # Add other 'pay_' variables here...
    
    # Use number inputs for bill amounts
    bill_amt1 = st.number_input("Bill Amount (Sep)", value=0)
    bill_amt2 = st.number_input("Bill Amount (Aug)", value=0)
    # Add other 'bill_amt' variables here...

    # Use number inputs for payment amounts
    pay_amt1 = st.number_input("Amount Paid (Sep)", value=0)
    pay_amt2 = st.number_input("Amount Paid (Aug)", value=0)
    # Add other 'pay_amt' variables here...
    
    # Create the prediction button
    submitted = st.form_submit_button("Predict Credit Risk")

    if submitted:
    # 1. Create a DataFrame from the user inputs
        user_data = {
            'LIMIT_BAL': [limit_bal],
            'SEX': [1 if sex == 'Male' else 2],
            'EDUCATION': [1 if education == 'Graduate School' else 2 if education == 'University' else 3 if education == 'High School' else 4],
            'MARRIAGE': [1 if marital_status == 'Married' else 2 if marital_status == 'Single' else 3],
            'AGE': [age],
            'PAY_0': [pay_0],
            'PAY_2': [pay_2],
            'BILL_AMT1': [bill_amt1],
            'BILL_AMT2': [bill_amt2],
            'PAY_AMT1': [pay_amt1],
            'PAY_AMT2': [pay_amt2],
            # Add all other features here...
        }
        user_df = pd.DataFrame(user_data)
        
        # 2. Preprocess the input data (if you used a scaler)
        # user_scaled = scaler.transform(user_df)

        # 3. Make the prediction
        prediction = model.predict(user_df)
        # Get the probability of default
        prediction_proba = model.predict_proba(user_df)[0][1]

        # 4. Display the results
        st.subheader("Prediction Result")
        if prediction[0] == 1:
            st.error("Likely to Default")
            st.markdown(f"**Confidence Score:** {prediction_proba*100:.2f}%")
            st.warning("This prediction suggests a high risk of default. Please consider seeking professional financial advice.")
        else:
            st.success("Not Likely to Default")
            st.markdown(f"**Confidence Score:** {prediction_proba*100:.2f}%")
            st.info("The model has high confidence that this user will not default based on the provided data.")