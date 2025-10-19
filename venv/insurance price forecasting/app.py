import streamlit as st
import pandas as pd
import numpy as np
import random

# --- Streamlit App Configuration ---
st.set_page_config(
    page_title="Insurance Cost Prediction App",
    page_icon="🏥",
    layout="wide",
)

st.title("Insurance Cost Prediction App 🏥")
st.write("This app predicts the estimated insurance cost based on a user's health profile.")

# --- Mock Prediction Model ---
def mock_predict_insurance_cost(features):
    """
    A mock prediction function that simulates a machine learning model.
    In a real-world scenario, you would load a trained model here (e.g., from scikit-learn).
    
    This mock model uses simple rules for demonstration purposes.
    - Age, BMI, and being a smoker are the strongest predictors.
    - The number of children and region also have an effect.
    """
    
    age = features['age']
    bmi = features['bmi']
    children = features['children']
    smoker = features['smoker']
    region = features['region']
    
    # Base cost
    cost = 5000 + (age * 100)
    
    # Impact of BMI
    if bmi > 30:
        cost += (bmi - 30) * 200
    else:
        cost += bmi * 50
        
    # Impact of being a smoker
    if smoker == 'yes':
        cost += 15000
        
    # Impact of children
    cost += children * 300
    
    # Impact of region
    if region == 'southeast':
        cost += 500
    elif region == 'southwest':
        cost -= 500
        
    # Add some randomness for a more dynamic mock result
    cost += random.randint(-500, 500)
    
    return max(cost, 0) # Ensure cost is not negative

# --- User Input Section ---
st.sidebar.header("Health Profile")

col1, col2 = st.sidebar.columns(2)
with col1:
    age = st.number_input("Age", min_value=18, max_value=100, value=30)
    bmi = st.number_input("BMI (Body Mass Index)", min_value=10.0, max_value=60.0, value=25.0)
with col2:
    children = st.number_input("Number of Children", min_value=0, max_value=10, value=1)
    smoker = st.selectbox("Smoker?", ("no", "yes"))
    
st.sidebar.markdown("---")
st.sidebar.header("Location Details")

region = st.sidebar.selectbox("Region", ("northeast", "northwest", "southeast", "southwest"))

# --- Prediction Button and Logic ---
st.markdown("---")

if st.button("Predict Insurance Cost"):
    
    # Create a dictionary of features
    features = {
        'age': age,
        'bmi': bmi,
        'children': children,
        'smoker': smoker,
        'region': region,
    }
    
    with st.spinner("Predicting..."):
        predicted_cost = mock_predict_insurance_cost(features)
    
    # Display the result
    st.success(f"**Predicted Insurance Cost:**")
    st.markdown(f"**${predicted_cost:,.2f}** per year")
    
    st.markdown("---")
    st.write("### Prediction Factors")
    st.markdown(
        """
        The prediction is based on the following mock-model insights:
        * **Age:** Older individuals generally have higher costs.
        * **BMI:** Higher BMI (especially over 30) leads to increased costs.
        * **Smoker Status:** Being a smoker significantly increases the predicted cost.
        * **Children:** Having more children slightly increases the cost.
        * **Region:** The region also influences the cost, with the southeast being more expensive and the southwest being cheaper in this simulation.
        """
    )
