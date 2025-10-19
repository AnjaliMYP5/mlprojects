# Required libraries:
# pip install streamlit joblib shap

import streamlit as st
import joblib
import pandas as pd
import shap
import numpy as np

# Load the trained model and feature list
model = joblib.load('phishing_model.pkl')
feature_list = joblib.load('feature_list.pkl')

# A simple function to extract key features from a URL (same as above)
def extract_features(url):
    features = {}
    features['url_length'] = len(url)
    features['num_dots'] = url.count('.')
    features['num_hyphens'] = url.count('-')
    features['has_https'] = 1 if 'https://' in url else 0
    features['is_shortened'] = 1 if 'bit.ly' in url or 'goo.gl' in url else 0
    features['has_at_symbol'] = 1 if '@' in url else 0
    features['has_redirect'] = 1 if '//' in url.replace('https://', '').replace('http://', '') else 0
    return features

# ---- UI Elements ----
st.title("🛡️ Phishing URL Detector")
st.markdown("Enter a URL below to check if it's a potential phishing site.")

url_input = st.text_input("Enter URL Here:")
predict_button = st.button("Check URL")

# ---- Main Prediction Logic ----
if predict_button and url_input:
    # 1. Feature Extraction
    features = extract_features(url_input)
    input_df = pd.DataFrame([features], columns=feature_list)

    # 2. Prediction
    prediction = model.predict(input_df)[0]
    prediction_proba = model.predict_proba(input_df)[0][1]

    # 3. Display Results with Explanations
    st.markdown("---")
    st.subheader("Prediction Result")
    
    if prediction == 1:
        st.error(f"⚠️ This looks like a **phishing URL**! (Confidence: {prediction_proba:.2f})")
    else:
        st.success(f"✅ This URL looks **legitimate**. (Confidence: {1 - prediction_proba:.2f})")
    
    st.markdown("---")

    # 4. Explainable AI (XAI) with SHAP
    st.subheader("Why this prediction was made")
    
    # Create a SHAP explainer
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(input_df)
    
    # The first element is for the 0 class, the second is for the 1 class
    # We want to explain the prediction of the phishing class (1)
    
    # Use SHAP to generate a force plot for a single prediction
    st.write("The chart below shows which features contributed to the model's decision.")
    # The SHAP force plot visualizes feature contributions
    st.set_option('deprecation.showPyplotGlobalUse', False)
    shap.force_plot(explainer.expected_value[1], shap_values[1][0], input_df.iloc[0], matplotlib=True, show=False)
    st.pyplot()
    

    # ---- Community Feedback Loop ----
    st.markdown("---")
    st.subheader("Was this prediction correct?")
    feedback = st.radio("Help improve the model!", ('Yes', 'No'))
    if st.button("Submit Feedback"):
        with open("feedback.csv", "a") as f:
            f.write(f"{url_input},{prediction},{feedback}\n")
        st.success("Thanks for your feedback! It will be used to improve the model.")