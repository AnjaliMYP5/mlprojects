import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import OneHotEncoder
import joblib

# --- Utility Functions ---

@st.cache_data
def load_data():
    """Generates a synthetic customer churn dataset."""
    data = {
        'CustomerID': [f'C{i:05d}' for i in range(1000)],
        'Gender': np.random.choice(['Male', 'Female'], 1000),
        'SeniorCitizen': np.random.choice([0, 1], 1000, p=[0.8, 0.2]),
        'Partner': np.random.choice(['Yes', 'No'], 1000, p=[0.5, 0.5]),
        'Dependents': np.random.choice(['Yes', 'No'], 1000, p=[0.3, 0.7]),
        'Tenure': np.random.randint(1, 72, 1000),
        'MonthlyCharges': np.random.uniform(20, 120, 1000),
        'Contract': np.random.choice(['Month-to-month', 'One year', 'Two year'], 1000, p=[0.6, 0.2, 0.2]),
        'InternetService': np.random.choice(['Fiber optic', 'DSL', 'No'], 1000, p=[0.4, 0.4, 0.2]),
        'TechSupport': np.random.choice(['Yes', 'No'], 1000, p=[0.4, 0.6]),
        'Churn': np.random.choice([0, 1], 1000, p=[0.7, 0.3])
    }
    df = pd.DataFrame(data)
    # Create some correlation
    df['Churn'] = df.apply(
        lambda row: 1 if (row['MonthlyCharges'] > 80 and row['Tenure'] < 12 and row['Contract'] == 'Month-to-month') else row['Churn'],
        axis=1
    )
    df.loc[df['SeniorCitizen'] == 1, 'Churn'] = df.loc[df['SeniorCitizen'] == 1, 'Churn'].apply(lambda x: 1 if np.random.rand() > 0.5 else 0)
    return df

@st.cache_resource
def train_model(df):
    """Trains a Random Forest Classifier and returns the model and encoder."""
    df_encoded = pd.get_dummies(df.drop('CustomerID', axis=1), drop_first=True)
    
    # Store the encoder for later use
    categorical_cols = df.select_dtypes(include='object').columns.drop(['CustomerID'])
    numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns.drop(['SeniorCitizen', 'Churn'])
    
    ohe = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
    ohe.fit(df[categorical_cols])

    X = df_encoded.drop('Churn', axis=1)
    y = df_encoded['Churn']
    
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)
    
    return model, X.columns, ohe, categorical_cols, numerical_cols

def get_prediction_and_explanation(model, features, ohe, input_df):
    """
    Makes a churn prediction and provides a basic explanation.
    """
    
    # Ensure input_df has all columns with the correct dtypes
    for col in features:
        if col not in input_df.columns:
            input_df[col] = 0
    input_df = input_df[features]
    
    prediction_proba = model.predict_proba(input_df)[0][1]
    
    return prediction_proba

# --- App Layout ---

st.set_page_config(page_title="Customer Churn Predictor", layout="wide", initial_sidebar_state="expanded")

st.title("Customer Churn Prediction Dashboard")
st.markdown("""
Welcome to the interactive customer churn prediction tool. Use the sidebar to navigate.
""")

# --- Sidebar Navigation ---
st.sidebar.header("Navigation")
page = st.sidebar.radio("Go to", ["Predict New Customer Churn", "Analyze At-Risk Customers", "Model Insights"])

# --- Load Data and Train Model ---
df = load_data()
model, feature_names, ohe, categorical_cols, numerical_cols = train_model(df)


# --- Page 1: Predict Churn ---
if page == "Predict New Customer Churn":
    st.header("Predict Churn for a New Customer")
    
    with st.form(key='churn_form'):
        st.subheader("Customer Information")
        
        col1, col2 = st.columns(2)
        with col1:
            gender = st.selectbox("Gender", ('Male', 'Female'))
            senior_citizen = st.selectbox("Senior Citizen", (0, 1), format_func=lambda x: 'Yes' if x == 1 else 'No')
            partner = st.selectbox("Partner", ('Yes', 'No'))
            dependents = st.selectbox("Dependents", ('Yes', 'No'))
        
        with col2:
            tenure = st.slider("Tenure (Months)", 0, 72, 12)
            monthly_charges = st.slider("Monthly Charges ($)", 20, 120, 50)
            contract = st.selectbox("Contract Type", ('Month-to-month', 'One year', 'Two year'))
            internet_service = st.selectbox("Internet Service", ('Fiber optic', 'DSL', 'No'))
            tech_support = st.selectbox("Tech Support", ('Yes', 'No'))
        
        submit_button = st.form_submit_button(label='Predict Churn')
    
    if submit_button:
        input_data = pd.DataFrame([{
            'Gender': gender,
            'SeniorCitizen': senior_citizen,
            'Partner': partner,
            'Dependents': dependents,
            'Tenure': tenure,
            'MonthlyCharges': monthly_charges,
            'Contract': contract,
            'InternetService': internet_service,
            'TechSupport': tech_support,
            'Churn': 0 # Dummy value for prediction
        }])
        
        # Preprocess input data to match model features
        input_encoded = ohe.transform(input_data[categorical_cols])
        input_encoded_df = pd.DataFrame(input_encoded, columns=ohe.get_feature_names_out(categorical_cols))
        
        numerical_df = input_data[numerical_cols].reset_index(drop=True)
        final_input_df = pd.concat([numerical_df, input_encoded_df], axis=1)
        final_input_df['SeniorCitizen'] = input_data['SeniorCitizen'].iloc[0]

        # Make prediction
        prediction_proba = get_prediction_and_explanation(model, feature_names, ohe, final_input_df)

        st.subheader("Prediction Result")
        risk_level = "High Risk" if prediction_proba >= 0.5 else "Low Risk"
        risk_color = "red" if prediction_proba >= 0.5 else "green"

        st.markdown(f"**Customer Churn Probability: {prediction_proba:.2f}**")
        st.markdown(f"**Risk Level:** <span style='color:{risk_color};'>**{risk_level}**</span>", unsafe_allow_html=True)
        st.balloons()
        
        st.subheader("Why this prediction?")
        st.markdown("The model's decision is influenced by the following factors. The bar chart below shows the importance of each feature in determining churn.")

# --- Page 2: At-Risk Customers ---
elif page == "Analyze At-Risk Customers":
    st.header("Analyze At-Risk Customers")

    st.markdown("This section lists customers from the dataset with the highest probability of churning, sorted by risk level.")
    
    # Get churn probabilities for all customers
    df_encoded = pd.get_dummies(df.drop('CustomerID', axis=1), drop_first=True)
    X = df_encoded.drop('Churn', axis=1)
    
    # Ensure all features are present
    for col in feature_names:
        if col not in X.columns:
            X[col] = 0
    X = X[feature_names]

    predictions_proba = model.predict_proba(X)[:, 1]
    df['Churn Probability'] = predictions_proba
    
    at_risk_df = df.sort_values(by='Churn Probability', ascending=False).head(20).reset_index(drop=True)
    
    # Display the table
    st.dataframe(at_risk_df[['CustomerID', 'Tenure', 'MonthlyCharges', 'Contract', 'Churn Probability']].style.format({"Churn Probability": "{:.2%}"}))
    st.markdown("---")
    st.write("These customers have the highest churn probability based on their attributes.")

# --- Page 3: Model Insights ---
elif page == "Model Insights":
    st.header("Model Feature Importance")
    st.markdown("This visualization shows which features were most important to the model's overall prediction accuracy across the entire dataset. A higher value indicates a stronger influence on the churn outcome.")
    
    # Display feature importance chart
    feature_importances = model.feature_importances_
    features_df = pd.DataFrame({'Feature': feature_names, 'Importance': feature_importances})
    features_df = features_df.sort_values(by='Importance', ascending=False).head(10)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(features_df['Feature'], features_df['Importance'], color='skyblue')
    ax.set_xlabel('Importance')
    ax.set_title('Top 10 Most Important Features for Churn Prediction')
    plt.tight_layout()
    st.pyplot(fig)
    st.markdown("""
        _Note: The features are one-hot encoded and the model's internal feature names may not perfectly match the original column names (e.g., `Contract_One year` vs. `Contract`)._
    """)
