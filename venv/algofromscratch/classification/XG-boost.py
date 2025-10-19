import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import xgboost as xgb

# Set up the Streamlit page
st.set_page_config(page_title="XGBoost Parameter Visualizer", layout="wide")
st.title("XGBoost Parameter Visualizer")
st.markdown("### An interactive tool to explore the impact of XGBoost hyperparameters on model performance and decision boundaries.")

# --- Sidebar for user input ---
st.sidebar.header("Model Hyperparameters")
n_estimators = st.sidebar.slider(
    "Number of Estimators", min_value=1, max_value=200, value=100, step=1
)
max_depth = st.sidebar.slider(
    "Max Depth", min_value=1, max_value=10, value=3, step=1
)
learning_rate = st.sidebar.slider(
    "Learning Rate", min_value=0.01, max_value=1.0, value=0.1, step=0.01
)
subsample = st.sidebar.slider(
    "Subsample Ratio", min_value=0.1, max_value=1.0, value=1.0, step=0.05
)
colsample_bytree = st.sidebar.slider(
    "Colsample by Tree", min_value=0.1, max_value=1.0, value=1.0, step=0.05
)

# --- Generate synthetic dataset ---
st.sidebar.header("Dataset Configuration")
n_samples = st.sidebar.slider(
    "Number of Samples", min_value=100, max_value=1000, value=500, step=50
)
random_state = st.sidebar.number_input(
    "Random Seed", min_value=0, value=42
)
X, y = make_classification(
    n_samples=n_samples,
    n_features=2,
    n_redundant=0,
    n_informative=2,
    random_state=random_state,
    n_clusters_per_class=1
)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=random_state
)

# --- Train the XGBoost model ---
@st.cache_resource
def train_model(n_estimators, max_depth, learning_rate, subsample, colsample_bytree):
    model = xgb.XGBClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        subsample=subsample,
        colsample_bytree=colsample_bytree,
        use_label_encoder=False,
        eval_metric="logloss",
        random_state=42
    )
    model.fit(X_train, y_train)
    return model

model = train_model(n_estimators, max_depth, learning_rate, subsample, colsample_bytree)

# --- Make predictions and evaluate ---
y_pred = model.predict(X_test)

# --- Display results ---
st.subheader("Model Performance on Test Data")
col1, col2, col3, col4 = st.columns(4)
col1.metric("Accuracy", f"{accuracy_score(y_test, y_pred):.2f}")
col2.metric("Precision", f"{precision_score(y_test, y_pred):.2f}")
col3.metric("Recall", f"{recall_score(y_test, y_pred):.2f}")
col4.metric("F1-Score", f"{f1_score(y_test, y_pred):.2f}")

st.subheader("Decision Boundary Visualization")

# --- Function to plot decision boundary ---
def plot_decision_boundary(model, X, y):
    h = 0.02  # step size in the mesh
    cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF'])
    cmap_bold = ListedColormap(['#FF0000', '#00FF00', '#0000FF'])

    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.contourf(xx, yy, Z, cmap=cmap_light, alpha=0.8)

    # Plot the training points
    ax.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=cmap_bold, edgecolor='k', s=20, label="Training Data")
    
    # Plot the test points
    ax.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap=cmap_bold, edgecolor='k', s=50, alpha=0.6, label="Test Data")
    
    ax.set_title("XGBoost Decision Boundary")
    ax.set_xlabel("Feature 1")
    ax.set_ylabel("Feature 2")
    ax.legend()
    st.pyplot(fig)

plot_decision_boundary(model, X, y)

st.info("The background colors represent the model's predicted class. The solid dots are the training data, and the larger, semi-transparent dots are the test data.")

st.subheader("Feature Importance")
feature_importance = model.feature_importances_
fig, ax = plt.subplots()
ax.bar(["Feature 1", "Feature 2"], feature_importance)
ax.set_title("Feature Importance from XGBoost Model")
ax.set_ylabel("Importance")
st.pyplot(fig)
