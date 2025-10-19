import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier

# Set up the Streamlit page
st.set_page_config(page_title="ML Algorithm Visualizer", layout="wide")
st.title("ML Algorithm Visualizer")
st.markdown("### Compare the impact of hyperparameters for different classification algorithms.")

# --- Sidebar for user input ---
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

st.sidebar.header("Algorithm and Hyperparameters")
classifier_name = st.sidebar.selectbox(
    "Select Classifier",
    ("Logistic Regression", "SVC", "Decision Tree", "Random Forest", "K-Nearest Neighbors")
)

# --- Hyperparameter selection based on classifier ---
def get_params(clf_name):
    params = {}
    if clf_name == "Logistic Regression":
        C = st.sidebar.slider("Regularization (C)", 0.01, 10.0, 1.0, step=0.01)
        params["C"] = C
    elif clf_name == "SVC":
        C = st.sidebar.slider("Regularization (C)", 0.01, 10.0, 1.0, step=0.01)
        kernel = st.sidebar.selectbox("Kernel", ("rbf", "linear", "poly"))
        params["C"] = C
        params["kernel"] = kernel
    elif clf_name == "Decision Tree":
        max_depth = st.sidebar.slider("Max Depth", 1, 10, 3)
        params["max_depth"] = max_depth
    elif clf_name == "Random Forest":
        n_estimators = st.sidebar.slider("Number of Estimators", 1, 200, 100)
        max_depth = st.sidebar.slider("Max Depth", 1, 10, 3)
        params["n_estimators"] = n_estimators
        params["max_depth"] = max_depth
    elif clf_name == "K-Nearest Neighbors":
        n_neighbors = st.sidebar.slider("Number of Neighbors (k)", 1, 20, 5)
        params["n_neighbors"] = n_neighbors
    return params

params = get_params(classifier_name)

# --- Train the model ---
@st.cache_resource(experimental_allow_widgets=True)
def train_model(clf_name, params):
    if clf_name == "Logistic Regression":
        model = LogisticRegression(C=params["C"])
    elif clf_name == "SVC":
        model = SVC(C=params["C"], kernel=params["kernel"])
    elif clf_name == "Decision Tree":
        model = DecisionTreeClassifier(max_depth=params["max_depth"])
    elif clf_name == "Random Forest":
        model = RandomForestClassifier(n_estimators=params["n_estimators"], max_depth=params["max_depth"])
    elif clf_name == "K-Nearest Neighbors":
        model = KNeighborsClassifier(n_neighbors=params["n_neighbors"])
    
    model.fit(X_train, y_train)
    return model

model = train_model(classifier_name, params)

# --- Make predictions and evaluate ---
y_pred = model.predict(X_test)

st.subheader("Model Performance on Test Data")
col1, col2, col3, col4 = st.columns(4)
col1.metric("Accuracy", f"{accuracy_score(y_test, y_pred):.2f}")
col2.metric("Precision", f"{precision_score(y_test, y_pred, zero_division=0):.2f}")
col3.metric("Recall", f"{recall_score(y_test, y_pred, zero_division=0):.2f}")
col4.metric("F1-Score", f"{f1_score(y_test, y_pred, zero_division=0):.2f}")

st.subheader("Decision Boundary Visualization")

# --- Function to plot decision boundary ---
def plot_decision_boundary(model, X, y):
    h = 0.02  # step size in the mesh
    cmap_light = ListedColormap(['#FFCCCC', '#CCFFCC', '#CCCCFF'])
    cmap_bold = ListedColormap(['#FF0000', '#00FF00', '#0000FF'])

    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.contourf(xx, yy, Z, cmap=cmap_light, alpha=0.8)

    ax.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=cmap_bold, edgecolor='k', s=20, label="Training Data")
    ax.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap=cmap_bold, edgecolor='k', s=50, alpha=0.6, label="Test Data")
    
    ax.set_title(f"{classifier_name} Decision Boundary")
    ax.set_xlabel("Feature 1")
    ax.set_ylabel("Feature 2")
    ax.legend()
    st.pyplot(fig)

plot_decision_boundary(model, X, y)

st.info("The background colors represent the model's predicted class. Use the sidebar to change the algorithm or tune its parameters.")
