import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import plot_tree
import matplotlib.pyplot as plt
import numpy as np

def run_random_forest_app():
    """Main function to run the Streamlit application."""

    st.title("🌲 Feature Impact in a Random Forest")
    st.write("Upload a dataset to see how different features influence the decision-making process of individual trees within a Random Forest model. ")

    st.markdown("---")

    # Sidebar for user inputs
    st.sidebar.header("User Input")
    uploaded_file = st.sidebar.file_uploader("Upload your CSV file", type=["csv"])

    if uploaded_file is not None:
        try:
            data = pd.read_csv(uploaded_file)
            st.subheader("Uploaded Data")
            st.dataframe(data.head())

            # Get column names for selection
            all_columns = data.columns.tolist()
            if not all_columns:
                st.error("The CSV file appears to be empty or malformed.")
                return

            # Feature and target selection
            target_column = st.sidebar.selectbox("Select the Target Variable (y)", all_columns)
            feature_columns = st.sidebar.multiselect("Select the Feature Variables (X)", all_columns, default=[col for col in all_columns if col != target_column])

            if not feature_columns:
                st.sidebar.warning("Please select at least one feature.")
            else:
                # Prepare data
                X = data[feature_columns]
                y = data[target_column]

                # Convert all data to numeric, handling potential errors
                try:
                    X = X.apply(pd.to_numeric, errors='coerce').fillna(0)
                    y = y.apply(pd.to_numeric, errors='coerce').fillna(0)
                except Exception as e:
                    st.error(f"Error converting data to numeric. Please ensure your data is clean. Details: {e}")
                    return

                # Split data
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

                # Hyperparameter tuning
                st.sidebar.subheader("Random Forest Parameters")
                n_estimators = st.sidebar.slider("Number of Trees", 1, 50, 10, help="The number of decision trees in the forest.")
                max_depth = st.sidebar.slider("Max Tree Depth", 1, 10, 3, help="The maximum depth of each tree.")

                # Train the model
                rf_classifier = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
                rf_classifier.fit(X_train, y_train)

                st.markdown("---")

                # --- Section 1: Overall Model Performance and Feature Importance ---
                st.header("1. Overall Model Analysis")
                st.write("First, let's see how the model performed and which features were most influential across the entire forest.")
                
                # Accuracy score
                score = rf_classifier.score(X_test, y_test)
                st.metric("Model Accuracy", f"{score:.2f}")
                
                # Feature importance plot
                st.subheader("Feature Importance")
                st.write("This chart shows the relative importance of each feature for the model's predictions. The most important features are at the top.")
                feature_importance = pd.Series(rf_classifier.feature_importances_, index=feature_columns).sort_values(ascending=False)
                fig_importance, ax_importance = plt.subplots(figsize=(10, 6))
                feature_importance.plot(kind='barh', ax=ax_importance, color='teal')
                ax_importance.set_xlabel("Feature Importance Score")
                ax_importance.set_ylabel("Features")
                plt.tight_layout()
                st.pyplot(fig_importance)

                st.markdown("---")

                # --- Section 2: Visualizing Individual Trees ---
                st.header("2. Exploring a Single Decision Tree")
                st.write("A Random Forest is a collection of many decision trees. Let's visualize one of them to understand how a single tree uses features to make a decision. You can select which tree to view.")

                tree_index = st.slider("Select a Tree to Visualize", 0, n_estimators - 1, 0)
                
                st.subheader(f"Tree Visualization (Tree #{tree_index})")
                
                tree_to_visualize = rf_classifier.estimators_[tree_index]
                
                fig_tree, ax_tree = plt.subplots(figsize=(30, 15))
                plot_tree(tree_to_visualize,
                          filled=True,
                          feature_names=feature_columns,
                          class_names=[str(c) for c in np.unique(y)],
                          rounded=True,
                          ax=ax_tree)
                ax_tree.set_title(f"Decision Tree #{tree_index} from the Random Forest")
                st.pyplot(fig_tree)

        except Exception as e:
            st.error(f"An error occurred: {e}. Please check your file format and data.")

# Run the app
if __name__ == "__main__":
    run_random_forest_app()