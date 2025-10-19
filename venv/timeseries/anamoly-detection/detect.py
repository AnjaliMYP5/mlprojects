import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

# --- Streamlit App Configuration ---
st.set_page_config(
    page_title="ECG Anomaly Detection",
    page_icon="❤️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("ECG Anomaly Detection App ❤️")
st.write("This application helps you visualize an ECG signal and detect anomalies based on a statistical approach.")

# --- Sidebar for File Upload and Instructions ---
st.sidebar.header("Instructions")
st.sidebar.write("1. Upload a CSV file with your ECG data.")
st.sidebar.write("2. The CSV file should have at least two columns: 'time' and 'signal'.")
st.sidebar.markdown("---")
uploaded_file = st.sidebar.file_uploader("Upload a CSV file", type="csv")

# --- Main Application Logic ---
if uploaded_file is not None:
    try:
        # Load the data
        df = pd.read_csv(uploaded_file)
        
        # Check for required columns
        if 'time' not in df.columns or 'signal' not in df.columns:
            st.error("The uploaded CSV must contain 'time' and 'signal' columns.")
        else:
            # --- Mock Anomaly Detection Logic ---
            # In a real-world scenario, you would replace this with a trained model.
            # This mock logic detects anomalies as points more than 3 standard deviations from the mean.
            st.info("Using a simple statistical method to detect anomalies. For a real-world application, this would be replaced by a machine learning model.")
            
            signal = df['signal']
            mean_signal = np.mean(signal)
            std_signal = np.std(signal)
            
            # Define a threshold for anomaly detection
            threshold = 3 * std_signal
            
            # Find anomalies
            df['anomaly'] = np.abs(signal - mean_signal) > threshold
            anomalies = df[df['anomaly']]

            st.subheader("ECG Signal and Detected Anomalies")
            
            # --- Plotly Visualization ---
            fig = go.Figure()
            
            # Plot the main signal
            fig.add_trace(go.Scatter(
                x=df['time'],
                y=df['signal'],
                mode='lines',
                name='ECG Signal',
                line=dict(color='royalblue', width=1)
            ))
            
            # Plot the anomalies
            if not anomalies.empty:
                fig.add_trace(go.Scatter(
                    x=anomalies['time'],
                    y=anomalies['signal'],
                    mode='markers',
                    name='Anomaly',
                    marker=dict(color='red', size=8, symbol='circle')
                ))
            
            fig.update_layout(
                title='ECG Signal with Anomaly Highlighting',
                xaxis_title='Time',
                yaxis_title='Signal Value',
                hovermode='x unified',
                height=600
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # --- Display Anomaly Summary ---
            if not anomalies.empty:
                st.subheader("Detected Anomalies Summary")
                st.write(f"Found **{len(anomalies)}** anomalies.")
                st.dataframe(anomalies[['time', 'signal']])
            else:
                st.success("No anomalies were detected in the uploaded signal.")
                
    except Exception as e:
        st.error(f"An error occurred: {e}")

else:
    st.info("Please upload a CSV file to get started.")
    st.markdown("You can use a sample CSV with columns `time` and `signal` like this:")
    sample_data = {
        'time': range(100),
        'signal': np.sin(np.linspace(0, 10*np.pi, 100)) + np.random.randn(100)*0.1
    }
    sample_df = pd.DataFrame(sample_data)
    sample_df.loc[30, 'signal'] = 3.0 # Introduce an anomaly
    st.dataframe(sample_df.head())
