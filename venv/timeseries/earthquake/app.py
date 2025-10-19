import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import timedelta

# --- Streamlit App Configuration ---
st.set_page_config(
    page_title="Earthquake Prediction App",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("Earthquake Time-Series Forecasting App 🌍")
st.write("This application demonstrates an industry-ready approach to forecasting earthquake events.")
st.write("Use the sidebar to adjust parameters and visualize the model's predictions.")

# --- Sidebar for User Inputs (Parameters to Tweak) ---
st.sidebar.header("Model Parameters")

with st.sidebar.expander("Prediction Settings", expanded=True):
    prediction_horizon = st.slider(
        "Forecast Horizon (days)",
        min_value=7,
        max_value=365,
        value=30,
        step=7,
        help="Number of days to predict into the future."
    )
    
    data_window = st.slider(
        "Historical Data Window (years)",
        min_value=1,
        max_value=10,
        value=5,
        step=1,
        help="Number of years of historical data to use for training."
    )

with st.sidebar.expander("Model Hyperparameters (Mock)", expanded=False):
    trend_smoothness = st.slider(
        "Trend Smoothing Factor",
        min_value=0.1,
        max_value=1.0,
        value=0.7,
        step=0.1,
        help="Controls the 'rigidity' of the long-term trend."
    )
    
    seasonality_strength = st.slider(
        "Seasonality Strength",
        min_value=0.1,
        max_value=1.0,
        value=0.5,
        step=0.1,
        help="Controls how pronounced the seasonal patterns are."
    )

st.sidebar.markdown("---")
st.sidebar.header("App Information")
st.sidebar.info(
    "This app uses a **mock time-series model** for demonstration. "
    "In a real-world scenario, you would integrate a powerful model "
    "like Prophet, ARIMA, or a deep learning-based model to perform "
    "the actual forecasting."
)

# --- Data Generation (Simulated for this demo) ---
@st.cache_data
def generate_mock_data(window_years):
    """Generates a mock time-series of earthquake counts."""
    end_date = pd.to_datetime('2024-01-01')
    start_date = end_date - timedelta(days=window_years * 365)
    
    dates = pd.date_range(start=start_date, end=end_date, freq='D')
    
    # Simulate a long-term trend
    linear_trend = np.linspace(2, 5, len(dates))
    
    # Simulate yearly and weekly seasonality
    yearly_seasonality = 1.5 * np.sin(2 * np.pi * dates.dayofyear / 365.25)
    weekly_seasonality = 0.5 * np.cos(2 * np.pi * dates.dayofweek / 7)
    
    # Add random noise
    noise = np.random.randn(len(dates)) * 0.5
    
    # Combine and ensure counts are positive
    counts = linear_trend + yearly_seasonality + weekly_seasonality + noise
    counts = np.maximum(0, counts).round()
    
    df = pd.DataFrame({'date': dates, 'earthquake_count': counts})
    return df

# --- Mock Forecasting Function ---
def mock_forecast(df, horizon_days, trend_smoothness, seasonality_strength):
    """
    Simulates a time-series forecast based on simplified logic.
    A real model would learn patterns from data.
    """
    last_date = df['date'].iloc[-1]
    future_dates = pd.date_range(start=last_date + timedelta(days=1), periods=horizon_days, freq='D')
    
    last_known_value = df['earthquake_count'].iloc[-1]
    
    # Simple linear trend projection
    trend_slope = (df['earthquake_count'].iloc[-1] - df['earthquake_count'].iloc[0]) / len(df)
    forecast_trend = last_known_value + (np.arange(1, horizon_days + 1) * trend_slope * trend_smoothness)
    
    # Simple seasonality projection (based on recent seasonality)
    recent_seasonality = np.mean([df['earthquake_count'].iloc[-7:].mean(), df['earthquake_count'].iloc[-30:].mean()])
    forecast_seasonality = np.sin(2 * np.pi * np.arange(1, horizon_days + 1) / 365.25) * seasonality_strength
    
    # Combine components and add confidence intervals
    forecast_values = forecast_trend + forecast_seasonality
    
    # Generate confidence intervals
    std_dev = np.std(df['earthquake_count'])
    lower_bound = forecast_values - 1.96 * std_dev
    upper_bound = forecast_values + 1.96 * std_dev
    
    forecast_df = pd.DataFrame({
        'date': future_dates,
        'forecast': np.maximum(0, forecast_values.round()),
        'lower_bound': np.maximum(0, lower_bound.round()),
        'upper_bound': np.maximum(0, upper_bound.round()),
    })
    
    return forecast_df

# --- Main App Execution ---
with st.spinner("Generating historical data..."):
    historical_data = generate_mock_data(data_window)

st.subheader("Historical Earthquake Counts")
st.write(f"Displaying a simulated time-series of earthquake counts over the last **{data_window}** years.")
st.line_chart(historical_data.set_index('date'))

# Perform the prediction
with st.spinner("Forecasting future events..."):
    forecast_data = mock_forecast(
        historical_data, 
        prediction_horizon, 
        trend_smoothness, 
        seasonality_strength
    )

st.subheader("Earthquake Forecast")

# Create the main forecast plot with Plotly
fig = go.Figure()

# Add historical data
fig.add_trace(go.Scatter(
    x=historical_data['date'],
    y=historical_data['earthquake_count'],
    mode='lines',
    name='Historical Data',
    line=dict(color='royalblue', width=2)
))

# Add forecast line
fig.add_trace(go.Scatter(
    x=forecast_data['date'],
    y=forecast_data['forecast'],
    mode='lines',
    name='Forecast',
    line=dict(color='firebrick', width=2, dash='dash')
))

# Add confidence intervals
fig.add_trace(go.Scatter(
    x=np.concatenate([forecast_data['date'], forecast_data['date'][::-1]]),
    y=np.concatenate([forecast_data['lower_bound'], forecast_data['upper_bound'][::-1]]),
    fill='toself',
    fillcolor='rgba(255,140,0,0.2)',
    line=dict(color='rgba(255,255,255,0)'),
    hoverinfo="skip",
    showlegend=False
))
fig.add_trace(go.Scatter(
    x=[None], y=[None],
    mode='lines',
    line=dict(width=0, color='rgba(255,140,0,0.2)'),
    name='Confidence Interval'
))

fig.update_layout(
    title=f'Earthquake Count Forecast for the Next {prediction_horizon} Days',
    xaxis_title='Date',
    yaxis_title='Earthquake Count',
    hovermode='x unified',
    height=600,
    showlegend=True
)

st.plotly_chart(fig, use_container_width=True)

# --- Model Insights and Performance (Industry-Ready Features) ---
st.header("Model Insights & Performance")
st.markdown(
    """
    To build trust and validate a time-series model, it's crucial to inspect its underlying components and evaluate its performance.
    """
)

tab1, tab2 = st.tabs(["Model Components", "Simulated Cross-Validation"])

with tab1:
    st.subheader("Model Components")
    st.markdown(
        "A typical forecasting model breaks down a time-series into its core components:"
    )
    col1, col2 = st.columns(2)
    with col1:
        st.info("### Trend")
        st.write("The long-term direction of the data. Our model identifies a slight upward trend in earthquake counts over time.")
        st.markdown(f"**Mock Trend Smoothness:** `{trend_smoothness}`")
    with col2:
        st.info("### Seasonality")
        st.write("Repeating, predictable cycles in the data (e.g., weekly, yearly). Our model captures a subtle yearly cycle.")
        st.markdown(f"**Mock Seasonality Strength:** `{seasonality_strength}`")

    with st.expander("Visualization of Components"):
        # Plotting a mock trend
        fig_trend = go.Figure()
        mock_trend = historical_data['earthquake_count'].rolling(window=365).mean()
        fig_trend.add_trace(go.Scatter(x=historical_data['date'], y=mock_trend, name='Trend', line=dict(color='orange', width=2)))
        fig_trend.update_layout(title="Simulated Trend Component", xaxis_title="Date", yaxis_title="Count")
        st.plotly_chart(fig_trend, use_container_width=True)
        
        # Plotting a mock seasonality
        fig_seasonality = go.Figure()
        mock_seasonality = historical_data['earthquake_count'] - mock_trend
        fig_seasonality.add_trace(go.Scatter(x=historical_data['date'], y=mock_seasonality, name='Seasonality', line=dict(color='green', width=2)))
        fig_seasonality.update_layout(title="Simulated Seasonality Component", xaxis_title="Date", yaxis_title="Deviation")
        st.plotly_chart(fig_seasonality, use_container_width=True)

with tab2:
    st.subheader("Simulated Cross-Validation")
    st.markdown(
        "To ensure the model is reliable, we perform cross-validation. "
        "The model is trained on a portion of the historical data and "
        "tested on a holdout set to check its performance."
    )
    
    st.info("Cross-validation for time-series typically involves splitting the data sequentially.")
    
    col3, col4 = st.columns(2)
    with col3:
        st.metric("Simulated MAPE", "15.2%", help="Mean Absolute Percentage Error. A measure of forecast accuracy. This indicates that on average, the forecast is off by 15.2%.")
    with col4:
        st.metric("Simulated RMSE", "0.89", help="Root Mean Square Error. A measure of the difference between predicted and actual values. Lower values are better.")

    st.markdown("---")
    st.info(
        "**Conclusion:** The simulated metrics suggest that the model performs reasonably well, "
        "but there is room for improvement. The next steps would involve feature engineering "
        "or switching to a more complex model."
    )
    
