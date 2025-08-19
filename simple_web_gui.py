#!/usr/bin/env python3
"""
Simple Web GUI for Asset ML Strategy
A streamlined web interface that works with core dependencies only
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import io
import os

# Set page configuration
st.set_page_config(
    page_title="💰 Asset ML Strategy",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #FF6B6B, #4ECDC4);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: #f0f2f6;
        padding: 1rem;
        border-radius: 5px;
        border-left: 4px solid #4ECDC4;
    }
</style>
""", unsafe_allow_html=True)

def generate_sample_data():
    """Generate sample financial data for demonstration"""
    dates = pd.date_range(start='2020-01-01', end='2024-01-01', freq='D')
    np.random.seed(42)
    
    # Generate realistic stock price data
    base_price = 100
    prices = [base_price]
    
    for i in range(len(dates) - 1):
        change = np.random.normal(0, 0.02)  # 2% daily volatility
        new_price = prices[-1] * (1 + change)
        prices.append(max(new_price, 1))  # Ensure price stays positive
    
    data = pd.DataFrame({
        'Date': dates,
        'Open': [p * np.random.uniform(0.98, 1.02) for p in prices],
        'High': [p * np.random.uniform(1.00, 1.05) for p in prices],
        'Low': [p * np.random.uniform(0.95, 1.00) for p in prices],
        'Close': prices,
        'Adj Close': prices,
        'Volume': np.random.randint(1000000, 10000000, len(dates))
    })
    
    return data

def add_technical_indicators(data):
    """Add technical indicators to the data"""
    data = data.copy()
    
    # Simple Moving Averages
    data['SMA_5'] = data['Close'].rolling(window=5).mean()
    data['SMA_10'] = data['Close'].rolling(window=10).mean()
    data['SMA_20'] = data['Close'].rolling(window=20).mean()
    
    # Volatility (20-day rolling standard deviation)
    data['Volatility'] = data['Close'].rolling(window=20).std()
    
    # Price changes
    data['Price_Change'] = data['Close'].pct_change()
    data['Price_Change_5d'] = data['Close'].pct_change(periods=5)
    
    # Volume moving average
    data['Volume_MA'] = data['Volume'].rolling(window=20).mean()
    
    return data

def train_ml_model(data, target_col='Close', test_size=0.2):
    """Train a machine learning model"""
    # Prepare features
    feature_cols = ['Open', 'High', 'Low', 'Adj Close', 'Volume', 'SMA_5', 'SMA_10', 'SMA_20', 'Volatility', 'Price_Change']
    
    # Remove rows with NaN values
    clean_data = data.dropna()
    
    if len(clean_data) < 10:
        st.error("Not enough data for training (need at least 10 rows after cleaning)")
        return None, None, {}
    
    X = clean_data[feature_cols]
    y = clean_data[target_col]
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
    
    # Train model
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Predictions
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)
    
    # Metrics
    metrics = {
        'train_mse': mean_squared_error(y_train, y_pred_train),
        'test_mse': mean_squared_error(y_test, y_pred_test),
        'train_r2': r2_score(y_train, y_pred_train),
        'test_r2': r2_score(y_test, y_pred_test),
        'feature_importance': dict(zip(feature_cols, model.feature_importances_))
    }
    
    return model, {'X_test': X_test, 'y_test': y_test, 'y_pred': y_pred_test}, metrics

def main():
    """Main Streamlit application"""
    
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>💰 Asset ML Strategy - Web Interface</h1>
        <p>Complete local machine learning tool for financial data analysis</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar
    st.sidebar.title("📊 Controls")
    
    # Data source selection
    data_source = st.sidebar.selectbox(
        "Data Source",
        ["Generate Sample Data", "Upload Excel File"]
    )
    
    # Initialize data
    data = None
    
    if data_source == "Generate Sample Data":
        st.sidebar.success("Using generated sample data")
        data = generate_sample_data()
    
    elif data_source == "Upload Excel File":
        uploaded_file = st.sidebar.file_uploader(
            "Upload Excel file",
            type=['xlsx', 'xls'],
            help="File must contain columns: Date, Open, High, Low, Close, Adj Close, Volume"
        )
        
        if uploaded_file is not None:
            try:
                data = pd.read_excel(uploaded_file)
                st.sidebar.success(f"Loaded {len(data)} rows")
            except Exception as e:
                st.sidebar.error(f"Error loading file: {e}")
    
    if data is not None:
        # Add technical indicators
        data = add_technical_indicators(data)
        
        # Main content tabs
        tab1, tab2, tab3, tab4 = st.tabs(["📊 Data Overview", "📈 Visualizations", "🤖 ML Analysis", "📋 Results"])
        
        with tab1:
            st.subheader("📊 Data Overview")
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Total Records", len(data))
            with col2:
                st.metric("Date Range", f"{(data['Date'].max() - data['Date'].min()).days} days")
            with col3:
                st.metric("Avg Close Price", f"${data['Close'].mean():.2f}")
            with col4:
                st.metric("Price Volatility", f"{data['Close'].std():.2f}")
            
            # Display data
            st.subheader("Raw Data (Last 100 rows)")
            st.dataframe(data.tail(100), use_container_width=True)
            
            # Data statistics
            st.subheader("Statistical Summary")
            st.dataframe(data.describe(), use_container_width=True)
        
        with tab2:
            st.subheader("📈 Visualizations")
            
            # Price chart
            st.subheader("Price Chart with Moving Averages")
            fig_price = go.Figure()
            
            fig_price.add_trace(go.Scatter(
                x=data['Date'], y=data['Close'],
                mode='lines', name='Close Price',
                line=dict(color='blue', width=2)
            ))
            
            fig_price.add_trace(go.Scatter(
                x=data['Date'], y=data['SMA_20'],
                mode='lines', name='SMA 20',
                line=dict(color='red', width=1)
            ))
            
            fig_price.add_trace(go.Scatter(
                x=data['Date'], y=data['SMA_5'],
                mode='lines', name='SMA 5',
                line=dict(color='green', width=1)
            ))
            
            fig_price.update_layout(
                title="Stock Price with Moving Averages",
                xaxis_title="Date",
                yaxis_title="Price ($)",
                height=500
            )
            
            st.plotly_chart(fig_price, use_container_width=True)
            
            # Volume chart
            st.subheader("Volume Chart")
            fig_volume = px.bar(data, x='Date', y='Volume', title="Trading Volume")
            fig_volume.update_layout(height=400)
            st.plotly_chart(fig_volume, use_container_width=True)
            
            # Correlation matrix
            st.subheader("Correlation Matrix")
            numeric_cols = ['Open', 'High', 'Low', 'Close', 'Volume', 'SMA_5', 'SMA_10', 'SMA_20', 'Volatility']
            corr_data = data[numeric_cols].corr()
            
            fig_corr = px.imshow(corr_data, 
                               text_auto=True,
                               aspect="auto",
                               title="Feature Correlation Matrix")
            st.plotly_chart(fig_corr, use_container_width=True)
        
        with tab3:
            st.subheader("🤖 ML Analysis")
            
            # ML Configuration
            col1, col2 = st.columns(2)
            
            with col1:
                target_column = st.selectbox(
                    "Target Variable",
                    ['Close', 'High', 'Low', 'Open', 'Adj Close'],
                    index=0
                )
            
            with col2:
                test_size = st.slider(
                    "Test Set Size",
                    min_value=0.1,
                    max_value=0.5,
                    value=0.2,
                    step=0.1
                )
            
            if st.button("🚀 Train Model", type="primary"):
                with st.spinner("Training machine learning model..."):
                    model, predictions, metrics = train_ml_model(data, target_column, test_size)
                
                if model is not None:
                    st.success("✅ Model trained successfully!")
                    
                    # Display metrics
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        st.metric("Training R²", f"{metrics['train_r2']:.4f}")
                    with col2:
                        st.metric("Testing R²", f"{metrics['test_r2']:.4f}")
                    with col3:
                        st.metric("Training MSE", f"{metrics['train_mse']:.4f}")
                    with col4:
                        st.metric("Testing MSE", f"{metrics['test_mse']:.4f}")
                    
                    # Feature importance
                    st.subheader("Feature Importance")
                    importance_df = pd.DataFrame(
                        list(metrics['feature_importance'].items()),
                        columns=['Feature', 'Importance']
                    ).sort_values('Importance', ascending=False)
                    
                    fig_importance = px.bar(
                        importance_df,
                        x='Importance',
                        y='Feature',
                        orientation='h',
                        title="Feature Importance"
                    )
                    st.plotly_chart(fig_importance, use_container_width=True)
                    
                    # Store results in session state
                    st.session_state['model'] = model
                    st.session_state['predictions'] = predictions
                    st.session_state['metrics'] = metrics
        
        with tab4:
            st.subheader("📋 Analysis Results")
            
            if 'model' in st.session_state and 'predictions' in st.session_state:
                predictions = st.session_state['predictions']
                metrics = st.session_state['metrics']
                
                # Predictions vs Actual chart
                st.subheader("Predictions vs Actual Values")
                
                comparison_df = pd.DataFrame({
                    'Actual': predictions['y_test'],
                    'Predicted': predictions['y_pred']
                })
                
                fig_pred = px.scatter(
                    comparison_df,
                    x='Actual',
                    y='Predicted',
                    title="Predictions vs Actual Values"
                )
                
                # Add perfect prediction line
                min_val = min(comparison_df['Actual'].min(), comparison_df['Predicted'].min())
                max_val = max(comparison_df['Actual'].max(), comparison_df['Predicted'].max())
                fig_pred.add_shape(
                    type="line",
                    x0=min_val, y0=min_val,
                    x1=max_val, y1=max_val,
                    line=dict(color="red", dash="dash")
                )
                
                st.plotly_chart(fig_pred, use_container_width=True)
                
                # Model summary
                st.subheader("Model Summary")
                st.write(f"**Model Type:** Random Forest Regressor")
                st.write(f"**Training Samples:** {len(predictions['X_test']) * (1/(test_size if test_size > 0 else 0.2) - 1):.0f}")
                st.write(f"**Testing Samples:** {len(predictions['X_test'])}")
                st.write(f"**Test R² Score:** {metrics['test_r2']:.4f}")
                st.write(f"**Test MSE:** {metrics['test_mse']:.4f}")
                
                # Download results
                if st.button("📥 Download Predictions"):
                    csv = comparison_df.to_csv(index=False)
                    st.download_button(
                        label="Download CSV",
                        data=csv,
                        file_name=f"predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv"
                    )
            else:
                st.info("👆 Train a model in the ML Analysis tab to see results here.")
    
    else:
        st.info("👈 Please select a data source from the sidebar to begin analysis.")
    
    # Footer
    st.markdown("---")
    st.markdown("💡 **Asset ML Strategy** - Free, local machine learning tool for financial data analysis")

if __name__ == "__main__":
    main()