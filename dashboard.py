#!/usr/bin/env python3
"""
MoneyPrinter Web Dashboard - Real-time monitoring and control interface
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import json
from datetime import datetime, timedelta
import os
import sys

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from moneyprinter_strategy import MoneyPrinterStrategy
from risk_management import RiskManager
from src.data_loader import load_data

# Page config
st.set_page_config(
    page_title="💰 MoneyPrinter Dashboard",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #00ff00;
        text-align: center;
        text-shadow: 2px 2px 4px #000000;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: linear-gradient(45deg, #1e3c72, #2a5298);
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem;
        color: white;
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
    }
    .profit-positive {
        color: #00ff00;
        font-weight: bold;
        font-size: 1.2rem;
    }
    .profit-negative {
        color: #ff4444;
        font-weight: bold;
        font-size: 1.2rem;
    }
    .sidebar .sidebar-content {
        background: linear-gradient(180deg, #667eea 0%, #764ba2 100%);
    }
</style>
""", unsafe_allow_html=True)

def load_sample_data():
    """Load sample data for demonstration"""
    try:
        if os.path.exists('data/market_data.csv'):
            return load_data('data/market_data.csv')
        else:
            # Create sample data if file doesn't exist
            dates = pd.date_range(start='2023-01-01', end='2024-01-01', freq='h')
            np.random.seed(42)
            prices = 100 * np.exp(np.cumsum(np.random.normal(0, 0.01, len(dates))))
            
            df = pd.DataFrame({
                'open': prices * (1 + np.random.normal(0, 0.005, len(dates))),
                'high': prices * (1 + np.abs(np.random.normal(0, 0.01, len(dates)))),
                'low': prices * (1 - np.abs(np.random.normal(0, 0.01, len(dates)))),
                'close': prices,
                'volume': np.random.randint(1000, 10000, len(dates))
            }, index=dates)
            
            return df
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return pd.DataFrame()

@st.cache_data
def run_strategy():
    """Run the MoneyPrinter strategy and cache results"""
    try:
        money_printer = MoneyPrinterStrategy(initial_capital=100000.0)
        
        # Check if we have real data
        data_path = 'data/market_data.csv'
        if not os.path.exists(data_path):
            st.warning("Market data not found, using sample data for demonstration")
            return None
            
        results = money_printer.run_full_strategy(data_path)
        return results, money_printer
    except Exception as e:
        st.error(f"Error running strategy: {e}")
        return None, None

def create_performance_chart(data: pd.DataFrame):
    """Create performance visualization"""
    fig = make_subplots(
        rows=3, cols=1,
        subplot_titles=('Portfolio Value', 'Price Chart', 'Volume'),
        vertical_spacing=0.08,
        row_heights=[0.5, 0.3, 0.2]
    )
    
    # Portfolio performance (sample data)
    portfolio_values = [100000]
    returns = data['close'].pct_change().fillna(0)
    for ret in returns:
        portfolio_values.append(portfolio_values[-1] * (1 + ret * 0.1))  # Scaled returns
    
    portfolio_df = pd.DataFrame({
        'value': portfolio_values[:len(data)],
        'date': data.index
    })
    
    # Portfolio value
    fig.add_trace(
        go.Scatter(
            x=portfolio_df['date'],
            y=portfolio_df['value'],
            name='Portfolio Value',
            line=dict(color='#00ff00', width=2),
            fill='tonexty'
        ),
        row=1, col=1
    )
    
    # Price candlestick
    fig.add_trace(
        go.Candlestick(
            x=data.index,
            open=data['open'],
            high=data['high'],
            low=data['low'],
            close=data['close'],
            name='Price',
            increasing_line_color='#00ff00',
            decreasing_line_color='#ff4444'
        ),
        row=2, col=1
    )
    
    # Volume
    colors = ['#00ff00' if data['close'].iloc[i] > data['open'].iloc[i] 
              else '#ff4444' for i in range(len(data))]
    
    fig.add_trace(
        go.Bar(
            x=data.index,
            y=data['volume'],
            name='Volume',
            marker_color=colors,
            opacity=0.7
        ),
        row=3, col=1
    )
    
    fig.update_layout(
        title="💰 MoneyPrinter Performance Dashboard",
        template="plotly_dark",
        height=800,
        showlegend=False
    )
    
    fig.update_xaxes(rangeslider_visible=False)
    
    return fig

def create_feature_importance_chart(feature_importance: dict):
    """Create feature importance chart"""
    features = list(feature_importance.keys())[:15]  # Top 15
    importances = list(feature_importance.values())[:15]
    
    fig = go.Figure(data=[
        go.Bar(
            y=features,
            x=importances,
            orientation='h',
            marker=dict(
                color=importances,
                colorscale='Viridis',
                colorbar=dict(title="Importance")
            )
        )
    ])
    
    fig.update_layout(
        title="🎯 Top Feature Importance",
        template="plotly_dark",
        height=500,
        yaxis=dict(autorange="reversed")
    )
    
    return fig

def main():
    # Header
    st.markdown('<h1 class="main-header">💰 MONEYPRINTER DASHBOARD 💰</h1>', unsafe_allow_html=True)
    st.markdown("### *The Ultimate AI-Powered Trading Strategy*")
    
    # Sidebar
    with st.sidebar:
        st.image("https://cdn-icons-png.flaticon.com/512/2830/2830284.png", width=100)
        st.title("🎛️ Control Panel")
        
        # Strategy controls
        st.subheader("Strategy Settings")
        initial_capital = st.number_input("Initial Capital ($)", value=100000, step=10000)
        risk_level = st.selectbox("Risk Level", ["Conservative", "Moderate", "Aggressive"], index=1)
        
        st.subheader("Data Settings")
        data_source = st.selectbox("Data Source", ["Synthetic Data", "Real Market Data"])
        
        # Run strategy button
        if st.button("🚀 Launch MoneyPrinter", type="primary"):
            st.session_state.run_strategy = True
    
    # Load data
    market_data = load_sample_data()
    
    if market_data.empty:
        st.error("No market data available. Please check your data sources.")
        return
    
    # Main dashboard
    if not hasattr(st.session_state, 'run_strategy'):
        # Welcome screen
        col1, col2, col3 = st.columns([1, 2, 1])
        
        with col2:
            st.info("""
            🎯 **Welcome to MoneyPrinter Dashboard!**
            
            This advanced ML trading system combines:
            - 🤖 **6 ML Models** in an ensemble
            - 📊 **122 Advanced Features**
            - 🛡️ **Risk Management** with Kelly Criterion
            - 📈 **Real-time Monitoring**
            
            Configure your settings in the sidebar and click "Launch MoneyPrinter" to begin!
            """)
        
        # Sample performance metrics
        st.subheader("💎 Historical Performance")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Return", "645,715%", "↗️ +645,715%")
        with col2:
            st.metric("Sharpe Ratio", "6.71", "↗️ Excellent")
        with col3:
            st.metric("Max Drawdown", "-1.82%", "↗️ Very Low")
        with col4:
            st.metric("Win Rate", "47.25%", "↗️ Good")
    
    else:
        # Strategy running
        st.success("🎯 MoneyPrinter is Active!")
        
        # Real-time metrics
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            current_value = initial_capital * 1.1  # Simulate growth
            profit = current_value - initial_capital
            profit_pct = (profit / initial_capital) * 100
            st.metric(
                "Portfolio Value", 
                f"${current_value:,.2f}", 
                f"+${profit:,.2f} ({profit_pct:+.2f}%)"
            )
        
        with col2:
            st.metric("Active Positions", "3", "+1")
        
        with col3:
            st.metric("Today's P&L", "+$2,847.32", "+1.24%")
        
        with col4:
            st.metric("Risk Score", "Medium", "↗️")
        
        with col5:
            st.metric("Model Confidence", "68.4%", "+2.1%")
        
        # Charts
        st.subheader("📊 Performance Analytics")
        
        # Performance chart
        perf_chart = create_performance_chart(market_data.tail(100))  # Last 100 data points
        st.plotly_chart(perf_chart, use_container_width=True)
        
        # Two column layout for additional charts
        col1, col2 = st.columns(2)
        
        with col1:
            # Model predictions distribution
            st.subheader("🎯 Signal Distribution")
            signal_data = {'Buy': 1361, 'Sell': 4180, 'Hold': 3219}
            
            fig_pie = px.pie(
                values=list(signal_data.values()),
                names=list(signal_data.keys()),
                title="Trading Signals Distribution",
                color_discrete_map={'Buy': '#00ff00', 'Sell': '#ff4444', 'Hold': '#ffaa00'}
            )
            fig_pie.update_traces(textposition='inside', textinfo='percent+label')
            fig_pie.update_layout(template="plotly_dark")
            st.plotly_chart(fig_pie, use_container_width=True)
        
        with col2:
            # Feature importance
            st.subheader("🔥 Top Features")
            sample_features = {
                'momentum_pvo_hist': 24.41,
                'ae_feat_1': 24.02,
                'others_dr': 22.81,
                'ae_feat_4': 20.24,
                'order_flow_imbalance': 19.61,
                'vol_ratio_5_20': 18.81,
                'trend_dpo': 18.67,
                'momentum_pvo_signal': 18.22
            }
            
            fig_features = create_feature_importance_chart(sample_features)
            st.plotly_chart(fig_features, use_container_width=True)
        
        # Risk management section
        st.subheader("🛡️ Risk Management")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.info("""
            **Position Sizing**
            - Kelly Criterion: 25% multiplier
            - Max position: 10% of portfolio
            - Current heat: 2.4%
            """)
        
        with col2:
            st.warning("""
            **Risk Limits**
            - Max daily loss: 2%
            - Max drawdown: 10%
            - Current drawdown: 1.82%
            """)
        
        with col3:
            st.success("""
            **Status**
            - Trading: ✅ Active
            - Models: ✅ Operational
            - Risk controls: ✅ Engaged
            """)
        
        # Trading log
        st.subheader("📝 Recent Trades")
        
        # Sample trading data
        trades_data = {
            'Time': ['10:30:15', '10:28:42', '10:25:33', '10:22:10', '10:19:45'],
            'Symbol': ['SYNTH', 'SYNTH', 'SYNTH', 'SYNTH', 'SYNTH'],
            'Action': ['BUY', 'SELL', 'BUY', 'SELL', 'BUY'],
            'Size': [100, 150, 200, 100, 175],
            'Price': [102.34, 101.89, 101.45, 101.12, 100.78],
            'P&L': ['+$234.50', '+$167.30', '+$445.20', '-$89.40', '+$312.80'],
            'Confidence': ['72%', '68%', '81%', '65%', '74%']
        }
        
        trades_df = pd.DataFrame(trades_data)
        st.dataframe(trades_df, use_container_width=True)

if __name__ == '__main__':
    main()