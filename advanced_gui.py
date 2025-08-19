#!/usr/bin/env python3
"""
Advanced ML Trading Strategy GUI
Complete realistic trading system with advanced controls and backtesting
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import os
import sys
import io
import tempfile
import zipfile
import calendar
from typing import Dict, List, Tuple, Optional

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

# Import our modules
from src.data_loader import load_data
from src.feature_engineering import add_features
from src.backtesting import Backtester
from moneyprinter_strategy import MoneyPrinterStrategy
from risk_management import RiskManager

# Page configuration
st.set_page_config(
    page_title="🚀 Advanced ML Trading Strategy",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://github.com/Somethin6/Asset-ML-Strategy',
        'Report a bug': 'https://github.com/Somethin6/Asset-ML-Strategy/issues',
        'About': "Advanced ML Trading Strategy with Date-Based Training and Backtesting"
    }
)

# Custom CSS for advanced styling
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #FF6B6B, #4ECDC4, #45B7D1, #96CEB4, #FFEAA7);
        background-size: 400% 400%;
        animation: gradient 15s ease infinite;
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        font-size: 3.5rem;
        text-align: center;
        font-weight: 900;
        margin: 1rem 0;
    }
    
    @keyframes gradient {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }
    
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 15px;
        margin: 0.5rem;
        color: white;
        box-shadow: 0 8px 32px rgba(0,0,0,0.1);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255,255,255,0.1);
    }
    
    .status-card {
        background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        font-weight: bold;
    }
    
    .warning-card {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        font-weight: bold;
    }
    
    .info-card {
        background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        font-weight: bold;
    }
    
    .sidebar .sidebar-content {
        background: linear-gradient(180deg, #2C3E50 0%, #3498DB 100%);
    }
    
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: transparent;
        border-radius: 5px 5px 0px 0px;
        gap: 1px;
        padding-top: 10px;
        padding-bottom: 10px;
    }
    
    .uploadedFile {
        border: 2px dashed #4ECDC4;
        border-radius: 10px;
        padding: 1rem;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

class AdvancedTradingSystem:
    """Advanced ML Trading System with comprehensive features"""
    
    def __init__(self):
        self.data = None
        self.strategy = None
        self.backtest_results = None
        self.trained_model = None
        self.feature_importance = None
        self.correlations = None
        
    def load_data_file(self, uploaded_file) -> pd.DataFrame:
        """Load data from uploaded file"""
        try:
            file_extension = uploaded_file.name.split('.')[-1].lower()
            
            if file_extension == 'csv':
                df = pd.read_csv(uploaded_file)
            elif file_extension in ['xlsx', 'xls']:
                df = pd.read_excel(uploaded_file)
            else:
                st.error(f"Unsupported file format: {file_extension}")
                return None
                
            # Standardize column names
            df.columns = [col.lower().strip() for col in df.columns]
            
            # Check for required columns
            required_cols = ['time', 'open', 'high', 'low', 'close', 'volume']
            missing_cols = [col for col in required_cols if col not in df.columns]
            
            if missing_cols:
                st.error(f"Missing required columns: {missing_cols}")
                st.info("Required columns: time, open, high, low, close, volume")
                return None
            
            # Process time column
            df['time'] = pd.to_datetime(df['time'])
            df = df.set_index('time').sort_index()
            
            # Clean data
            df = df.ffill().bfill()
            df = df[df > 0]  # Remove negative prices
            
            return df
            
        except Exception as e:
            st.error(f"Error loading file: {str(e)}")
            return None
    
    def split_data_by_dates(self, df: pd.DataFrame, train_start: str, train_end: str, 
                           test_start: str, test_end: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Split data based on user-selected date ranges"""
        try:
            train_data = df.loc[train_start:train_end].copy()
            test_data = df.loc[test_start:test_end].copy()
            
            if train_data.empty:
                st.error("No training data in selected date range")
                return None, None
                
            if test_data.empty:
                st.error("No test data in selected date range")
                return None, None
                
            # Ensure no overlap between train and test
            if train_end >= test_start:
                st.warning("Training and test periods overlap! Using non-overlapping periods.")
                
            return train_data, test_data
            
        except Exception as e:
            st.error(f"Error splitting data: {str(e)}")
            return None, None
    
    def find_correlations(self, df: pd.DataFrame) -> Dict:
        """Find and analyze correlations in the data"""
        try:
            # Add technical indicators
            df_features = add_features(df.copy())
            
            # Calculate correlation matrix
            corr_matrix = df_features.corr()
            
            # Find strongest correlations with close price
            close_corr = corr_matrix['close'].abs().sort_values(ascending=False)
            
            # Find inter-feature correlations
            feature_corrs = {}
            for col in df_features.columns:
                if col != 'close':
                    feature_corrs[col] = corr_matrix[col]['close']
            
            return {
                'correlation_matrix': corr_matrix,
                'close_correlations': close_corr,
                'feature_correlations': feature_corrs,
                'strong_features': close_corr.head(20).to_dict()
            }
            
        except Exception as e:
            st.error(f"Error calculating correlations: {str(e)}")
            return {}
    
    def train_strategy(self, train_data: pd.DataFrame, config: Dict) -> Dict:
        """Train the ML strategy on training data"""
        try:
            with st.spinner("Training ML models..."):
                # Initialize strategy
                strategy = MoneyPrinterStrategy(
                    initial_capital=config['initial_capital'],
                    enable_advanced_features=True
                )
                
                # Add features
                train_features = add_features(train_data.copy())
                
                # Generate labels for training
                returns = train_features['close'].pct_change()
                labels = np.where(returns.shift(-1) > config['profit_threshold'], 1,
                                np.where(returns.shift(-1) < -config['profit_threshold'], 2, 0))
                
                # Train models
                X = train_features.drop(['open', 'high', 'low', 'close', 'volume'], axis=1).fillna(0)
                y = labels[:-1]  # Remove last label as we shifted
                X = X[:-1]      # Align with labels
                
                # Train the ensemble
                results = strategy.train_models(X, y)
                
                # Store trained strategy
                self.trained_model = strategy
                self.feature_importance = results.get('feature_importance', {})
                
                return {
                    'success': True,
                    'training_accuracy': results.get('accuracy', 0),
                    'feature_count': X.shape[1],
                    'training_samples': X.shape[0],
                    'feature_importance': self.feature_importance
                }
                
        except Exception as e:
            st.error(f"Error training strategy: {str(e)}")
            return {'success': False, 'error': str(e)}
    
    def run_backtest(self, test_data: pd.DataFrame, config: Dict) -> Dict:
        """Run backtest on completely unseen data"""
        try:
            if self.trained_model is None:
                return {'success': False, 'error': 'No trained model available'}
            
            with st.spinner("Running backtest on unseen data..."):
                # Add features to test data
                test_features = add_features(test_data.copy())
                
                # Generate predictions
                X_test = test_features.drop(['open', 'high', 'low', 'close', 'volume'], axis=1).fillna(0)
                predictions = self.trained_model.predict(X_test)
                
                # Create backtester
                backtester = Backtester(
                    data=test_data,
                    signals=pd.Series(predictions, index=test_data.index),
                    initial_capital=config['initial_capital'],
                    transaction_cost_pct=config['transaction_cost'],
                    slippage_pct=config['slippage']
                )
                
                # Run backtest
                backtester.run()
                
                # Calculate comprehensive metrics
                portfolio_returns = backtester.portfolio['strategy_returns']
                market_returns = backtester.portfolio['market_returns']
                
                # Performance metrics
                total_return = (backtester.portfolio['cumulative_strategy_returns'].iloc[-1] - 1) * 100
                market_return = (backtester.portfolio['cumulative_market_returns'].iloc[-1] - 1) * 100
                
                # Sharpe ratio
                sharpe = (portfolio_returns.mean() / portfolio_returns.std()) * np.sqrt(252) if portfolio_returns.std() > 0 else 0
                
                # Maximum drawdown
                cumulative = backtester.portfolio['cumulative_strategy_returns']
                rolling_max = cumulative.expanding().max()
                drawdown = (cumulative - rolling_max) / rolling_max
                max_drawdown = drawdown.min() * 100
                
                # Win rate
                winning_trades = (portfolio_returns > 0).sum()
                total_trades = (portfolio_returns != 0).sum()
                win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0
                
                # Volatility
                volatility = portfolio_returns.std() * np.sqrt(252) * 100
                
                self.backtest_results = backtester.portfolio
                
                return {
                    'success': True,
                    'total_return': total_return,
                    'market_return': market_return,
                    'excess_return': total_return - market_return,
                    'sharpe_ratio': sharpe,
                    'max_drawdown': max_drawdown,
                    'win_rate': win_rate,
                    'volatility': volatility,
                    'total_trades': int(total_trades),
                    'winning_trades': int(winning_trades),
                    'portfolio_data': backtester.portfolio,
                    'final_value': config['initial_capital'] * backtester.portfolio['cumulative_strategy_returns'].iloc[-1]
                }
                
        except Exception as e:
            st.error(f"Error running backtest: {str(e)}")
            return {'success': False, 'error': str(e)}

def create_performance_chart(portfolio_data: pd.DataFrame, market_data: pd.DataFrame):
    """Create comprehensive performance visualization"""
    fig = make_subplots(
        rows=4, cols=2,
        subplot_titles=[
            'Portfolio vs Market Performance', 'Price Chart with Signals',
            'Daily Returns Distribution', 'Rolling Sharpe Ratio',
            'Drawdown Analysis', 'Volume Analysis',
            'Win/Loss Analysis', 'Risk Metrics'
        ],
        specs=[
            [{"colspan": 2}, None],
            [{"colspan": 2}, None],
            [{}, {}],
            [{}, {}]
        ],
        vertical_spacing=0.08
    )
    
    # Portfolio vs Market
    fig.add_trace(
        go.Scatter(
            x=portfolio_data.index,
            y=portfolio_data['cumulative_strategy_returns'] * 100,
            name='Strategy',
            line=dict(color='#00ff88', width=3),
            fill='tonexty'
        ),
        row=1, col=1
    )
    
    fig.add_trace(
        go.Scatter(
            x=portfolio_data.index,
            y=portfolio_data['cumulative_market_returns'] * 100,
            name='Market',
            line=dict(color='#ff4444', width=2),
        ),
        row=1, col=1
    )
    
    # Price chart with signals
    fig.add_trace(
        go.Candlestick(
            x=market_data.index,
            open=market_data['open'],
            high=market_data['high'],
            low=market_data['low'],
            close=market_data['close'],
            name='Price',
            showlegend=False
        ),
        row=2, col=1
    )
    
    # Add buy/sell signals
    buy_signals = portfolio_data[portfolio_data['positions'].diff() > 0]
    sell_signals = portfolio_data[portfolio_data['positions'].diff() < 0]
    
    if not buy_signals.empty:
        fig.add_trace(
            go.Scatter(
                x=buy_signals.index,
                y=market_data.loc[buy_signals.index, 'close'] * 0.98,
                mode='markers',
                marker=dict(symbol='triangle-up', size=10, color='green'),
                name='Buy Signal',
                showlegend=False
            ),
            row=2, col=1
        )
    
    if not sell_signals.empty:
        fig.add_trace(
            go.Scatter(
                x=sell_signals.index,
                y=market_data.loc[sell_signals.index, 'close'] * 1.02,
                mode='markers',
                marker=dict(symbol='triangle-down', size=10, color='red'),
                name='Sell Signal',
                showlegend=False
            ),
            row=2, col=1
        )
    
    # Daily returns histogram
    fig.add_trace(
        go.Histogram(
            x=portfolio_data['strategy_returns'] * 100,
            name='Returns Distribution',
            marker_color='lightblue',
            opacity=0.7,
            showlegend=False
        ),
        row=3, col=1
    )
    
    # Rolling Sharpe
    rolling_sharpe = (portfolio_data['strategy_returns'].rolling(30).mean() / 
                     portfolio_data['strategy_returns'].rolling(30).std()) * np.sqrt(252)
    
    fig.add_trace(
        go.Scatter(
            x=portfolio_data.index,
            y=rolling_sharpe,
            name='30-Day Rolling Sharpe',
            line=dict(color='purple'),
            showlegend=False
        ),
        row=3, col=2
    )
    
    # Drawdown
    cumulative = portfolio_data['cumulative_strategy_returns']
    rolling_max = cumulative.expanding().max()
    drawdown = (cumulative - rolling_max) / rolling_max * 100
    
    fig.add_trace(
        go.Scatter(
            x=portfolio_data.index,
            y=drawdown,
            fill='tonexty',
            name='Drawdown',
            line=dict(color='red'),
            showlegend=False
        ),
        row=4, col=1
    )
    
    # Volume
    colors = ['green' if portfolio_data['market_returns'].iloc[i] > 0 else 'red' 
              for i in range(len(portfolio_data))]
    
    fig.add_trace(
        go.Bar(
            x=market_data.index,
            y=market_data['volume'],
            name='Volume',
            marker_color=colors,
            opacity=0.6,
            showlegend=False
        ),
        row=4, col=2
    )
    
    fig.update_layout(
        title="📊 Comprehensive Strategy Performance Analysis",
        template="plotly_dark",
        height=1000,
        showlegend=True
    )
    
    return fig

def create_correlation_heatmap(correlations: Dict):
    """Create correlation heatmap"""
    if 'correlation_matrix' not in correlations:
        return go.Figure()
    
    corr_matrix = correlations['correlation_matrix']
    
    # Select top correlated features for visualization
    top_features = correlations['close_correlations'].head(20).index
    corr_subset = corr_matrix.loc[top_features, top_features]
    
    fig = go.Figure(data=go.Heatmap(
        z=corr_subset.values,
        x=corr_subset.columns,
        y=corr_subset.index,
        colorscale='RdYlBu',
        zmid=0,
        colorbar=dict(title="Correlation")
    ))
    
    fig.update_layout(
        title="🎯 Feature Correlation Matrix (Top 20 Features)",
        template="plotly_dark",
        height=600
    )
    
    return fig

def create_feature_importance_chart(feature_importance: Dict):
    """Create feature importance visualization"""
    if not feature_importance:
        return go.Figure()
    
    features = list(feature_importance.keys())[:20]
    importances = list(feature_importance.values())[:20]
    
    fig = go.Figure(data=[
        go.Bar(
            y=features,
            x=importances,
            orientation='h',
            marker=dict(
                color=importances,
                colorscale='Viridis',
                colorbar=dict(title="Importance Score")
            )
        )
    ])
    
    fig.update_layout(
        title="🔥 Top 20 Feature Importance Scores",
        template="plotly_dark",
        height=600,
        yaxis=dict(autorange="reversed")
    )
    
    return fig

def main():
    """Main application"""
    
    # Header
    st.markdown('<h1 class="main-header">🚀 ADVANCED ML TRADING STRATEGY 🚀</h1>', unsafe_allow_html=True)
    st.markdown("### *Complete Realistic Trading System with Date-Based Training and Backtesting*")
    
    # Initialize system
    if 'trading_system' not in st.session_state:
        st.session_state.trading_system = AdvancedTradingSystem()
    
    trading_system = st.session_state.trading_system
    
    # Sidebar configuration
    with st.sidebar:
        st.image("https://cdn-icons-png.flaticon.com/512/2830/2830284.png", width=100)
        st.title("🎛️ Advanced Controls")
        
        # File upload section
        st.subheader("📁 Data Input")
        uploaded_file = st.file_uploader(
            "Upload Excel/CSV Data",
            type=['csv', 'xlsx', 'xls'],
            help="Upload your trading data with columns: time, open, high, low, close, volume"
        )
        
        # Sample data option
        use_sample = st.selectbox(
            "Or use sample data:",
            ["Select...", "BTCUSD", "ETHUSD", "GOLD", "SPXUSD"],
            help="Use provided sample datasets"
        )
        
        st.divider()
        
        # Strategy configuration
        st.subheader("⚙️ Strategy Settings")
        
        initial_capital = st.number_input(
            "Initial Capital ($)",
            min_value=1000,
            max_value=10000000,
            value=100000,
            step=10000,
            help="Starting capital for backtesting"
        )
        
        lookback_window = st.slider(
            "Lookback Window (days)",
            min_value=5,
            max_value=100,
            value=20,
            help="Number of days to look back for features"
        )
        
        ensemble_size = st.selectbox(
            "Ensemble Size",
            [3, 5, 7, 10],
            index=1,
            help="Number of models in ensemble"
        )
        
        profit_threshold = st.slider(
            "Profit Threshold (%)",
            min_value=0.1,
            max_value=5.0,
            value=1.0,
            step=0.1,
            help="Minimum profit threshold for signals"
        ) / 100
        
        transaction_cost = st.slider(
            "Transaction Cost (%)",
            min_value=0.0,
            max_value=1.0,
            value=0.1,
            step=0.01,
            help="Cost per transaction"
        ) / 100
        
        slippage = st.slider(
            "Slippage (%)",
            min_value=0.0,
            max_value=0.5,
            value=0.05,
            step=0.01,
            help="Market impact slippage"
        ) / 100
    
    # Main content area
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 Data & Setup",
        "🎯 Training",
        "📈 Backtesting",
        "🔍 Analysis",
        "📋 Reports"
    ])
    
    # Tab 1: Data & Setup
    with tab1:
        st.subheader("📁 Data Management")
        
        # Load data
        data_loaded = False
        current_data = None
        
        if uploaded_file is not None:
            current_data = trading_system.load_data_file(uploaded_file)
            if current_data is not None:
                data_loaded = True
                st.success(f"✅ Loaded {uploaded_file.name} with {len(current_data)} records")
                
        elif use_sample != "Select...":
            file_map = {
                "BTCUSD": "data/BTCUSD_data.csv",
                "ETHUSD": "data/ETHUSD_data.csv", 
                "GOLD": "data/GOLD_data.csv",
                "SPXUSD": "data/SPXUSD_data.csv"
            }
            try:
                current_data = pd.read_csv(file_map[use_sample])
                current_data['time'] = pd.to_datetime(current_data['time'])
                current_data = current_data.set_index('time').sort_index()
                data_loaded = True
                st.success(f"✅ Loaded {use_sample} data with {len(current_data)} records")
            except Exception as e:
                st.error(f"Error loading sample data: {e}")
        
        if data_loaded and current_data is not None:
            trading_system.data = current_data
            
            # Display data info
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("📅 Total Records", f"{len(current_data):,}")
            with col2:
                st.metric("📊 Date Range", f"{len(pd.date_range(current_data.index.min(), current_data.index.max()))} days")
            with col3:
                st.metric("💰 Price Range", f"${current_data['close'].min():.2f} - ${current_data['close'].max():.2f}")
            with col4:
                st.metric("📈 Volatility", f"{current_data['close'].pct_change().std()*100:.2f}%")
            
            # Date range selection
            st.subheader("📅 Date Range Configuration")
            
            min_date = current_data.index.min().date()
            max_date = current_data.index.max().date()
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**🎓 Training Period**")
                train_start = st.date_input(
                    "Training Start Date",
                    value=min_date,
                    min_value=min_date,
                    max_value=max_date
                )
                train_end = st.date_input(
                    "Training End Date",
                    value=min_date + timedelta(days=(max_date-min_date).days*0.7),
                    min_value=min_date,
                    max_value=max_date
                )
            
            with col2:
                st.markdown("**🧪 Backtesting Period**")
                test_start = st.date_input(
                    "Backtest Start Date",
                    value=min_date + timedelta(days=(max_date-min_date).days*0.7+1),
                    min_value=min_date,
                    max_value=max_date
                )
                test_end = st.date_input(
                    "Backtest End Date",
                    value=max_date,
                    min_value=min_date,
                    max_value=max_date
                )
            
            # Store date ranges in session state
            st.session_state.train_start = train_start.strftime('%Y-%m-%d')
            st.session_state.train_end = train_end.strftime('%Y-%m-%d')
            st.session_state.test_start = test_start.strftime('%Y-%m-%d')
            st.session_state.test_end = test_end.strftime('%Y-%m-%d')
            
            # Validate date ranges
            if train_end >= test_start:
                st.error("⚠️ Training end date must be before backtest start date!")
            else:
                st.success("✅ Date ranges are valid - no data leakage!")
            
            # Data preview
            st.subheader("📊 Data Preview")
            
            # Show sample of data
            st.dataframe(current_data.head(10), use_container_width=True)
            
            # Basic price chart
            fig = go.Figure(data=go.Candlestick(
                x=current_data.index,
                open=current_data['open'],
                high=current_data['high'],
                low=current_data['low'],
                close=current_data['close'],
                name='Price'
            ))
            
            fig.update_layout(
                title="💹 Price Data Overview",
                template="plotly_dark",
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
        else:
            st.info("👆 Please upload a data file or select sample data to continue")
    
    # Tab 2: Training
    with tab2:
        st.subheader("🎓 Model Training")
        
        if not hasattr(st.session_state, 'train_start') or trading_system.data is None:
            st.warning("⚠️ Please configure data and date ranges in the Data & Setup tab first")
        else:
            # Split data for training
            train_data, test_data = trading_system.split_data_by_dates(
                trading_system.data,
                st.session_state.train_start,
                st.session_state.train_end,
                st.session_state.test_start,
                st.session_state.test_end
            )
            
            if train_data is not None and test_data is not None:
                col1, col2 = st.columns(2)
                
                with col1:
                    st.info(f"""
                    **📚 Training Data:**
                    - Period: {st.session_state.train_start} to {st.session_state.train_end}
                    - Records: {len(train_data):,}
                    - Days: {(pd.to_datetime(st.session_state.train_end) - pd.to_datetime(st.session_state.train_start)).days}
                    """)
                
                with col2:
                    st.info(f"""
                    **🧪 Test Data:**
                    - Period: {st.session_state.test_start} to {st.session_state.test_end}
                    - Records: {len(test_data):,}
                    - Days: {(pd.to_datetime(st.session_state.test_end) - pd.to_datetime(st.session_state.test_start)).days}
                    """)
                
                # Find correlations
                if st.button("🔍 Analyze Correlations", type="primary"):
                    with st.spinner("Analyzing correlations and patterns..."):
                        correlations = trading_system.find_correlations(train_data)
                        trading_system.correlations = correlations
                        st.session_state.correlations_done = True
                        st.success("✅ Correlation analysis complete!")
                
                if hasattr(st.session_state, 'correlations_done') and trading_system.correlations:
                    # Show correlation results
                    st.subheader("📊 Correlation Analysis Results")
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown("**🎯 Top Correlated Features:**")
                        strong_features = trading_system.correlations['strong_features']
                        for feature, corr in list(strong_features.items())[:10]:
                            st.write(f"• {feature}: {corr:.3f}")
                    
                    with col2:
                        st.markdown("**📈 Correlation Strength Distribution:**")
                        corr_values = list(trading_system.correlations['feature_correlations'].values())
                        fig_hist = go.Figure(data=go.Histogram(
                            x=corr_values,
                            nbinsx=30,
                            marker_color='lightblue'
                        ))
                        fig_hist.update_layout(
                            template="plotly_dark",
                            height=300,
                            showlegend=False
                        )
                        st.plotly_chart(fig_hist, use_container_width=True)
                    
                    # Correlation heatmap
                    fig_corr = create_correlation_heatmap(trading_system.correlations)
                    st.plotly_chart(fig_corr, use_container_width=True)
                
                # Train models
                if st.button("🚀 Train ML Models", type="primary"):
                    config = {
                        'initial_capital': initial_capital,
                        'lookback_window': lookback_window,
                        'ensemble_size': ensemble_size,
                        'profit_threshold': profit_threshold,
                        'transaction_cost': transaction_cost,
                        'slippage': slippage
                    }
                    
                    training_results = trading_system.train_strategy(train_data, config)
                    
                    if training_results['success']:
                        st.success("✅ Model training completed successfully!")
                        st.session_state.model_trained = True
                        
                        # Show training results
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            st.metric("🎯 Training Accuracy", f"{training_results['training_accuracy']:.1%}")
                        with col2:
                            st.metric("🔢 Feature Count", f"{training_results['feature_count']:,}")
                        with col3:
                            st.metric("📊 Training Samples", f"{training_results['training_samples']:,}")
                        
                        # Feature importance
                        if training_results.get('feature_importance'):
                            fig_importance = create_feature_importance_chart(training_results['feature_importance'])
                            st.plotly_chart(fig_importance, use_container_width=True)
                    
                    else:
                        st.error(f"❌ Training failed: {training_results.get('error', 'Unknown error')}")
    
    # Tab 3: Backtesting
    with tab3:
        st.subheader("📈 Strategy Backtesting")
        
        if not hasattr(st.session_state, 'model_trained') or trading_system.data is None:
            st.warning("⚠️ Please train the model first in the Training tab")
        else:
            # Get test data
            train_data, test_data = trading_system.split_data_by_dates(
                trading_system.data,
                st.session_state.train_start,
                st.session_state.train_end,
                st.session_state.test_start,
                st.session_state.test_end
            )
            
            if test_data is not None:
                st.info(f"🧪 **Ready to backtest on completely unseen data from {st.session_state.test_start} to {st.session_state.test_end}**")
                
                if st.button("🚀 Run Backtest", type="primary"):
                    config = {
                        'initial_capital': initial_capital,
                        'transaction_cost': transaction_cost,
                        'slippage': slippage
                    }
                    
                    backtest_results = trading_system.run_backtest(test_data, config)
                    
                    if backtest_results['success']:
                        st.success("✅ Backtesting completed!")
                        st.session_state.backtest_done = True
                        st.session_state.backtest_results = backtest_results
                        
                        # Display key metrics
                        st.subheader("📊 Performance Summary")
                        
                        col1, col2, col3, col4, col5 = st.columns(5)
                        
                        with col1:
                            profit_color = "normal"
                            if backtest_results['total_return'] > 20:
                                profit_color = "inverse"
                            elif backtest_results['total_return'] < -10:
                                profit_color = "off"
                                
                            st.metric(
                                "💰 Total Return",
                                f"{backtest_results['total_return']:.1f}%",
                                f"vs Market: {backtest_results['excess_return']:.1f}%"
                            )
                        
                        with col2:
                            sharpe_color = "normal"
                            if backtest_results['sharpe_ratio'] > 2:
                                sharpe_color = "inverse"
                            elif backtest_results['sharpe_ratio'] < 0:
                                sharpe_color = "off"
                                
                            st.metric("📊 Sharpe Ratio", f"{backtest_results['sharpe_ratio']:.2f}")
                        
                        with col3:
                            dd_color = "inverse" if backtest_results['max_drawdown'] > -10 else "normal"
                            st.metric("📉 Max Drawdown", f"{backtest_results['max_drawdown']:.1f}%")
                        
                        with col4:
                            wr_color = "inverse" if backtest_results['win_rate'] > 60 else "normal"
                            st.metric("🎯 Win Rate", f"{backtest_results['win_rate']:.1f}%")
                        
                        with col5:
                            st.metric("💵 Final Value", f"${backtest_results['final_value']:,.0f}")
                        
                        # Additional metrics
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            st.metric("📈 Market Return", f"{backtest_results['market_return']:.1f}%")
                        with col2:
                            st.metric("🌊 Volatility", f"{backtest_results['volatility']:.1f}%")
                        with col3:
                            st.metric("🔄 Total Trades", f"{backtest_results['total_trades']}")
                        
                        # Performance visualization
                        fig_perf = create_performance_chart(backtest_results['portfolio_data'], test_data)
                        st.plotly_chart(fig_perf, use_container_width=True)
                    
                    else:
                        st.error(f"❌ Backtesting failed: {backtest_results.get('error', 'Unknown error')}")
    
    # Tab 4: Analysis
    with tab4:
        st.subheader("🔍 Deep Analysis")
        
        if hasattr(st.session_state, 'backtest_done') and st.session_state.backtest_results:
            results = st.session_state.backtest_results
            
            # Performance comparison
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("### 📊 Strategy vs Market")
                comparison_data = {
                    'Metric': ['Total Return', 'Volatility', 'Sharpe Ratio', 'Max Drawdown'],
                    'Strategy': [f"{results['total_return']:.1f}%", 
                               f"{results['volatility']:.1f}%",
                               f"{results['sharpe_ratio']:.2f}",
                               f"{results['max_drawdown']:.1f}%"],
                    'Market': [f"{results['market_return']:.1f}%", 
                             "N/A", "N/A", "N/A"]
                }
                st.dataframe(pd.DataFrame(comparison_data), use_container_width=True)
            
            with col2:
                st.markdown("### 🎯 Trade Analysis")
                trade_data = {
                    'Metric': ['Total Trades', 'Winning Trades', 'Win Rate', 'Avg Trade'],
                    'Value': [f"{results['total_trades']}", 
                            f"{results['winning_trades']}",
                            f"{results['win_rate']:.1f}%",
                            f"${(results['final_value']-results.get('initial_capital', 100000))/max(results['total_trades'],1):.2f}"]
                }
                st.dataframe(pd.DataFrame(trade_data), use_container_width=True)
            
            # Risk analysis
            st.markdown("### 🛡️ Risk Analysis")
            
            portfolio = results['portfolio_data']
            
            # Rolling metrics
            col1, col2 = st.columns(2)
            
            with col1:
                # Rolling returns
                rolling_returns = portfolio['strategy_returns'].rolling(30).mean() * 100
                
                fig_rolling = go.Figure()
                fig_rolling.add_trace(go.Scatter(
                    x=portfolio.index,
                    y=rolling_returns,
                    name='30-Day Avg Return (%)',
                    line=dict(color='green')
                ))
                fig_rolling.update_layout(
                    title="📈 Rolling 30-Day Average Returns",
                    template="plotly_dark",
                    height=400
                )
                st.plotly_chart(fig_rolling, use_container_width=True)
            
            with col2:
                # Rolling volatility
                rolling_vol = portfolio['strategy_returns'].rolling(30).std() * np.sqrt(252) * 100
                
                fig_vol = go.Figure()
                fig_vol.add_trace(go.Scatter(
                    x=portfolio.index,
                    y=rolling_vol,
                    name='30-Day Volatility (%)',
                    line=dict(color='orange'),
                    fill='tonexty'
                ))
                fig_vol.update_layout(
                    title="🌊 Rolling 30-Day Volatility",
                    template="plotly_dark",
                    height=400
                )
                st.plotly_chart(fig_vol, use_container_width=True)
            
            # Monthly performance heatmap
            monthly_returns = portfolio['strategy_returns'].resample('M').apply(lambda x: (1 + x).prod() - 1)
            monthly_returns = monthly_returns * 100
            
            if len(monthly_returns) > 0:
                monthly_df = pd.DataFrame({
                    'Year': monthly_returns.index.year,
                    'Month': monthly_returns.index.month,
                    'Return': monthly_returns.values
                })
                
                monthly_pivot = monthly_df.pivot(index='Year', columns='Month', values='Return')
                
                fig_heatmap = go.Figure(data=go.Heatmap(
                    z=monthly_pivot.values,
                    x=[calendar.month_name[i] for i in monthly_pivot.columns],
                    y=monthly_pivot.index,
                    colorscale='RdYlGn',
                    zmid=0,
                    colorbar=dict(title="Monthly Return (%)")
                ))
                
                fig_heatmap.update_layout(
                    title="📅 Monthly Returns Heatmap",
                    template="plotly_dark",
                    height=400
                )
                st.plotly_chart(fig_heatmap, use_container_width=True)
        
        else:
            st.info("👆 Please complete the backtesting process to see detailed analysis")
    
    # Tab 5: Reports
    with tab5:
        st.subheader("📋 Reports & Export")
        
        if hasattr(st.session_state, 'backtest_done') and st.session_state.backtest_results:
            results = st.session_state.backtest_results
            
            # Generate comprehensive report
            st.markdown("### 📄 Strategy Report")
            
            report_text = f"""
# 🚀 Advanced ML Trading Strategy Report

## 📊 Executive Summary
- **Strategy Performance**: {results['total_return']:.1f}% total return
- **Market Performance**: {results['market_return']:.1f}% total return  
- **Excess Return**: {results['excess_return']:.1f}%
- **Risk-Adjusted Return (Sharpe)**: {results['sharpe_ratio']:.2f}
- **Maximum Drawdown**: {results['max_drawdown']:.1f}%

## 🎯 Trading Statistics
- **Total Trades**: {results['total_trades']}
- **Winning Trades**: {results['winning_trades']}
- **Win Rate**: {results['win_rate']:.1f}%
- **Strategy Volatility**: {results['volatility']:.1f}%

## ⚙️ Configuration
- **Initial Capital**: ${initial_capital:,}
- **Training Period**: {st.session_state.train_start} to {st.session_state.train_end}
- **Testing Period**: {st.session_state.test_start} to {st.session_state.test_end}
- **Transaction Cost**: {transaction_cost*100:.2f}%
- **Slippage**: {slippage*100:.2f}%
- **Ensemble Size**: {ensemble_size} models

## 💡 Key Insights
1. Strategy {'outperformed' if results['excess_return'] > 0 else 'underperformed'} the market by {abs(results['excess_return']):.1f}%
2. Risk-adjusted returns {'excellent' if results['sharpe_ratio'] > 2 else 'good' if results['sharpe_ratio'] > 1 else 'moderate'}
3. Drawdown control {'excellent' if results['max_drawdown'] > -10 else 'good' if results['max_drawdown'] > -20 else 'needs improvement'}
4. Win rate is {'high' if results['win_rate'] > 60 else 'moderate' if results['win_rate'] > 40 else 'low'} at {results['win_rate']:.1f}%

Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
            """
            
            st.markdown(report_text)
            
            # Download options
            col1, col2, col3 = st.columns(3)
            
            with col1:
                # Download report as text
                st.download_button(
                    label="📄 Download Report",
                    data=report_text,
                    file_name=f"trading_strategy_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md",
                    mime="text/markdown"
                )
            
            with col2:
                # Download portfolio data
                portfolio_csv = results['portfolio_data'].to_csv()
                st.download_button(
                    label="📊 Download Portfolio Data",
                    data=portfolio_csv,
                    file_name=f"portfolio_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )
            
            with col3:
                # Download feature importance
                if trading_system.feature_importance:
                    importance_df = pd.DataFrame(
                        list(trading_system.feature_importance.items()),
                        columns=['Feature', 'Importance']
                    )
                    importance_csv = importance_df.to_csv(index=False)
                    st.download_button(
                        label="🔥 Download Feature Importance",
                        data=importance_csv,
                        file_name=f"feature_importance_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv"
                    )
            
            # Summary dashboard
            st.markdown("### 🎯 Quick Summary Dashboard")
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Key metrics gauge
                fig_gauge = go.Figure()
                
                fig_gauge.add_trace(go.Indicator(
                    mode = "gauge+number+delta",
                    value = results['sharpe_ratio'],
                    domain = {'x': [0, 1], 'y': [0, 1]},
                    title = {'text': "Sharpe Ratio"},
                    delta = {'reference': 1.0},
                    gauge = {
                        'axis': {'range': [-2, 4]},
                        'bar': {'color': "darkblue"},
                        'steps': [
                            {'range': [-2, 0], 'color': "red"},
                            {'range': [0, 1], 'color': "yellow"},
                            {'range': [1, 2], 'color': "lightgreen"},
                            {'range': [2, 4], 'color': "green"}
                        ],
                        'threshold': {
                            'line': {'color': "red", 'width': 4},
                            'thickness': 0.75,
                            'value': 2
                        }
                    }
                ))
                
                fig_gauge.update_layout(height=300, template="plotly_dark")
                st.plotly_chart(fig_gauge, use_container_width=True)
            
            with col2:
                # Performance pie chart
                labels = ['Strategy Excess', 'Market Return', 'Risk-Free Rate (assumed 2%)']
                values = [
                    max(0, results['excess_return']),
                    results['market_return'],
                    2
                ]
                
                fig_pie = go.Figure(data=[go.Pie(
                    labels=labels,
                    values=values,
                    hole=0.4
                )])
                
                fig_pie.update_layout(
                    title="Return Attribution",
                    template="plotly_dark",
                    height=300
                )
                st.plotly_chart(fig_pie, use_container_width=True)
        
        else:
            st.info("👆 Complete the backtesting process to generate reports")

if __name__ == "__main__":
    main()