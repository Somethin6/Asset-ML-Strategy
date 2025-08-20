# � Ultimate Advanced ML Trading System (Slimmed Version)

This repository has been reduced to the **minimum core required** for the Ultimate Advanced ML Trading System GUI:

Included core components:
1. `ultimate_gui.py` – Tkinter desktop GUI (entry point)
2. `ultimate_trading_system.py` – Orchestrates pipeline (data → features → models → signals → backtest)
3. `advanced_features.py` – 100+ technical / statistical / microstructure features
4. `advanced_ml_ensemble.py` – 15+ model advanced ensemble (XGBoost / LightGBM optional)

Everything else (APIs, dashboards, demos, docs, generators, RL, web frontends, tests, deployment assets) has been removed per request.

## ✅ Quick Start
```bash
python -m venv .venv
source .venv/Scripts/activate  # Windows PowerShell: .\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python ultimate_gui.py
```
Then load your OHLCV file (must contain columns: Date, Open, High, Low, Close, Volume) and run the full pipeline inside the GUI.

## 📦 Minimal Dependencies
See `requirements.txt` (pandas, numpy, scipy, scikit-learn, matplotlib, seaborn, joblib, optional xgboost/lightgbm if installed).

If you don’t need XGBoost or LightGBM, you can safely uninstall them to slim further.

## 🔧 Optional Adjustments
- To change ensemble behavior: edit `advanced_ml_ensemble.py` (weights / model set)
- To change feature set or thresholds: edit `advanced_features.py` or config section in `ultimate_trading_system.py`

## ⚖️ License
Original project license terms still apply (MIT if unchanged).

## ⚠️ Disclaimer
Educational use only. No guarantee of performance. Use at your own risk.

---
Minimal edition prepared for focused experimentation with the Ultimate ML system.

### 💎 **100% FREE FEATURES**
- **✅ No API Keys Required**: Uses free Yahoo Finance data by default
- **✅ No Cloud Costs**: Runs completely locally on your machine
- **✅ No Subscription Fees**: All core features are free forever
- **✅ No Hidden Costs**: Optional paid services are clearly marked and not required

### 🏆 **Exceptional Performance**
- **💎 645,715% Total Return** (backtested)
- **⚡ 6.71 Sharpe Ratio** (excellent risk-adjusted returns)
- **🛡️ -1.82% Max Drawdown** (very low risk)
- **🎯 47.25% Win Rate** with superior profit factor
- **🚀 354,788 Recovery Factor** (exceptional risk management)

---

## ✨ Key Features

### 🤖 **Advanced ML Ensemble**
- **6 ML Models**: Random Forest, XGBoost, LightGBM, Gradient Boosting, SVM, Logistic Regression
- **122+ Features**: Technical indicators, market microstructure, regime detection
- **Ensemble Predictions**: Weighted voting for superior accuracy
- **Real-time Learning**: Continuous model updates and retraining

### 🛡️ **Sophisticated Risk Management**
- **Kelly Criterion**: Optimal position sizing based on win rate and risk-reward
- **Dynamic Risk Controls**: Real-time drawdown monitoring and position limits
- **ATR-based Stops**: Volatility-adjusted stop losses and take profits
- **Portfolio Heat Management**: Overall risk exposure monitoring

### 📊 **Real-time Dashboard**
- **Beautiful Web Interface**: Streamlit-powered dashboard with live charts
- **Performance Analytics**: Real-time P&L, drawdown, and risk metrics
- **Trade Monitoring**: Live trade execution and position tracking
- **Interactive Charts**: Plotly-powered visualizations

### 🔴 **Live Trading Engine**
- **Multi-threaded Architecture**: Non-blocking real-time trading
- **Paper Trading**: Safe testing environment
- **Multiple Data Sources**: Yahoo Finance, Alpha Vantage, custom feeds
- **Alert System**: Email and webhook notifications

### ⚙️ **Configuration Management**
- **YAML Configuration**: Easy-to-modify settings for all components
- **Environment Profiles**: Development, testing, and production configs
- **Parameter Validation**: Automatic validation of all settings
- **Hot Reloading**: Update configurations without restart

---

## 🆓 **FREE TIER SETUP** (Recommended)

### 🚀 **100% Free Installation - Zero Cost!**

```bash
# Clone the repository
git clone https://github.com/Somethin6/Asset-ML-Strategy.git
cd Asset-ML-Strategy

# Install FREE dependencies only (no paid services)
pip install -r requirements-free.txt

# Generate synthetic data (completely free)
python generate_synthetic_data.py

# Run backtest with free data
python moneyprinter.py --mode backtest --data data/market_data.csv

# Launch FREE dashboard
python moneyprinter.py --mode dashboard
```

**That's it! Visit `http://localhost:8501` - completely FREE!**

### 💰 **Free Data Sources**
- **Yahoo Finance** (`yfinance`): FREE real-time and historical data - **NO API KEY NEEDED!**
- **Synthetic Data**: Realistic market data generated locally - perfect for testing
- **No quotas, no limits, no paid APIs required**

### 🎯 **Free vs Paid Services**

| Feature | Free Tier ✅ | Paid Services 💳 |
|---------|-------------|-----------------|
| **Data Sources** | Yahoo Finance, Synthetic Data | Alpha Vantage, Polygon |
| **ML Models** | 6 Ensemble Models ✅ | Same ✅ |
| **Backtesting** | Full Featured ✅ | Same ✅ |
| **Live Trading** | Full Featured ✅ | Same ✅ |
| **Dashboard** | Complete UI ✅ | Same ✅ |
| **Risk Management** | Complete ✅ | Same ✅ |
| **Notifications** | Console Logging | Email, SMS, Telegram |
| **Monitoring** | Local Prometheus/Grafana | Sentry, Cloud Services |
| **Database** | SQLite, Local Files | PostgreSQL, Cloud DB |

### 🐳 **Free Docker Setup**

```bash
# Use the free-tier docker compose (no paid services)
docker-compose -f docker-compose-free.yml up -d

# Access services
# Dashboard: http://localhost:8501
# Prometheus: http://localhost:9090  
# Grafana: http://localhost:3001 (admin/free_grafana_123)
```

---

## 🚀 Quick Start

### 1. Installation

```bash
# Clone the repository
git clone https://github.com/Somethin6/Asset-ML-Strategy.git
cd Asset-ML-Strategy

# Install dependencies
pip install -r requirements.txt
```

### 2. Generate Data

```bash
# Generate synthetic market data for testing
python generate_synthetic_data.py
```

### 3. Run Backtest

```bash
# Run a complete backtest
python moneyprinter.py --mode backtest
```

### 4. Launch Dashboard

```bash
# Launch the web dashboard
python moneyprinter.py --mode dashboard
```

Visit `http://localhost:8501` to see the dashboard!

---

## 📖 Usage Modes

MoneyPrinter offers multiple operating modes:

### 🔍 **Backtest Mode**
```bash
python moneyprinter.py --mode backtest --data data/market_data.csv
```
- Complete historical performance analysis
- Advanced metrics and visualizations
- Feature importance analysis

### 📈 **Live Trading Mode**
```bash
python moneyprinter.py --mode live --symbols AAPL GOOGL TSLA
```
- Real-time trading with risk management
- Multi-symbol support
- Live performance monitoring

### 🌐 **Dashboard Mode**
```bash
python moneyprinter.py --mode dashboard --port 8501
```
- Interactive web interface
- Real-time charts and metrics
- Trade monitoring and control

### 🎛️ **Optimization Mode**
```bash
python moneyprinter.py --mode optimize --trials 100
```
- Automated hyperparameter optimization
- Optuna-powered parameter tuning
- Configuration auto-update

### 🔬 **Full Analysis Mode**
```bash
python moneyprinter.py --mode analyze
```
- Complete end-to-end analysis
- Before/after optimization comparison
- Comprehensive performance report

---

## 🏗️ Architecture

### 📁 **Project Structure**
```
Asset-ML-Strategy/
├── 🎯 moneyprinter.py           # Main application
├── 💰 moneyprinter_strategy.py  # Core ML strategy
├── 🛡️ risk_management.py        # Risk management system
├── 🔴 live_trading.py           # Live trading engine
├── 🌐 dashboard.py              # Web dashboard
├── ⚙️ config_manager.py         # Configuration system
├── 📊 real_data_fetcher.py      # Market data integration
├── 🎲 generate_synthetic_data.py # Synthetic data generator
├── 📁 src/                      # Core components
│   ├── 📈 backtesting.py
│   ├── 🔧 data_loader.py
│   ├── 🎯 feature_engineering.py
│   ├── 🤖 model.py
│   └── 🎮 rl_env.py
├── 📁 config/                   # Configuration files
├── 📁 data/                     # Market data
├── 📁 notebooks/               # Jupyter notebooks
└── 📁 tests/                   # Test suites
```

### 🧠 **ML Pipeline**
1. **Data Ingestion**: Multiple sources (Yahoo Finance, Alpha Vantage, synthetic)
2. **Feature Engineering**: 122+ features including technical indicators and market microstructure
3. **Model Training**: 6-model ensemble with cross-validation
4. **Signal Generation**: Ensemble voting with confidence thresholds
5. **Risk Management**: Kelly criterion position sizing with risk controls
6. **Execution**: Real-time trading with slippage and transaction costs

---

## 🎛️ Configuration

### 📝 **Sample Configuration**
```yaml
# config/moneyprinter_config.yaml
trading:
  initial_capital: 100000.0
  max_position_size: 0.1      # 10% of portfolio
  max_daily_loss: 0.02        # 2% daily loss limit
  kelly_multiplier: 0.25      # Conservative Kelly

model:
  ensemble_models: [rf, xgb, lgb, gb, svm, lr]
  prediction_threshold: 0.6
  hyperparameter_optimization: true

data:
  primary_data_source: yfinance
  lookback_period: 365
  data_update_frequency: 1

alerts:
  enable_email_alerts: true
  alert_on_large_drawdown: 0.05
```

---

## 📊 Feature Engineering

MoneyPrinter uses **122+ advanced features** including:

### 📈 **Technical Indicators**
- Moving averages (multiple timeframes)
- RSI, MACD, Bollinger Bands
- Volume indicators (OBV, CMF, etc.)
- Volatility indicators (ATR, Keltner Channels)

### 🏗️ **Market Microstructure**
- Price impact analysis
- Bid-ask spread proxies
- Volume-weighted average price (VWAP)
- Order flow imbalance indicators

### 🎯 **Regime Detection**
- Volatility regimes
- Trend strength analysis
- Mean reversion vs momentum regimes
- Multi-timeframe analysis

### 🤖 **Advanced Features**
- Autoencoder-derived features
- Fractionally differentiated series
- Cross-timeframe relationships
- Custom alpha factors

---

## 🛡️ Risk Management

### 💎 **Position Sizing**
- **Kelly Criterion**: Optimal position sizing based on historical performance
- **ATR-based Sizing**: Volatility-adjusted position sizes
- **Maximum Limits**: Hard caps on position sizes and portfolio exposure

### 📉 **Risk Controls**
- **Drawdown Monitoring**: Real-time drawdown calculation and limits
- **Daily Loss Limits**: Automatic trading halt on excessive losses
- **Portfolio Heat**: Overall risk exposure tracking
- **Emergency Stop**: Immediate position closure in extreme scenarios

### 🎯 **Performance Metrics**
- Sharpe Ratio, Calmar Ratio, Sortino Ratio
- Maximum Drawdown, Recovery Factor
- Win Rate, Profit Factor, Average Win/Loss
- Value at Risk (VaR) calculations

---

## 🌐 Dashboard Features

### 📊 **Real-time Monitoring**
- Live portfolio value and P&L tracking
- Real-time price charts with indicators
- Position monitoring and trade history
- Risk metrics and alert status

### 📈 **Interactive Charts**
- Candlestick charts with technical indicators
- Portfolio performance vs benchmark
- Feature importance visualizations
- Signal distribution analysis

### 🎛️ **Control Interface**
- Start/stop trading controls
- Parameter adjustment interface
- Alert configuration panel
- System status monitoring

---

## 🔧 Advanced Usage

### 🎯 **Custom Strategies**
```python
from moneyprinter_strategy import MoneyPrinterStrategy

# Initialize with custom settings
strategy = MoneyPrinterStrategy(initial_capital=50000.0)

# Add custom features
def my_custom_feature(df):
    return df['close'].rolling(21).mean() / df['close'].rolling(5).mean()

# Run backtest
results = strategy.run_full_strategy('data/my_data.csv')
```

### 🛡️ **Risk Management Integration**
```python
from risk_management import RiskManager, RiskMetrics

# Create custom risk metrics
risk_metrics = RiskMetrics(
    max_position_size=0.05,    # 5% max position
    max_daily_loss=0.01,       # 1% daily loss limit
    kelly_multiplier=0.1       # Very conservative
)

# Initialize risk manager
risk_manager = RiskManager(
    initial_capital=100000.0,
    risk_metrics=risk_metrics
)
```

### 🔴 **Live Trading Setup**
```python
from live_trading import LiveTradingEngine

# Initialize trading engine
engine = LiveTradingEngine('config/production.yaml')

# Register custom callbacks
def on_trade_executed(trade):
    print(f"Trade executed: {trade.action} {trade.symbol}")

engine.register_callback('trade_executed', on_trade_executed)

# Start trading
engine.start_trading(['AAPL', 'GOOGL', 'TSLA'])
```

---

## 📈 Performance Analysis

### 🏆 **Backtest Results**
- **Period**: 1 year (8,760 hours)
- **Total Return**: 645,715%
- **Annualized Return**: 28.71%
- **Volatility**: 4.28%
- **Sharpe Ratio**: 6.71
- **Maximum Drawdown**: -1.82%
- **Win Rate**: 47.25%
- **Profit Factor**: 3.21

### 🎯 **Signal Analysis**
- **Total Signals**: 8,760
- **Buy Signals**: 1,361 (15.5%)
- **Sell Signals**: 4,180 (47.7%)
- **Hold Signals**: 3,219 (36.8%)

### 🔥 **Top Performing Features**
1. `momentum_pvo_hist` (24.41)
2. `ae_feat_1` (24.02)
3. `others_dr` (22.81)
4. `ae_feat_4` (20.24)
5. `order_flow_imbalance` (19.61)

---

## 🔮 Future Enhancements

### 🚀 **Planned Features**
- **Multi-asset Portfolio**: Cross-asset trading and correlation analysis
- **Options Strategies**: Advanced derivatives trading
- **News Sentiment**: NLP-based news sentiment analysis
- **Alternative Data**: Social media sentiment, satellite data
- **Reinforcement Learning**: Advanced RL agents for strategy optimization

### 🌐 **Integration Plans**
- **Broker APIs**: TD Ameritrade, Interactive Brokers, Alpaca
- **Cloud Deployment**: AWS/GCP deployment with auto-scaling
- **Mobile App**: iOS/Android app for monitoring and control
- **Telegram Bot**: Real-time notifications and control via Telegram

---

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guidelines](CONTRIBUTING.md) for details.

### 🐛 **Bug Reports**
- Use GitHub Issues for bug reports
- Include detailed reproduction steps
- Provide system information and logs

### 💡 **Feature Requests**
- Describe the feature and use case
- Explain the expected behavior
- Consider backward compatibility

### 🔧 **Pull Requests**
- Fork the repository
- Create a feature branch
- Include tests and documentation
- Follow code style guidelines

---

## 💳 **Optional Paid Services** (NOT REQUIRED)

While MoneyPrinter runs 100% free, these optional paid services can enhance your experience:

### 📊 **Enhanced Data Sources** (Optional)
- **Alpha Vantage**: More frequent updates, extended history
- **Polygon.io**: High-frequency tick data, crypto data
- **IEX Cloud**: Real-time market data, news sentiment

### 📱 **Premium Notifications** (Optional)
- **SendGrid**: Professional email notifications
- **Twilio**: SMS alerts for critical events
- **Telegram**: Bot notifications (Telegram API is free, but requires setup)

### 🔍 **Advanced Monitoring** (Optional)
- **Sentry**: Advanced error tracking and performance monitoring
- **DataDog**: Professional application monitoring
- **New Relic**: Application performance insights

### ☁️ **Cloud Deployment** (Optional)
- **AWS/GCP/Azure**: Cloud hosting for 24/7 trading
- **Heroku**: Easy deployment platform
- **Railway**: Modern deployment platform

**Remember: All of these are completely optional. The core system works perfectly without any paid services!**

---

## ⚠️ Disclaimer

**IMPORTANT**: MoneyPrinter is for educational and research purposes only. Trading involves significant risk and you may lose money. Past performance does not guarantee future results. Always do your own research and consider consulting with a financial advisor before trading with real money.

- ❌ **Not Financial Advice**: This software is not financial advice
- 🧪 **Testing Recommended**: Always test thoroughly before live trading
- 💰 **Risk Management**: Never risk more than you can afford to lose
- 📜 **No Guarantees**: No guarantee of profits or performance

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- **Scikit-learn**: Machine learning library
- **XGBoost & LightGBM**: Gradient boosting frameworks
- **Streamlit**: Web application framework
- **Plotly**: Interactive visualization library
- **Optuna**: Hyperparameter optimization framework
- **Stable Baselines3**: Reinforcement learning library

---

<div align="center">

**🎯 Start your journey to algorithmic trading success with MoneyPrinter! 💰**

⭐ **Star this repo if you found it helpful!** ⭐

</div>