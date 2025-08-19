#!/usr/bin/env python3
"""
Enhanced ML Test - Demonstrate improved machine learning capabilities
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend

# Advanced ML libraries
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import xgboost as xgb
import lightgbm as lgb

# Technical analysis
import ta

# Real-time data (demo)
import yfinance as yf

# Hyperparameter optimization
import optuna
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

def generate_enhanced_sample_data(n_samples=500):
    """Generate enhanced sample financial data with realistic patterns"""
    print("📊 Generating enhanced sample data...")
    
    dates = pd.date_range('2020-01-01', periods=n_samples, freq='D')
    
    # Generate more realistic price data with trends and volatility
    np.random.seed(42)
    
    # Base trend
    trend = np.linspace(100, 150, n_samples) + np.random.normal(0, 5, n_samples)
    
    # Add cyclical patterns
    cyclical = 10 * np.sin(np.arange(n_samples) * 2 * np.pi / 252)  # Annual cycle
    weekly = 2 * np.sin(np.arange(n_samples) * 2 * np.pi / 5)      # Weekly cycle
    
    # Add volatility clustering
    volatility = np.abs(np.random.normal(0, 1, n_samples))
    volatility = pd.Series(volatility).rolling(window=10, min_periods=1).mean().values
    
    # Generate OHLC data
    close_prices = trend + cyclical + weekly + np.random.normal(0, volatility * 2)
    close_prices = np.maximum(close_prices, 50)  # Minimum price
    
    data = []
    for i in range(n_samples):
        base_price = close_prices[i]
        high = base_price + np.random.exponential(1)
        low = base_price - np.random.exponential(1)
        open_price = low + np.random.random() * (high - low)
        
        # Ensure OHLC relationships
        high = max(high, open_price, base_price)
        low = min(low, open_price, base_price)
        
        volume = int(np.random.lognormal(12, 0.5))  # Realistic volume distribution
        
        data.append({
            'Date': dates[i],
            'Open': round(open_price, 2),
            'High': round(high, 2),
            'Low': round(low, 2),
            'Close': round(base_price, 2),
            'Adj Close': round(base_price, 2),
            'Volume': volume
        })
    
    df = pd.DataFrame(data)
    print(f"✓ Generated {len(df)} samples with realistic patterns")
    print(f"  Price range: ${df['Close'].min():.2f} to ${df['Close'].max():.2f}")
    print(f"  Average volume: {df['Volume'].mean():.0f}")
    
    return df

def add_advanced_technical_indicators(df):
    """Add comprehensive technical indicators using the 'ta' library"""
    print("🔧 Adding advanced technical indicators...")
    
    # Price-based indicators
    df['RSI'] = ta.momentum.RSIIndicator(df['Close']).rsi()
    df['MACD'] = ta.trend.MACD(df['Close']).macd()
    df['MACD_Signal'] = ta.trend.MACD(df['Close']).macd_signal()
    
    # Bollinger Bands
    bb = ta.volatility.BollingerBands(df['Close'])
    df['BB_Upper'] = bb.bollinger_hband()
    df['BB_Lower'] = bb.bollinger_lband()
    df['BB_Width'] = df['BB_Upper'] - df['BB_Lower']
    
    # Moving averages
    df['SMA_5'] = ta.trend.sma_indicator(df['Close'], window=5)
    df['SMA_10'] = ta.trend.sma_indicator(df['Close'], window=10)
    df['SMA_20'] = ta.trend.sma_indicator(df['Close'], window=20)
    df['EMA_12'] = ta.trend.ema_indicator(df['Close'], window=12)
    df['EMA_26'] = ta.trend.ema_indicator(df['Close'], window=26)
    
    # Volume indicators
    df['OBV'] = ta.volume.on_balance_volume(df['Close'], df['Volume'])
    df['Volume_SMA'] = ta.trend.sma_indicator(df['Volume'], window=10)
    
    # Volatility indicators
    df['ATR'] = ta.volatility.AverageTrueRange(df['High'], df['Low'], df['Close']).average_true_range()
    
    # Momentum indicators
    df['Stochastic'] = ta.momentum.StochasticOscillator(df['High'], df['Low'], df['Close']).stoch()
    df['Williams_R'] = ta.momentum.WilliamsRIndicator(df['High'], df['Low'], df['Close']).williams_r()
    
    # Price patterns
    df['Price_Change'] = df['Close'].pct_change()
    df['High_Low_Pct'] = (df['High'] - df['Low']) / df['Close']
    df['Close_to_High'] = df['Close'] / df['High']
    df['Close_to_Low'] = df['Close'] / df['Low']
    
    # Lagged features
    for lag in [1, 2, 3]:
        df[f'Close_lag_{lag}'] = df['Close'].shift(lag)
        df[f'Volume_lag_{lag}'] = df['Volume'].shift(lag)
    
    print(f"✓ Added {len([col for col in df.columns if col not in ['Date', 'Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']])} technical indicators")
    
    return df

def optimize_xgboost_hyperparameters(X_train, y_train, n_trials=50):
    """Optimize XGBoost hyperparameters using Optuna"""
    print(f"🎯 Optimizing XGBoost hyperparameters ({n_trials} trials)...")
    
    def objective(trial):
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 50, 300),
            'max_depth': trial.suggest_int('max_depth', 3, 10),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
            'random_state': 42
        }
        
        model = xgb.XGBRegressor(**params)
        scores = cross_val_score(model, X_train, y_train, cv=3, scoring='r2')
        return scores.mean()
    
    study = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler())
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    
    print(f"✓ Best R² score: {study.best_value:.4f}")
    print(f"✓ Best parameters: {study.best_params}")
    
    return study.best_params

def compare_ml_models(X_train, X_test, y_train, y_test):
    """Compare different ML models"""
    print("🤖 Comparing ML models...")
    
    models = {
        'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
        'XGBoost': xgb.XGBRegressor(n_estimators=100, random_state=42),
        'LightGBM': lgb.LGBMRegressor(n_estimators=100, random_state=42, verbose=-1)
    }
    
    results = {}
    
    for name, model in models.items():
        # Train model
        model.fit(X_train, y_train)
        
        # Make predictions
        y_pred_train = model.predict(X_train)
        y_pred_test = model.predict(X_test)
        
        # Calculate metrics
        train_r2 = r2_score(y_train, y_pred_train)
        test_r2 = r2_score(y_test, y_pred_test)
        test_mse = mean_squared_error(y_test, y_pred_test)
        test_mae = mean_absolute_error(y_test, y_pred_test)
        
        results[name] = {
            'train_r2': train_r2,
            'test_r2': test_r2,
            'test_mse': test_mse,
            'test_mae': test_mae,
            'model': model
        }
        
        print(f"  {name}:")
        print(f"    Train R²: {train_r2:.4f}")
        print(f"    Test R²:  {test_r2:.4f}")
        print(f"    Test MSE: {test_mse:.4f}")
        print(f"    Test MAE: {test_mae:.4f}")
    
    # Find best model
    best_model_name = max(results.keys(), key=lambda x: results[x]['test_r2'])
    print(f"🏆 Best model: {best_model_name} (Test R²: {results[best_model_name]['test_r2']:.4f})")
    
    return results, best_model_name

def create_advanced_visualizations(data, results, best_model_name):
    """Create advanced visualizations"""
    print("📊 Creating advanced visualizations...")
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Enhanced Asset ML Strategy - Advanced Analysis', fontsize=16, fontweight='bold')
    
    # 1. Price and indicators
    ax1 = axes[0, 0]
    ax1.plot(data['Date'], data['Close'], label='Close Price', linewidth=2)
    ax1.plot(data['Date'], data['SMA_20'], label='SMA 20', alpha=0.7)
    ax1.fill_between(data['Date'], data['BB_Lower'], data['BB_Upper'], alpha=0.2, label='Bollinger Bands')
    ax1.set_title('Price with Technical Indicators')
    ax1.set_xlabel('Date')
    ax1.set_ylabel('Price ($)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. RSI
    ax2 = axes[0, 1]
    ax2.plot(data['Date'], data['RSI'], color='purple', linewidth=1.5)
    ax2.axhline(y=70, color='r', linestyle='--', alpha=0.7, label='Overbought')
    ax2.axhline(y=30, color='g', linestyle='--', alpha=0.7, label='Oversold')
    ax2.set_title('RSI (Relative Strength Index)')
    ax2.set_xlabel('Date')
    ax2.set_ylabel('RSI')
    ax2.set_ylim(0, 100)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. Volume analysis
    ax3 = axes[0, 2]
    ax3.bar(data['Date'], data['Volume'], alpha=0.6, color='orange')
    ax3.plot(data['Date'], data['Volume_SMA'], color='red', linewidth=2, label='Volume SMA')
    ax3.set_title('Volume Analysis')
    ax3.set_xlabel('Date')
    ax3.set_ylabel('Volume')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. Model comparison
    ax4 = axes[1, 0]
    model_names = list(results.keys())
    test_r2_scores = [results[name]['test_r2'] for name in model_names]
    bars = ax4.bar(model_names, test_r2_scores, color=['lightblue', 'lightgreen', 'lightcoral'])
    ax4.set_title('Model Performance Comparison')
    ax4.set_ylabel('Test R² Score')
    ax4.set_ylim(0, 1)
    
    # Add value labels on bars
    for i, (bar, score) in enumerate(zip(bars, test_r2_scores)):
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{score:.3f}', ha='center', va='bottom', fontweight='bold')
        if model_names[i] == best_model_name:
            bar.set_color('gold')
            bar.set_edgecolor('darkgoldenrod')
            bar.set_linewidth(2)
    
    ax4.grid(True, alpha=0.3)
    
    # 5. Feature importance (for best model)
    ax5 = axes[1, 1]
    best_model = results[best_model_name]['model']
    
    if hasattr(best_model, 'feature_importances_'):
        # Get feature names (excluding target and non-numeric columns)
        feature_cols = [col for col in data.columns if col not in ['Date', 'Close']]
        # Remove columns with all NaN values
        valid_data = data[feature_cols].dropna(axis=1, how='all')
        feature_names = valid_data.columns.tolist()
        
        if len(feature_names) > 0:
            importances = best_model.feature_importances_[:len(feature_names)]
            
            # Get top 10 features
            indices = np.argsort(importances)[-10:]
            top_features = [feature_names[i] for i in indices]
            top_importances = importances[indices]
            
            ax5.barh(top_features, top_importances, color='skyblue')
            ax5.set_title(f'Top 10 Feature Importance ({best_model_name})')
            ax5.set_xlabel('Importance')
    
    ax5.grid(True, alpha=0.3)
    
    # 6. Correlation heatmap of top features
    ax6 = axes[1, 2]
    
    # Select key features for correlation
    key_features = ['Close', 'RSI', 'MACD', 'BB_Width', 'Volume', 'ATR']
    available_features = [f for f in key_features if f in data.columns]
    
    if len(available_features) > 1:
        corr_data = data[available_features].corr()
        im = ax6.imshow(corr_data.values, cmap='coolwarm', aspect='auto', vmin=-1, vmax=1)
        ax6.set_xticks(range(len(available_features)))
        ax6.set_yticks(range(len(available_features)))
        ax6.set_xticklabels(available_features, rotation=45)
        ax6.set_yticklabels(available_features)
        ax6.set_title('Feature Correlation Matrix')
        
        # Add correlation values
        for i in range(len(available_features)):
            for j in range(len(available_features)):
                text = ax6.text(j, i, f'{corr_data.iloc[i, j]:.2f}',
                               ha="center", va="center", color="black", fontsize=8)
        
        plt.colorbar(im, ax=ax6)
    
    plt.tight_layout()
    plt.savefig('/home/runner/work/Asset-ML-Strategy/Asset-ML-Strategy/enhanced_ml_analysis.png', 
                dpi=300, bbox_inches='tight')
    plt.close()
    
    print("✓ Advanced visualizations saved to 'enhanced_ml_analysis.png'")

def test_real_time_data_capability():
    """Test real-time data fetching capability"""
    print("\n💰 Testing real-time data capability...")
    
    try:
        # Test with a popular stock (Apple)
        ticker = "AAPL"
        period = "5d"  # Last 5 days
        
        print(f"  Fetching {period} data for {ticker}...")
        stock = yf.Ticker(ticker)
        data = stock.history(period=period)
        
        if not data.empty:
            print(f"✓ Successfully fetched real-time data:")
            print(f"  • Symbol: {ticker}")
            print(f"  • Period: {period}")
            print(f"  • Records: {len(data)}")
            print(f"  • Latest close: ${data['Close'].iloc[-1]:.2f}")
            print(f"  • Date range: {data.index[0].date()} to {data.index[-1].date()}")
            return True
        else:
            print("⚠ No data received")
            return False
            
    except Exception as e:
        print(f"⚠ Real-time data test failed: {e}")
        print("  (This might be due to network restrictions)")
        return False

def main():
    """Main enhanced testing function"""
    print("🚀 Enhanced Asset ML Strategy Test")
    print("=" * 60)
    
    # Generate enhanced data
    data = generate_enhanced_sample_data(n_samples=300)
    
    # Add advanced technical indicators
    data = add_advanced_technical_indicators(data)
    
    # Prepare features (remove NaN values)
    feature_cols = [col for col in data.columns if col not in ['Date', 'Close']]
    X = data[feature_cols].copy()
    y = data['Close'].copy()
    
    # Remove rows with NaN values
    mask = ~(X.isnull().any(axis=1) | y.isnull())
    X = X[mask]
    y = y[mask]
    
    print(f"\n🎯 Prepared dataset:")
    print(f"  • Samples: {len(X)}")
    print(f"  • Features: {len(X.columns)}")
    print(f"  • Target variable: Close price")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, shuffle=False
    )
    
    print(f"  • Training samples: {len(X_train)}")
    print(f"  • Testing samples: {len(X_test)}")
    
    # Compare models
    results, best_model = compare_ml_models(X_train, X_test, y_train, y_test)
    
    # Optimize best model (if it's XGBoost)
    if best_model == 'XGBoost':
        print(f"\n🎯 Optimizing {best_model}...")
        best_params = optimize_xgboost_hyperparameters(X_train, y_train, n_trials=30)
        
        # Train optimized model
        optimized_model = xgb.XGBRegressor(**best_params)
        optimized_model.fit(X_train, y_train)
        
        # Test optimized model
        y_pred_optimized = optimized_model.predict(X_test)
        optimized_r2 = r2_score(y_test, y_pred_optimized)
        
        print(f"✓ Optimized model Test R²: {optimized_r2:.4f}")
        print(f"✓ Improvement: {optimized_r2 - results[best_model]['test_r2']:.4f}")
        
        # Update results
        results['XGBoost (Optimized)'] = {
            'train_r2': r2_score(y_train, optimized_model.predict(X_train)),
            'test_r2': optimized_r2,
            'test_mse': mean_squared_error(y_test, y_pred_optimized),
            'test_mae': mean_absolute_error(y_test, y_pred_optimized),
            'model': optimized_model
        }
        best_model = 'XGBoost (Optimized)'
    
    # Create visualizations
    create_advanced_visualizations(data, results, best_model)
    
    # Test real-time data capability
    real_time_success = test_real_time_data_capability()
    
    # Final summary
    print(f"\n🏆 ENHANCED ML STRATEGY RESULTS")
    print("=" * 60)
    print(f"✅ Best model: {best_model}")
    print(f"✅ Best Test R²: {results[best_model]['test_r2']:.4f}")
    print(f"✅ Technical indicators: {len(feature_cols)} features")
    print(f"✅ Real-time data: {'Available' if real_time_success else 'Limited'}")
    print(f"✅ Advanced visualizations: Created")
    print(f"✅ Hyperparameter optimization: {'Applied' if 'Optimized' in best_model else 'Available'}")
    
    improvement_score = (
        (results[best_model]['test_r2'] * 100) +  # Performance score
        (len(feature_cols) * 0.5) +               # Feature richness
        (10 if real_time_success else 0) +        # Real-time capability
        (5 if 'Optimized' in best_model else 0)   # Optimization bonus
    )
    
    print(f"\n🎯 Overall Enhancement Score: {improvement_score:.1f}/150")
    
    if improvement_score > 120:
        print("🌟 EXCELLENT - Professional-grade ML strategy!")
    elif improvement_score > 100:
        print("🚀 GREAT - Solid enhanced ML capabilities!")
    elif improvement_score > 80:
        print("👍 GOOD - Significant improvements achieved!")
    else:
        print("📈 BASIC - More enhancements possible!")
    
    return True

if __name__ == "__main__":
    main()