#!/usr/bin/env python3
"""
Robust ML Test - Fix overfitting and create a production-ready ML pipeline
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend

# ML libraries
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, TimeSeriesSplit, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler, RobustScaler
import xgboost as xgb
import lightgbm as lgb

# Technical analysis
import ta

import warnings
warnings.filterwarnings('ignore')

def generate_realistic_data(n_samples=1000):
    """Generate more realistic financial data"""
    print("📊 Generating realistic financial data...")
    
    dates = pd.date_range('2020-01-01', periods=n_samples, freq='D')
    np.random.seed(42)
    
    # Create more realistic price movements
    returns = np.random.normal(0.0005, 0.02, n_samples)  # Daily returns
    
    # Add momentum and mean reversion
    for i in range(1, len(returns)):
        momentum = returns[i-1] * 0.1  # Momentum effect
        returns[i] += momentum
    
    # Convert to price levels
    prices = 100 * np.exp(np.cumsum(returns))
    
    # Generate OHLC with realistic relationships
    data = []
    for i in range(n_samples):
        close = prices[i]
        
        # Generate realistic OHLC relationships
        volatility = abs(np.random.normal(0, 0.01))
        high_low_spread = close * volatility
        
        high = close + np.random.exponential(high_low_spread/3)
        low = close - np.random.exponential(high_low_spread/3)
        
        # Open based on previous close with gap
        if i == 0:
            open_price = close
        else:
            gap = np.random.normal(0, close * 0.005)
            open_price = max(low, min(high, prices[i-1] + gap))
        
        # Ensure OHLC relationships
        high = max(high, open_price, close)
        low = min(low, open_price, close)
        
        # Volume with realistic distribution
        volume = int(np.random.lognormal(10, 1))
        
        data.append({
            'Date': dates[i],
            'Open': round(open_price, 2),
            'High': round(high, 2),
            'Low': round(low, 2),
            'Close': round(close, 2),
            'Adj Close': round(close, 2),
            'Volume': volume
        })
    
    df = pd.DataFrame(data)
    print(f"✓ Generated {len(df)} realistic samples")
    print(f"  Price range: ${df['Close'].min():.2f} to ${df['Close'].max():.2f}")
    
    return df

def add_robust_features(df):
    """Add robust technical features that avoid data leakage"""
    print("🔧 Adding robust technical features...")
    
    # Basic price features (no future data)
    df['Returns'] = df['Close'].pct_change()
    df['Log_Returns'] = np.log(df['Close'] / df['Close'].shift(1))
    df['Price_Range'] = (df['High'] - df['Low']) / df['Close']
    df['Body_Size'] = abs(df['Close'] - df['Open']) / df['Close']
    
    # Moving averages (lagged to avoid lookahead bias)
    for window in [5, 10, 20]:
        df[f'SMA_{window}'] = df['Close'].shift(1).rolling(window=window).mean()
        df[f'Price_vs_SMA_{window}'] = (df['Close'] / df[f'SMA_{window}'] - 1)
    
    # Volatility features
    df['Volatility_5'] = df['Returns'].shift(1).rolling(window=5).std()
    df['Volatility_20'] = df['Returns'].shift(1).rolling(window=20).std()
    
    # Volume features
    df['Volume_SMA'] = df['Volume'].shift(1).rolling(window=10).mean()
    df['Volume_Ratio'] = df['Volume'] / df['Volume_SMA']
    
    # RSI (properly calculated to avoid lookahead)
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    df['RSI'] = df['RSI'].shift(1)  # Shift to avoid lookahead
    
    # Lagged price features
    for lag in [1, 2, 3, 5]:
        df[f'Returns_lag_{lag}'] = df['Returns'].shift(lag)
        df[f'Volume_lag_{lag}'] = (df['Volume'] / df['Volume_SMA']).shift(lag)
    
    # Target preparation (next day's return)
    df['Target_Return'] = df['Close'].shift(-1) / df['Close'] - 1
    df['Target_Price'] = df['Close'].shift(-1)
    
    print(f"✓ Added robust features (no lookahead bias)")
    
    return df

def prepare_ml_data(df):
    """Prepare data for ML with proper validation"""
    print("🎯 Preparing ML dataset...")
    
    # Feature columns (exclude target and non-predictive columns)
    exclude_cols = ['Date', 'Close', 'High', 'Low', 'Open', 'Adj Close', 
                   'Target_Return', 'Target_Price']
    feature_cols = [col for col in df.columns if col not in exclude_cols]
    
    # Create feature matrix and target
    X = df[feature_cols].copy()
    y = df['Target_Return'].copy()  # Predict next day's return
    
    # Remove rows with NaN values
    valid_mask = ~(X.isnull().any(axis=1) | y.isnull())
    X = X[valid_mask]
    y = y[valid_mask]
    
    print(f"✓ Dataset prepared:")
    print(f"  • Valid samples: {len(X)}")
    print(f"  • Features: {len(feature_cols)}")
    print(f"  • Target: Next day returns")
    print(f"  • Features: {feature_cols[:5]}... (+{len(feature_cols)-5} more)")
    
    return X, y, feature_cols

def robust_model_comparison(X, y):
    """Compare models using proper time series validation"""
    print("🤖 Comparing ML models with time series validation...")
    
    # Use time series split for validation
    tscv = TimeSeriesSplit(n_splits=5, test_size=100)
    
    models = {
        'Random Forest': RandomForestRegressor(
            n_estimators=100, max_depth=8, min_samples_split=20,
            min_samples_leaf=10, random_state=42
        ),
        'XGBoost': xgb.XGBRegressor(
            n_estimators=100, max_depth=6, learning_rate=0.1,
            subsample=0.8, colsample_bytree=0.8, random_state=42
        ),
        'LightGBM': lgb.LGBMRegressor(
            n_estimators=100, max_depth=6, learning_rate=0.1,
            subsample=0.8, colsample_bytree=0.8, random_state=42, verbose=-1
        )
    }
    
    results = {}
    
    for name, model in models.items():
        print(f"  Testing {name}...")
        
        # Time series cross-validation scores
        cv_scores = cross_val_score(model, X, y, cv=tscv, scoring='r2')
        
        # Final split for detailed metrics
        split_idx = int(len(X) * 0.8)
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        
        # Scale features
        scaler = RobustScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Train model
        if name == 'Random Forest':
            # RF doesn't need scaling
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
        else:
            # Scale for tree-based models to be consistent
            model.fit(X_train_scaled, y_train)
            y_pred = model.predict(X_test_scaled)
        
        # Calculate metrics
        test_r2 = r2_score(y_test, y_pred)
        test_mse = mean_squared_error(y_test, y_pred)
        test_mae = mean_absolute_error(y_test, y_pred)
        
        # Directional accuracy (most important for trading)
        direction_correct = np.mean(np.sign(y_test) == np.sign(y_pred))
        
        results[name] = {
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std(),
            'test_r2': test_r2,
            'test_mse': test_mse,
            'test_mae': test_mae,
            'direction_accuracy': direction_correct,
            'model': model,
            'scaler': scaler if name != 'Random Forest' else None
        }
        
        print(f"    CV R² (mean±std): {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
        print(f"    Test R²: {test_r2:.4f}")
        print(f"    Directional Accuracy: {direction_correct:.1%}")
    
    # Find best model based on CV performance and directional accuracy
    best_model = max(results.keys(), 
                    key=lambda x: results[x]['cv_mean'] + results[x]['direction_accuracy'])
    
    print(f"\n🏆 Best model: {best_model}")
    print(f"  CV R²: {results[best_model]['cv_mean']:.4f}")
    print(f"  Direction Accuracy: {results[best_model]['direction_accuracy']:.1%}")
    
    return results, best_model

def create_trading_backtest(df, model_results, best_model_name):
    """Create a simple trading backtest"""
    print("💹 Running trading backtest...")
    
    # Get the best model
    best_model = model_results[best_model_name]['model']
    scaler = model_results[best_model_name].get('scaler')
    
    # Prepare features for the last part of data
    feature_cols = [col for col in df.columns if col not in 
                   ['Date', 'Close', 'High', 'Low', 'Open', 'Adj Close', 
                    'Target_Return', 'Target_Price']]
    
    # Use last 200 days for backtest
    backtest_data = df.iloc[-250:].copy()  # Extra for warmup
    backtest_data = backtest_data.dropna().iloc[-200:]  # Clean backtest period
    
    X_backtest = backtest_data[feature_cols]
    
    if scaler:
        X_backtest = scaler.transform(X_backtest)
    
    # Predict returns
    predicted_returns = best_model.predict(X_backtest)
    actual_returns = backtest_data['Target_Return'].values[:-1]  # Remove last NaN
    
    # Simple trading strategy: long when prediction > threshold
    threshold = np.percentile(predicted_returns, 60)  # Top 40% predictions
    
    signals = (predicted_returns[:-1] > threshold).astype(int)  # Align with actual returns
    
    # Calculate strategy performance
    strategy_returns = signals * actual_returns
    buy_hold_returns = actual_returns
    
    # Cumulative returns
    strategy_cumret = (1 + strategy_returns).cumprod()
    buyhold_cumret = (1 + buy_hold_returns).cumprod()
    
    # Performance metrics
    strategy_total_return = strategy_cumret[-1] - 1
    buyhold_total_return = buyhold_cumret[-1] - 1
    
    strategy_sharpe = strategy_returns.mean() / strategy_returns.std() * np.sqrt(252) if strategy_returns.std() > 0 else 0
    buyhold_sharpe = buy_hold_returns.mean() / buy_hold_returns.std() * np.sqrt(252)
    
    # Win rate
    win_rate = np.mean(strategy_returns > 0)
    
    print(f"✓ Backtest Results ({len(actual_returns)} days):")
    print(f"  Strategy Return: {strategy_total_return:.2%}")
    print(f"  Buy & Hold Return: {buyhold_total_return:.2%}")
    print(f"  Strategy Sharpe: {strategy_sharpe:.2f}")
    print(f"  Buy & Hold Sharpe: {buyhold_sharpe:.2f}")
    print(f"  Win Rate: {win_rate:.1%}")
    print(f"  Trades Taken: {signals.sum()}/{len(signals)} days")
    
    return {
        'strategy_returns': strategy_returns,
        'buyhold_returns': buy_hold_returns,
        'strategy_cumret': strategy_cumret,
        'buyhold_cumret': buyhold_cumret,
        'signals': signals,
        'predicted_returns': predicted_returns[:-1],
        'actual_returns': actual_returns,
        'performance': {
            'strategy_return': strategy_total_return,
            'buyhold_return': buyhold_total_return,
            'strategy_sharpe': strategy_sharpe,
            'buyhold_sharpe': buyhold_sharpe,
            'win_rate': win_rate
        }
    }

def create_comprehensive_visualizations(df, model_results, backtest_results):
    """Create comprehensive visualizations"""
    print("📊 Creating comprehensive visualizations...")
    
    fig, axes = plt.subplots(3, 2, figsize=(16, 18))
    fig.suptitle('Robust Asset ML Strategy - Comprehensive Analysis', fontsize=16, fontweight='bold')
    
    # 1. Price and features over time
    ax1 = axes[0, 0]
    recent_data = df.iloc[-200:]
    ax1.plot(recent_data['Date'], recent_data['Close'], label='Close Price', linewidth=2)
    if 'SMA_20' in recent_data.columns:
        ax1.plot(recent_data['Date'], recent_data['SMA_20'], label='SMA 20', alpha=0.7)
    ax1.set_title('Price with Technical Indicators')
    ax1.set_xlabel('Date')
    ax1.set_ylabel('Price ($)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Model performance comparison
    ax2 = axes[0, 1]
    models = list(model_results.keys())
    cv_scores = [model_results[model]['cv_mean'] for model in models]
    direction_acc = [model_results[model]['direction_accuracy'] for model in models]
    
    x = np.arange(len(models))
    width = 0.35
    
    bars1 = ax2.bar(x - width/2, cv_scores, width, label='CV R²', alpha=0.8)
    bars2 = ax2.bar(x + width/2, direction_acc, width, label='Direction Accuracy', alpha=0.8)
    
    ax2.set_title('Model Performance Comparison')
    ax2.set_xlabel('Models')
    ax2.set_ylabel('Score')
    ax2.set_xticks(x)
    ax2.set_xticklabels(models)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Add value labels
    for bar in bars1:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.3f}', ha='center', va='bottom', fontsize=8)
    
    for bar in bars2:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.3f}', ha='center', va='bottom', fontsize=8)
    
    # 3. Predictions vs Actual
    ax3 = axes[1, 0]
    pred_returns = backtest_results['predicted_returns']
    actual_returns = backtest_results['actual_returns']
    
    ax3.scatter(pred_returns, actual_returns, alpha=0.6, s=10)
    
    # Add trend line
    z = np.polyfit(pred_returns, actual_returns, 1)
    p = np.poly1d(z)
    ax3.plot(pred_returns, p(pred_returns), "r--", alpha=0.8)
    
    ax3.set_title('Predicted vs Actual Returns')
    ax3.set_xlabel('Predicted Returns')
    ax3.set_ylabel('Actual Returns')
    ax3.grid(True, alpha=0.3)
    
    # Add correlation text
    correlation = np.corrcoef(pred_returns, actual_returns)[0, 1]
    ax3.text(0.05, 0.95, f'Correlation: {correlation:.3f}', transform=ax3.transAxes,
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # 4. Trading strategy performance
    ax4 = axes[1, 1]
    dates = df.iloc[-len(backtest_results['strategy_cumret']):]['Date']
    
    ax4.plot(dates, backtest_results['strategy_cumret'], label='ML Strategy', linewidth=2)
    ax4.plot(dates, backtest_results['buyhold_cumret'], label='Buy & Hold', linewidth=2)
    
    ax4.set_title('Cumulative Returns Comparison')
    ax4.set_xlabel('Date')
    ax4.set_ylabel('Cumulative Return')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    # 5. Feature importance (if available)
    ax5 = axes[2, 0]
    best_model_name = max(model_results.keys(), 
                         key=lambda x: model_results[x]['cv_mean'])
    best_model = model_results[best_model_name]['model']
    
    if hasattr(best_model, 'feature_importances_'):
        feature_cols = [col for col in df.columns if col not in 
                       ['Date', 'Close', 'High', 'Low', 'Open', 'Adj Close', 
                        'Target_Return', 'Target_Price']]
        
        importances = best_model.feature_importances_
        indices = np.argsort(importances)[-10:]  # Top 10
        
        ax5.barh(range(len(indices)), importances[indices])
        ax5.set_yticks(range(len(indices)))
        ax5.set_yticklabels([feature_cols[i] for i in indices])
        ax5.set_title(f'Top 10 Feature Importance ({best_model_name})')
        ax5.set_xlabel('Importance')
    
    ax5.grid(True, alpha=0.3)
    
    # 6. Return distribution
    ax6 = axes[2, 1]
    strategy_returns = backtest_results['strategy_returns']
    buyhold_returns = backtest_results['buyhold_returns']
    
    ax6.hist(strategy_returns, bins=30, alpha=0.7, label='ML Strategy', density=True)
    ax6.hist(buyhold_returns, bins=30, alpha=0.7, label='Buy & Hold', density=True)
    
    ax6.set_title('Return Distributions')
    ax6.set_xlabel('Daily Returns')
    ax6.set_ylabel('Density')
    ax6.legend()
    ax6.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('/home/runner/work/Asset-ML-Strategy/Asset-ML-Strategy/robust_ml_analysis.png', 
                dpi=300, bbox_inches='tight')
    plt.close()
    
    print("✓ Comprehensive visualizations saved to 'robust_ml_analysis.png'")

def main():
    """Main robust testing function"""
    print("🚀 Robust Asset ML Strategy Test")
    print("=" * 60)
    
    # Generate realistic data
    df = generate_realistic_data(n_samples=1000)
    
    # Add robust features
    df = add_robust_features(df)
    
    # Prepare ML data
    X, y, feature_cols = prepare_ml_data(df)
    
    # Compare models with proper validation
    model_results, best_model = robust_model_comparison(X, y)
    
    # Run trading backtest
    backtest_results = create_trading_backtest(df, model_results, best_model)
    
    # Create visualizations
    create_comprehensive_visualizations(df, model_results, backtest_results)
    
    # Final summary
    print(f"\n🏆 ROBUST ML STRATEGY RESULTS")
    print("=" * 60)
    
    best_cv_score = model_results[best_model]['cv_mean']
    best_direction_acc = model_results[best_model]['direction_accuracy']
    strategy_performance = backtest_results['performance']
    
    print(f"✅ Best Model: {best_model}")
    print(f"✅ Cross-Validation R²: {best_cv_score:.4f}")
    print(f"✅ Directional Accuracy: {best_direction_acc:.1%}")
    print(f"✅ Strategy Return: {strategy_performance['strategy_return']:.2%}")
    print(f"✅ Buy & Hold Return: {strategy_performance['buyhold_return']:.2%}")
    print(f"✅ Strategy Sharpe Ratio: {strategy_performance['strategy_sharpe']:.2f}")
    print(f"✅ Win Rate: {strategy_performance['win_rate']:.1%}")
    
    # Calculate overall score
    performance_score = (
        max(0, best_cv_score * 50) +  # CV performance (0-50)
        (best_direction_acc * 30) +    # Direction accuracy (0-30)
        max(0, min(20, strategy_performance['strategy_sharpe'] * 10))  # Sharpe ratio (0-20)
    )
    
    print(f"\n🎯 Overall Performance Score: {performance_score:.1f}/100")
    
    if performance_score > 80:
        print("🌟 EXCELLENT - Production-ready ML strategy!")
    elif performance_score > 60:
        print("🚀 GREAT - Strong ML performance!")
    elif performance_score > 40:
        print("👍 GOOD - Solid ML foundation!")
    else:
        print("📈 DEVELOPING - Good start, more refinement needed!")
    
    # Actionable insights
    print(f"\n💡 Key Insights:")
    if best_direction_acc > 0.55:
        print(f"  ✓ Model shows predictive power for market direction")
    if strategy_performance['strategy_return'] > strategy_performance['buyhold_return']:
        print(f"  ✓ ML strategy outperforms buy & hold")
    if strategy_performance['strategy_sharpe'] > 1.0:
        print(f"  ✓ Strategy achieves strong risk-adjusted returns")
    
    return True

if __name__ == "__main__":
    main()