#!/usr/bin/env python3
"""
Quick Demo Script for Ultimate Excel MoneyPrinter
Shows how to transform basic Excel files into advanced ML trading system.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from ultimate_excel_moneyprinter import UltimateExcelMoneyPrinter
import warnings
warnings.filterwarnings('ignore')

def create_sample_excel_file(filename: str, asset_name: str, days: int = 1000):
    """Create a realistic sample Excel file with OHLCV data."""
    
    # Generate realistic price data
    np.random.seed(hash(asset_name) % 1000)  # Consistent seed based on asset name
    
    base_price = 100.0
    if 'BTC' in asset_name:
        base_price = 50000.0
    elif 'ETH' in asset_name:
        base_price = 3000.0
    elif 'SPX' in asset_name:
        base_price = 4000.0
    
    # Create realistic price movements
    returns = np.random.randn(days) * 0.02  # 2% daily volatility
    
    # Add trending behavior
    trend = np.sin(np.arange(days) * 0.01) * 0.005
    returns = returns + trend
    
    # Add momentum effects
    for i in range(1, len(returns)):
        returns[i] += returns[i-1] * 0.1  # Momentum effect
    
    # Calculate prices
    prices = base_price * np.cumprod(1 + returns)
    
    # Create OHLCV data
    data = {
        'Date': pd.date_range('2020-01-01', periods=days, freq='D'),
        'Open': prices,
        'High': prices * (1 + np.random.uniform(0, 0.03, days)),
        'Low': prices * (1 - np.random.uniform(0, 0.03, days)),
        'Close': prices * (1 + np.random.randn(days) * 0.01),
        'Adj Close': prices * (1 + np.random.randn(days) * 0.005),  # Less noisy than close
        'Volume': np.random.randint(10000, 1000000, days)
    }
    
    # Ensure high >= low and other realistic constraints
    df = pd.DataFrame(data)
    df['High'] = np.maximum(df[['Open', 'Close']].max(axis=1), df['High'])
    df['Low'] = np.minimum(df[['Open', 'Close']].min(axis=1), df['Low'])
    
    # Save to Excel
    df.to_excel(filename, index=False)
    print(f"✅ Created {filename} with {len(df)} rows of {asset_name} data")
    
    return filename

def quick_demo():
    """Run a comprehensive demo of the Ultimate Excel MoneyPrinter."""
    
    print("🚀 ULTIMATE EXCEL MONEYPRINTER - QUICK DEMO")
    print("=" * 60)
    
    try:
        # Step 1: Create sample Excel files
        print("\n📊 STEP 1: Creating Sample Excel Files")
        print("-" * 40)
        
        sample_files = []
        assets = ['BTCUSD', 'ETHUSD', 'SPXUSD', 'GOLD', 'AAPL']
        
        for asset in assets:
            filename = f"/tmp/demo_{asset}.xlsx"
            create_sample_excel_file(filename, asset, days=800)
            sample_files.append(filename)
        
        # Step 2: Initialize the system
        print(f"\n🔥 STEP 2: Initializing Ultimate Excel MoneyPrinter")
        print("-" * 50)
        
        moneyprinter = UltimateExcelMoneyPrinter()
        
        # Step 3: Process Excel files
        print(f"\n📈 STEP 3: Processing {len(sample_files)} Excel Files")
        print("-" * 45)
        
        processed_data = moneyprinter.process_excel_files(sample_files, parallel=True)
        
        print(f"\n✅ Successfully processed {len(processed_data)} datasets!")
        for name, df in processed_data.items():
            print(f"   📊 {name}: {len(df)} rows × {len(df.columns)} features")
        
        # Step 4: Train ML models (simplified for demo)
        print(f"\n🧠 STEP 4: Training Next-Generation ML Models")
        print("-" * 45)
        
        # Use a simplified configuration for the demo
        moneyprinter.ml_ensemble.config['epochs'] = 5  # Fast training for demo
        moneyprinter.ml_ensemble.config['hyperparameter_optimization'] = False
        moneyprinter.ml_ensemble.config['models'] = [
            moneyprinter.ml_ensemble.config['models'][0]  # Just use first model for demo
        ]
        
        training_results = moneyprinter.train_ml_models('combined')
        
        if training_results['success']:
            print(f"✅ Successfully trained {training_results['models_trained']} model(s)!")
            print(f"   📊 Training samples: {training_results['sample_count']:,}")
            print(f"   🔢 Features used: {training_results['feature_count']}")
        else:
            print(f"❌ Training failed: {training_results['error']}")
            return
        
        # Step 5: Make predictions
        print(f"\n🔮 STEP 5: Making Predictions & Generating Signals")
        print("-" * 48)
        
        prediction_results = moneyprinter.make_predictions('combined', latest_n_points=200)
        
        if prediction_results['success']:
            print(f"✅ Generated predictions successfully!")
            print(f"   🎯 Mean prediction: {prediction_results['mean_prediction']:.4f}")
            print(f"   📊 Mean confidence: {prediction_results['mean_confidence']:.2f}")
            
            signals = prediction_results['signal_summary']
            print(f"   🔔 Total signals: {signals['total_signals']}")
            print(f"   📈 Buy signals: {signals['buy_signals']}")
            print(f"   📉 Sell signals: {signals['sell_signals']}")
            print(f"   ⚡ Signal strength: {signals['avg_signal_strength']:.3f}")
        else:
            print(f"❌ Predictions failed: {prediction_results['error']}")
            return
        
        # Step 6: Backtest
        print(f"\n📈 STEP 6: Comprehensive Backtesting")
        print("-" * 35)
        
        backtest_results = moneyprinter.backtest_strategy('combined', initial_capital=100000)
        
        if backtest_results['success']:
            print(f"\n🏆 ULTIMATE EXCEL MONEYPRINTER RESULTS:")
            print(f"💰 Initial Capital: ${backtest_results['initial_capital']:,.2f}")
            print(f"💎 Final Capital: ${backtest_results['final_capital']:,.2f}")
            print(f"📊 Total Return: {backtest_results['total_return']:.2%}")
            print(f"📈 Annualized Return: {backtest_results['annualized_return']:.2%}")
            print(f"⚡ Sharpe Ratio: {backtest_results['sharpe_ratio']:.2f}")
            print(f"🛡️ Max Drawdown: {backtest_results['max_drawdown']:.2%}")
            print(f"🎯 Win Rate: {backtest_results['win_rate']:.2%}")
            print(f"💎 Profit Factor: {backtest_results['profit_factor']:.2f}")
            print(f"📋 Total Trades: {backtest_results['total_trades']}")
        else:
            print(f"❌ Backtesting failed: {backtest_results['error']}")
            return
        
        # Step 7: System status
        print(f"\n📊 STEP 7: System Status")
        print("-" * 25)
        
        status = moneyprinter.get_system_status()
        print(f"✅ System Status:")
        print(f"   📊 Datasets processed: {status['processed_datasets']}")
        print(f"   🧠 Models trained: {status['trained_models']}")
        print(f"   🔮 Live predictions: {status['live_predictions']}")
        print(f"   🔴 Live trading: {status['is_live_trading']}")
        
        # Success summary
        print(f"\n" + "🎉" * 25)
        print(f"🚀 DEMO COMPLETED SUCCESSFULLY! 🚀")
        print(f"🎉" * 25)
        
        print(f"\n💎 Your Excel files have been transformed into:")
        print(f"   ✅ {processed_data['combined'].shape[1]} advanced ML features")
        print(f"   ✅ Next-generation neural network models")
        print(f"   ✅ Sophisticated trading signals")
        print(f"   ✅ Comprehensive risk management")
        print(f"   ✅ Professional backtesting results")
        
        print(f"\n🔥 Key Achievements:")
        total_return_pct = backtest_results.get('total_return', 0) * 100
        if total_return_pct > 0:
            print(f"   💰 Profitable strategy: +{total_return_pct:.1f}% return")
        else:
            print(f"   📊 Strategy performance: {total_return_pct:.1f}% return")
        
        print(f"   ⚡ Advanced ML processing: {len(sample_files)} files → {processed_data['combined'].shape[1]} features")
        print(f"   🧠 AI-powered predictions with {prediction_results['mean_confidence']:.0%} confidence")
        print(f"   🛡️ Risk-managed trading signals")
        
        print(f"\n🎯 Ready for live trading with your actual Excel files!")
        print(f"   📁 Replace demo files with your real trading data")
        print(f"   🔄 Enable live monitoring for continuous learning")
        print(f"   💼 Scale up capital for maximum profits")
        
    except Exception as e:
        print(f"❌ Demo failed: {str(e)}")
        print(f"💡 Try running with debug logging for more details")
        import traceback
        traceback.print_exc()

def show_usage_examples():
    """Show practical usage examples."""
    
    print("\n📖 PRACTICAL USAGE EXAMPLES")
    print("=" * 40)
    
    print("""
🔥 Basic Usage:
from ultimate_excel_moneyprinter import UltimateExcelMoneyPrinter

# Initialize
moneyprinter = UltimateExcelMoneyPrinter()

# Process your Excel files
files = ['btc_data.xlsx', 'eth_data.xlsx', 'stock_data.xlsx']
data = moneyprinter.process_excel_files(files)

# Train AI models
moneyprinter.train_ml_models('combined')

# Make predictions
predictions = moneyprinter.make_predictions('combined')

# Backtest strategy
results = moneyprinter.backtest_strategy('combined', 100000)

🚀 Advanced Usage:
# Live monitoring (updates every 4 hours)
moneyprinter.start_live_monitoring(files, "4h")

# Save system for later use
moneyprinter.save_system_state("my_trading_system")

# Load saved system
moneyprinter.load_system_state("my_trading_system")

📊 Required Excel Format:
Your Excel files should have columns:
- Date (or Time/Timestamp)
- Open
- High  
- Low
- Close
- Adj Close
- Volume

💡 The system automatically handles column name variations!
""")

if __name__ == "__main__":
    print("🎯 Choose an option:")
    print("1. Run quick demo")
    print("2. Show usage examples")
    
    choice = input("Enter choice (1 or 2, or press Enter for demo): ").strip()
    
    if choice == "2":
        show_usage_examples()
    else:
        quick_demo()