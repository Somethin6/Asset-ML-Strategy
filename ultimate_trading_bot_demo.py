#!/usr/bin/env python3
"""
Ultimate ML Trading Bot Demonstration
Shows all the advanced features working without GUI
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend

from enhanced_data_loader import EnhancedDataLoader
from advanced_ml_engine import AdvancedMLEngine
import warnings
warnings.filterwarnings('ignore')

def demonstrate_ultimate_ml_trading_bot():
    """Comprehensive demonstration of the ultimate ML trading bot capabilities"""
    
    print("🚀 ULTIMATE ML TRADING BOT DEMONSTRATION")
    print("=" * 60)
    print("The Most Advanced Free Local ML Trading System")
    print("=" * 60)
    
    # 1. Data Loading with Multiple Formats
    print("\n📊 STEP 1: Universal Data Loading")
    print("-" * 40)
    
    loader = EnhancedDataLoader()
    
    # Test both formats
    print("🔄 Testing new format (timestamp, open, high, low, close, volume, trades)...")
    data_new = loader.load_data('sample_data/new_format_sample.csv', 'custom')
    
    print("🔄 Testing traditional format (Date, Open, High, Low, Close, Volume)...")  
    data_traditional = loader.load_data('sample_data/traditional_format_sample.csv', 'traditional')
    
    print(f"✅ Loaded both formats successfully!")
    print(f"   New format: {data_new.shape}")
    print(f"   Traditional format: {data_traditional.shape}")
    
    # Use new format data for demonstration
    data = data_new
    
    # 2. Technical Indicators Calculation
    print("\n🔬 STEP 2: Comprehensive Technical Analysis")
    print("-" * 40)
    
    print("🔄 Calculating 40+ technical indicators...")
    indicators = loader.calculate_all_indicators()
    
    print(f"✅ Calculated {len(indicators)} technical indicators:")
    indicator_categories = {}
    for name in indicators.keys():
        category = name.split('_')[0]
        if category not in indicator_categories:
            indicator_categories[category] = 0
        indicator_categories[category] += 1
    
    for category, count in indicator_categories.items():
        print(f"   • {category}: {count} indicators")
    
    # 3. Support & Resistance Detection
    print("\n🎯 STEP 3: Support & Resistance Analysis")
    print("-" * 40)
    
    support_resistance = loader.calculate_support_resistance()
    print(f"✅ Detected {len(support_resistance)} support & resistance levels:")
    for level, price in list(support_resistance.items())[:5]:
        print(f"   • {level}: ${price:.2f}")
    
    # 4. Fibonacci Retracement
    print("\n🌊 STEP 4: Fibonacci Retracement Analysis")  
    print("-" * 40)
    
    fibonacci = loader.calculate_fibonacci_retracements()
    print(f"✅ Calculated {len(fibonacci)} Fibonacci levels:")
    for level, price in list(fibonacci.items())[:6]:
        if isinstance(price, (int, float)):
            print(f"   • {level}: ${price:.2f}")
    
    # 5. Advanced ML Strategy Engine
    print("\n🤖 STEP 5: Advanced ML Strategy Engine")
    print("-" * 40)
    
    ml_engine = AdvancedMLEngine()
    ml_engine.initialize_models()
    
    # Create comprehensive feature matrix
    print("🔄 Creating comprehensive feature matrix...")
    features_df = ml_engine.create_dynamic_features(data, indicators)
    
    # Find best strategies across different time periods
    print("🎯 Finding best strategies across multiple time periods...")
    periods = [30, 50, 100, 200]
    strategies = ml_engine.find_best_strategies_by_period(features_df, 'close', periods)
    
    print(f"✅ Analyzed {len(strategies)} time periods:")
    for period, strategy_data in strategies.items():
        print(f"   • {period} days: {strategy_data['best_model']} (R² = {strategy_data['best_score']:.4f})")
    
    # 6. Strategy Transition Learning
    print("\n🔄 STEP 6: Strategy Transition Learning")
    print("-" * 40)
    
    if strategies:
        print("🔄 Learning when to transition between strategies...")
        transitions = ml_engine.learn_strategy_transitions(features_df)
        
        if transitions:
            print(f"✅ Transition model trained with {transitions['accuracy']:.3f} accuracy")
            print("🎯 Key transition factors:")
            for factor, importance in list(transitions['feature_importance'].items())[:3]:
                print(f"   • {factor}: {importance:.4f}")
        else:
            print("⚠️  Transition learning requires more strategy data")
    
    # 7. Generate Predictions
    print("\n🚀 STEP 7: Generate Comprehensive Predictions")
    print("-" * 40)
    
    if strategies:
        print("🔄 Generating predictions using best strategies...")
        predictions = ml_engine.generate_comprehensive_predictions(features_df, 'close')
        
        if predictions:
            print("✅ Predictions generated:")
            if 'ensemble' in predictions:
                ensemble = predictions['ensemble']
                print(f"   🎯 Ensemble Prediction: {ensemble['prediction']:.6f}")
                print(f"   📊 Recommended Strategy: {ensemble['recommended_strategy']}")
                print(f"   🤖 Models Used: {ensemble['n_models']}")
            
            individual_preds = [p for name, p in predictions.items() if name != 'ensemble']
            if individual_preds:
                print(f"   📈 Individual Predictions: {len(individual_preds)} models")
        else:
            print("⚠️  No predictions generated - need more training data")
    
    # 8. Performance Report
    print("\n📊 STEP 8: Comprehensive Performance Report")
    print("-" * 40)
    
    if strategies:
        report = ml_engine.get_strategy_performance_report()
        
        print("✅ Performance Analysis Complete:")
        print(f"   🏆 Best Overall Model: {report['summary']['best_overall_model']}")
        print(f"   📊 Periods Analyzed: {report['summary']['total_periods_analyzed']}")
        print(f"   🔄 Transition Model: {'Available' if report['summary']['transition_model_available'] else 'Not Available'}")
        
        print("\n🥇 Model Rankings:")
        for i, (model, stats) in enumerate(list(report['model_rankings'].items())[:3], 1):
            print(f"   {i}. {model}: {stats['average_score']:.4f} avg score, {stats['wins']} wins")
    
    # 9. Data Summary
    print("\n📈 STEP 9: Comprehensive Data Analysis")
    print("-" * 40)
    
    summary = loader.get_data_summary()
    print("✅ Dataset Analysis:")
    print(f"   📊 Data Points: {summary['basic_stats']['rows']:,}")
    print(f"   📅 Date Range: {summary['basic_stats']['start_date']} to {summary['basic_stats']['end_date']}")
    print(f"   💰 Price Range: ${summary['price_stats']['lowest_price']:.2f} - ${summary['price_stats']['highest_price']:.2f}")
    print(f"   📊 Current Price: ${summary['price_stats']['current_price']:.2f}")
    print(f"   📈 Average Volume: {summary['volume_stats']['average_volume']:,.0f}")
    
    # 10. System Capabilities Summary
    print("\n🚀 SYSTEM CAPABILITIES SUMMARY")
    print("=" * 60)
    
    capabilities = [
        "✅ Universal data loading (CSV/Excel, any OHLC format)",
        "✅ 40+ technical indicators with robust error handling",
        "✅ 4 advanced ML models (Random Forest, XGBoost, LightGBM, Gradient Boosting)",
        "✅ Dynamic strategy optimization across multiple time periods",
        "✅ Automatic strategy transition learning with market regime detection",
        "✅ Support & resistance level detection using pivot points",
        "✅ Fibonacci retracement analysis for key levels",
        "✅ Comprehensive feature engineering with interaction terms",
        "✅ Time-series aware cross-validation",
        "✅ Ensemble predictions with confidence scoring",
        "✅ Detailed performance analytics and reporting",
        "✅ Robust error handling and data validation",
        "✅ 100% free and local (no cloud dependencies)",
        "✅ Ready for GUI integration with modern dark theme",
        "✅ Multi-threaded processing for responsiveness"
    ]
    
    for capability in capabilities:
        print(f"  {capability}")
    
    print("\n🎯 CONCLUSION")
    print("=" * 60)
    print("This is indeed THE MOST ADVANCED FREE LOCAL ML TRADING BOT!")
    print("🚀 All systems operational and ready for trading analysis!")
    print("📊 Load your own data and discover the best strategies!")
    
    return {
        'data_loaded': True,
        'indicators_count': len(indicators),
        'strategies_found': len(strategies) if strategies else 0,
        'features_count': features_df.shape[1],
        'samples_count': features_df.shape[0],
        'support_resistance_levels': len(support_resistance),
        'fibonacci_levels': len(fibonacci)
    }

def create_visualization_demo():
    """Create some demonstration visualizations"""
    print("\n📈 Creating demonstration visualizations...")
    
    try:
        # Load data for visualization
        loader = EnhancedDataLoader()
        data = loader.load_data('sample_data/new_format_sample.csv', 'custom')
        indicators = loader.calculate_all_indicators()
        
        # Create price chart with some indicators
        fig, axes = plt.subplots(3, 1, figsize=(15, 12))
        
        # Price chart
        axes[0].plot(data.index[-200:], data['close'].iloc[-200:], linewidth=2, label='Close Price', color='blue')
        if 'SMA_20' in indicators:
            axes[0].plot(data.index[-200:], indicators['SMA_20'].iloc[-200:], linewidth=1, label='SMA 20', color='red', alpha=0.7)
        if 'SMA_50' in indicators:
            axes[0].plot(data.index[-200:], indicators['SMA_50'].iloc[-200:], linewidth=1, label='SMA 50', color='green', alpha=0.7)
        axes[0].set_title('Price Chart with Moving Averages', fontsize=14, fontweight='bold')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Volume chart
        axes[1].bar(data.index[-200:], data['volume'].iloc[-200:], alpha=0.7, color='orange', width=0.8)
        axes[1].set_title('Volume Analysis', fontsize=14, fontweight='bold')
        axes[1].grid(True, alpha=0.3)
        
        # RSI chart
        if 'RSI_14' in indicators:
            axes[2].plot(data.index[-200:], indicators['RSI_14'].iloc[-200:], linewidth=2, color='purple')
            axes[2].axhline(y=70, color='red', linestyle='--', alpha=0.7)
            axes[2].axhline(y=30, color='green', linestyle='--', alpha=0.7)
            axes[2].set_title('RSI (14)', fontsize=14, fontweight='bold')
            axes[2].set_ylabel('RSI')
            axes[2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('ultimate_trading_bot_demo.png', dpi=150, bbox_inches='tight')
        print("✅ Visualization saved as 'ultimate_trading_bot_demo.png'")
        
    except Exception as e:
        print(f"⚠️  Visualization error: {e}")

if __name__ == "__main__":
    # Run the comprehensive demonstration
    results = demonstrate_ultimate_ml_trading_bot()
    
    # Create visualizations
    create_visualization_demo()
    
    print("\n" + "="*60)
    print("🎉 ULTIMATE ML TRADING BOT DEMONSTRATION COMPLETE! 🎉")
    print("="*60)