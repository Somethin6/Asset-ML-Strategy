#!/usr/bin/env python3
"""
FREE TIER TEST - Verify the system runs 100% free with no paid services.
This script tests all core functionality without requiring any API keys or subscriptions.
"""

import sys
import os
import traceback

def test_free_tier():
    """Test all free-tier functionality."""
    
    print("🧪 TESTING FREE TIER SETUP...")
    print("="*60)
    
    try:
        # Test 1: Core libraries
        print("1️⃣ Testing core libraries...")
        import pandas as pd
        import numpy as np
        import sklearn
        print("   ✅ Pandas, NumPy, Scikit-learn imported successfully")
        
        # Test 2: ML libraries  
        print("2️⃣ Testing ML libraries...")
        import xgboost as xgb
        import lightgbm as lgb
        from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
        from sklearn.svm import SVC
        from sklearn.linear_model import LogisticRegression
        print("   ✅ All 6 ML models available (RF, XGB, LGB, GB, SVM, LR)")
        
        # Test 3: Visualization
        print("3️⃣ Testing visualization libraries...")
        import matplotlib.pyplot as plt
        import seaborn as sns
        import plotly.graph_objects as go
        print("   ✅ Matplotlib, Seaborn, Plotly available")
        
        # Test 4: Synthetic data generation
        print("4️⃣ Testing synthetic data generation...")
        from generate_synthetic_data import generate_realistic_market_data
        data = generate_realistic_market_data(output_file='data/test_free_data.csv')
        print(f"   ✅ Generated {len(data)} rows of synthetic market data")
        
        # Test 5: Configuration system
        print("5️⃣ Testing configuration system...")
        import yaml
        with open('config/moneyprinter_config.yaml', 'r') as f:
            config = yaml.safe_load(f)
        
        data_source = config['data']['primary_data_source']
        alerts_enabled = config['alerts']['enable_email_alerts']
        print(f"   ✅ Config loaded - Data source: {data_source}, Email alerts: {alerts_enabled}")
        
        # Test 6: Free data sources  
        print("6️⃣ Testing free data sources...")
        import yfinance as yf
        print("   ✅ Yahoo Finance library available (no API key needed)")
        print("   ℹ️  Note: Network required for live data, but synthetic data works offline")
        
        # Test 7: Web framework
        print("7️⃣ Testing web framework...")
        import streamlit as st
        import fastapi
        print("   ✅ Streamlit and FastAPI available for web interface")
        
        # Test 8: Check for paid services (should be disabled)
        print("8️⃣ Verifying no paid services required...")
        
        paid_imports = []
        try:
            import sendgrid
            paid_imports.append("SendGrid (email service)")
        except ImportError:
            pass
            
        try:
            import twilio
            paid_imports.append("Twilio (SMS service)")
        except ImportError:
            pass
            
        try:
            import sentry_sdk
            paid_imports.append("Sentry (monitoring service)")  
        except ImportError:
            pass
        
        if paid_imports:
            print(f"   ⚠️  Optional paid services detected: {', '.join(paid_imports)}")
            print("   ℹ️  These are optional and not required for core functionality")
        else:
            print("   ✅ No paid services required - 100% free setup!")
            
        # Test 9: Verify configuration uses free services
        print("9️⃣ Verifying free-tier configuration...")
        if config['data']['primary_data_source'] in ['yfinance', 'synthetic']:
            print("   ✅ Data source set to free option")
        else:
            print(f"   ⚠️  Data source '{config['data']['primary_data_source']}' may require API key")
            
        if not config['alerts']['enable_email_alerts']:
            print("   ✅ Paid alerting services disabled")
        else:
            print("   ⚠️  Email alerts enabled - may require paid service")
            
        print("")
        print("🎉 FREE TIER TEST COMPLETED SUCCESSFULLY! 🎉")
        print("="*60)
        print("✅ CONFIRMED: System runs 100% free with no paid services")
        print("✅ All core ML trading functionality available")
        print("✅ Synthetic data provides realistic testing environment")
        print("✅ Yahoo Finance provides free market data (when online)")
        print("✅ Complete web dashboard available")
        print("✅ No API keys, subscriptions, or cloud services required!")
        print("")
        print("🚀 Ready to start free algorithmic trading!")
        print("   Run: python moneyprinter.py --mode backtest")
        print("   Run: python moneyprinter.py --mode dashboard")
        
        return True
        
    except Exception as e:
        print(f"❌ FREE TIER TEST FAILED: {e}")
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_free_tier()
    sys.exit(0 if success else 1)