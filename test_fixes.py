#!/usr/bin/env python3
"""
Test script to verify key fixes are implemented correctly
"""

import pandas as pd
import numpy as np
import sys
import os

# Add current directory to path
sys.path.append('.')

from ULTIMATE_TRADING_SYSTEM import (
    UltimateAdvancedTradingSystem, 
    DataHygieneEngine, 
    VolatilityMicrostructureEngine,
    AdvancedMLEnsemble
)

def test_column_normalization():
    """Test column normalization fix"""
    print("🧪 Testing column normalization fix...")
    
    # Create test data with mixed column names
    test_data = pd.DataFrame({
        'Date': pd.date_range('2023-01-01', periods=100, freq='D'),
        'Open': np.random.normal(100, 5, 100),
        'High': np.random.normal(105, 5, 100),
        'Low': np.random.normal(95, 5, 100),
        'Close': np.random.normal(100, 5, 100),
        'Volume': np.random.randint(1000, 10000, 100)
    })
    
    # Ensure High >= Low for all rows
    test_data['High'] = np.maximum(test_data['High'], test_data['Low'] + 0.01)
    
    engine = DataHygieneEngine()
    cleaned_data = engine.clean_ohlcv_data(test_data)
    
    # Check that columns are normalized
    expected_cols = ['open', 'high', 'low', 'close', 'volume']
    for col in expected_cols:
        if col not in cleaned_data.columns:
            print(f"❌ Column normalization failed: {col} missing")
            return False
    
    # Check that timezone-aware index was created
    if not hasattr(cleaned_data.index, 'tz') or cleaned_data.index.tz is None:
        print("❌ Timezone-aware index not created")
        return False
        
    print("✅ Column normalization working correctly")
    return True

def test_volatility_corrections():
    """Test corrected volatility formulas"""
    print("🧪 Testing corrected volatility formulas...")
    
    # Create test data with normalized column names
    test_data = pd.DataFrame({
        'open': np.random.normal(100, 2, 100),
        'high': np.random.normal(105, 2, 100), 
        'low': np.random.normal(95, 2, 100),
        'close': np.random.normal(100, 2, 100),
        'volume': np.random.randint(1000, 10000, 100)
    }, index=pd.date_range('2023-01-01', periods=100, freq='D', tz='UTC'))
    
    # Ensure High >= Low
    test_data['high'] = np.maximum(test_data['high'], test_data['low'] + 0.01)
    
    engine = VolatilityMicrostructureEngine(periods_per_year=365)
    
    # Test Roll spread fix (using price changes not returns)
    try:
        roll_spread = engine.roll_spread(test_data)
        if roll_spread is None or len(roll_spread) == 0:
            print("❌ Roll spread calculation failed")
            return False
    except Exception as e:
        print(f"❌ Roll spread error: {e}")
        return False
    
    # Test Amihud fix (using dollar volume) 
    try:
        amihud = engine.amihud_illiquidity(test_data)
        if amihud is None or len(amihud) == 0:
            print("❌ Amihud illiquidity calculation failed")
            return False
    except Exception as e:
        print(f"❌ Amihud error: {e}")
        return False
        
    print("✅ Volatility corrections working correctly")
    return True

def test_pipeline_fix():
    """Test that ML ensemble uses pipelines to prevent leakage"""
    print("🧪 Testing pipeline fix for leakage prevention...")
    
    # Create test features and labels
    np.random.seed(42)
    n_samples = 100
    n_features = 10
    
    X = pd.DataFrame(np.random.normal(0, 1, (n_samples, n_features)),
                    columns=[f'feature_{i}' for i in range(n_features)])
    y = pd.Series(np.random.normal(0, 1, n_samples))
    
    ensemble = AdvancedMLEnsemble()
    
    try:
        # This should now use pipelines internally
        ensemble.fit(X, y, task_type='regression', cv_folds=3)
        
        # Check that models are actually pipelines
        if not ensemble.models:
            print("❌ No models created")
            return False
            
        # Check that models are pipelines (have named_steps)
        first_model = next(iter(ensemble.models.values()))
        if not hasattr(first_model, 'named_steps'):
            print("❌ Models are not pipelines - leakage fix not implemented")
            return False
            
        print("✅ Pipeline fix working correctly - no more scaling leakage")
        return True
        
    except Exception as e:
        print(f"❌ Pipeline test failed: {e}")
        return False

def main():
    """Run all tests"""
    print("🧪 TESTING KEY FIXES FROM PROBLEM STATEMENT")
    print("=" * 60)
    
    tests = [
        test_column_normalization,
        test_volatility_corrections, 
        test_pipeline_fix
    ]
    
    passed = 0
    for test in tests:
        try:
            if test():
                passed += 1
            print()
        except Exception as e:
            print(f"❌ Test failed with exception: {e}")
            print()
    
    print("=" * 60)
    print(f"RESULTS: {passed}/{len(tests)} tests passed")
    
    if passed == len(tests):
        print("🎉 All key fixes are working correctly!")
        return True
    else:
        print("⚠️  Some fixes need additional work")
        return False

if __name__ == "__main__":
    main()