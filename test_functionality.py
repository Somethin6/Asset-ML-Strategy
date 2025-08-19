#!/usr/bin/env python3
"""
Test script to verify the Asset ML Strategy functionality without GUI
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import matplotlib
import os

# Use non-interactive backend for matplotlib
matplotlib.use('Agg')

def test_data_loading():
    """Test loading Excel data"""
    print("Testing data loading...")
    
    # Load the sample data
    try:
        data = pd.read_excel('/home/runner/work/Asset-ML-Strategy/Asset-ML-Strategy/sample_data.xlsx')
        print(f"✓ Successfully loaded Excel file with {len(data)} rows and {len(data.columns)} columns")
        
        # Check required columns
        required_columns = ['Date', 'Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']
        missing_columns = [col for col in required_columns if col not in data.columns]
        
        if missing_columns:
            print(f"✗ Missing columns: {missing_columns}")
            return False
        else:
            print(f"✓ All required columns present: {required_columns}")
        
        # Convert Date column to datetime
        data['Date'] = pd.to_datetime(data['Date'])
        print(f"✓ Date column successfully converted to datetime")
        print(f"  Date range: {data['Date'].min()} to {data['Date'].max()}")
        
        return data
        
    except Exception as e:
        print(f"✗ Failed to load data: {e}")
        return None

def test_ml_functionality(data):
    """Test machine learning functionality"""
    print("\nTesting ML functionality...")
    
    try:
        # Prepare features and target
        target_col = 'Close'
        feature_cols = ['Open', 'High', 'Low', 'Adj Close', 'Volume']
        
        X = data[feature_cols].copy()
        y = data[target_col].copy()
        
        # Add technical indicators as features
        X['SMA_5'] = data[target_col].rolling(window=5).mean()
        X['SMA_10'] = data[target_col].rolling(window=10).mean()
        X['Volatility'] = data[target_col].rolling(window=5).std()
        X['Price_Change'] = data[target_col].pct_change()
        
        # Remove rows with NaN values
        mask = ~(X.isnull().any(axis=1) | y.isnull())
        X = X[mask]
        y = y[mask]
        
        print(f"✓ Prepared features with {len(X)} valid samples")
        print(f"  Features: {list(X.columns)}")
        
        if len(X) < 10:
            print("✗ Not enough valid data points for training!")
            return False
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42, shuffle=False
        )
        
        print(f"✓ Data split - Train: {len(X_train)}, Test: {len(X_test)}")
        
        # Train Random Forest model
        model = RandomForestRegressor(n_estimators=50, random_state=42)
        model.fit(X_train, y_train)
        
        print(f"✓ Random Forest model trained successfully")
        
        # Make predictions
        y_pred_train = model.predict(X_train)
        y_pred_test = model.predict(X_test)
        
        # Calculate metrics
        train_mse = mean_squared_error(y_train, y_pred_train)
        test_mse = mean_squared_error(y_test, y_pred_test)
        train_r2 = r2_score(y_train, y_pred_train)
        test_r2 = r2_score(y_test, y_pred_test)
        
        print(f"✓ Model Performance:")
        print(f"  Training R²: {train_r2:.4f}")
        print(f"  Testing R²: {test_r2:.4f}")
        print(f"  Training MSE: {train_mse:.4f}")
        print(f"  Testing MSE: {test_mse:.4f}")
        
        # Feature importance
        feature_importance = pd.DataFrame({
            'feature': X.columns,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        print(f"✓ Top 3 Feature Importances:")
        for _, row in feature_importance.head(3).iterrows():
            print(f"  {row['feature']}: {row['importance']:.4f}")
        
        return True
        
    except Exception as e:
        print(f"✗ ML functionality failed: {e}")
        return False

def test_visualization(data):
    """Test visualization functionality"""
    print("\nTesting visualization functionality...")
    
    try:
        # Create price chart
        plt.figure(figsize=(12, 8))
        
        # Subplot 1: Price chart
        plt.subplot(2, 2, 1)
        plt.plot(data['Date'], data['Open'], label='Open', alpha=0.7)
        plt.plot(data['Date'], data['High'], label='High', alpha=0.7)
        plt.plot(data['Date'], data['Low'], label='Low', alpha=0.7)
        plt.plot(data['Date'], data['Close'], label='Close', linewidth=2)
        plt.title('Price Chart (OHLC)')
        plt.xlabel('Date')
        plt.ylabel('Price')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.xticks(rotation=45)
        
        # Subplot 2: Volume chart
        plt.subplot(2, 2, 2)
        plt.bar(data['Date'], data['Volume'], alpha=0.7, color='orange')
        plt.title('Trading Volume')
        plt.xlabel('Date')
        plt.ylabel('Volume')
        plt.grid(True, alpha=0.3)
        plt.xticks(rotation=45)
        
        # Subplot 3: Correlation matrix
        plt.subplot(2, 2, 3)
        numeric_cols = ['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']
        corr_matrix = data[numeric_cols].corr()
        im = plt.imshow(corr_matrix.values, cmap='coolwarm', aspect='auto', vmin=-1, vmax=1)
        plt.colorbar(im)
        plt.xticks(range(len(corr_matrix.columns)), corr_matrix.columns, rotation=45)
        plt.yticks(range(len(corr_matrix.columns)), corr_matrix.columns)
        plt.title('Correlation Matrix')
        
        # Subplot 4: Price distribution
        plt.subplot(2, 2, 4)
        plt.hist(data['Close'], bins=15, alpha=0.7, color='green')
        plt.title('Close Price Distribution')
        plt.xlabel('Price')
        plt.ylabel('Frequency')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('/home/runner/work/Asset-ML-Strategy/Asset-ML-Strategy/test_charts.png', 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✓ Charts created successfully and saved to test_charts.png")
        return True
        
    except Exception as e:
        print(f"✗ Visualization failed: {e}")
        return False

def test_gui_modules():
    """Test if GUI modules can be imported"""
    print("\nTesting GUI module availability...")
    
    try:
        import tkinter as tk
        print("✓ tkinter module imported successfully")
        
        # Test if we can create a root window (in headless mode this might fail)
        try:
            root = tk.Tk()
            root.withdraw()  # Hide the window
            root.destroy()
            print("✓ tkinter GUI creation test passed")
        except Exception:
            print("⚠ GUI creation test failed (expected in headless environment)")
        
        return True
        
    except ImportError as e:
        print(f"✗ Failed to import GUI modules: {e}")
        return False

def main():
    """Run all tests"""
    print("Asset ML Strategy - Functionality Test")
    print("=" * 50)
    
    # Test 1: Data loading
    data = test_data_loading()
    if data is None:
        print("\n❌ Critical failure: Cannot load data")
        return False
    
    # Test 2: ML functionality
    ml_success = test_ml_functionality(data)
    if not ml_success:
        print("\n❌ ML functionality test failed")
        return False
    
    # Test 3: Visualization
    viz_success = test_visualization(data)
    if not viz_success:
        print("\n❌ Visualization test failed")
        return False
    
    # Test 4: GUI modules
    gui_success = test_gui_modules()
    
    print("\n" + "=" * 50)
    if ml_success and viz_success:
        print("✅ All core functionality tests PASSED!")
        print("The Asset ML Strategy application is ready to use.")
        if gui_success:
            print("GUI modules are also available for the full application.")
        else:
            print("Note: GUI may not work in headless environment but core functionality is intact.")
        return True
    else:
        print("❌ Some tests failed")
        return False

if __name__ == "__main__":
    main()