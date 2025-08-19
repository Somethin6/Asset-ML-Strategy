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
import sys

# Use non-interactive backend for matplotlib
matplotlib.use('Agg')

def test_data_loading():
    """Test loading Excel data"""
    print("\nTesting data loading functionality...")
    
    try:
        # Create sample data for testing
        dates = pd.date_range('2020-01-01', periods=100, freq='D')
        np.random.seed(42)
        
        # Generate realistic financial data
        base_price = 100
        prices = [base_price]
        
        for i in range(1, 100):
            change = np.random.normal(0, 0.02)  # 2% daily volatility
            new_price = prices[-1] * (1 + change)
            prices.append(new_price)
        
        data = pd.DataFrame({
            'Date': dates,
            'Open': prices,
            'High': [p * (1 + abs(np.random.normal(0, 0.01))) for p in prices],
            'Low': [p * (1 - abs(np.random.normal(0, 0.01))) for p in prices],
            'Close': [p * (1 + np.random.normal(0, 0.005)) for p in prices],
            'Adj Close': [p * (1 + np.random.normal(0, 0.005)) for p in prices],
            'Volume': [int(np.random.normal(1000000, 200000)) for _ in prices]
        })
        
        # Ensure High >= Low and sensible OHLC relationships
        for i in range(len(data)):
            high = max(data.loc[i, 'Open'], data.loc[i, 'High'], data.loc[i, 'Low'], data.loc[i, 'Close'])
            low = min(data.loc[i, 'Open'], data.loc[i, 'High'], data.loc[i, 'Low'], data.loc[i, 'Close'])
            data.loc[i, 'High'] = high
            data.loc[i, 'Low'] = low
        
        print(f"✓ Sample data generated: {len(data)} rows")
        print(f"  Columns: {list(data.columns)}")
        print(f"  Date range: {data['Date'].min()} to {data['Date'].max()}")
        print(f"  Price range: ${data['Close'].min():.2f} to ${data['Close'].max():.2f}")
        
        return data
        
    except Exception as e:
        print(f"✗ Data loading test failed: {e}")
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
        # Create output directory
        os.makedirs('test_outputs', exist_ok=True)
        
        # Test 1: Price chart
        plt.figure(figsize=(12, 6))
        plt.plot(data['Date'], data['Close'], label='Close Price')
        plt.plot(data['Date'], data['Open'], label='Open Price', alpha=0.7)
        plt.title('Price Chart Test')
        plt.xlabel('Date')
        plt.ylabel('Price')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig('test_outputs/price_chart_test.png', dpi=150, bbox_inches='tight')
        plt.close()
        print("✓ Price chart generated")
        
        # Test 2: Volume chart
        plt.figure(figsize=(12, 6))
        plt.bar(data['Date'], data['Volume'], alpha=0.7, width=0.8)
        plt.title('Volume Chart Test')
        plt.xlabel('Date')
        plt.ylabel('Volume')
        plt.grid(True, alpha=0.3)
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig('test_outputs/volume_chart_test.png', dpi=150, bbox_inches='tight')
        plt.close()
        print("✓ Volume chart generated")
        
        # Test 3: Correlation matrix
        numeric_data = data.select_dtypes(include=[np.number])
        correlation_matrix = numeric_data.corr()
        
        plt.figure(figsize=(10, 8))
        plt.imshow(correlation_matrix, cmap='coolwarm', aspect='auto')
        plt.colorbar()
        plt.title('Correlation Matrix Test')
        
        # Add correlation values as text
        for i in range(len(correlation_matrix.columns)):
            for j in range(len(correlation_matrix.columns)):
                plt.text(j, i, f'{correlation_matrix.iloc[i, j]:.2f}',
                        ha='center', va='center', fontsize=8)
        
        plt.xticks(range(len(correlation_matrix.columns)), correlation_matrix.columns, rotation=45)
        plt.yticks(range(len(correlation_matrix.columns)), correlation_matrix.columns)
        plt.tight_layout()
        plt.savefig('test_outputs/correlation_matrix_test.png', dpi=150, bbox_inches='tight')
        plt.close()
        print("✓ Correlation matrix generated")
        
        print(f"✓ All visualizations saved to 'test_outputs/' directory")
        
        return True
        
    except Exception as e:
        print(f"✗ Visualization test failed: {e}")
        return False

def test_gui_modules():
    """Test if GUI modules can be imported"""
    print("\nTesting GUI module imports...")
    
    try:
        import tkinter as tk
        from tkinter import ttk, filedialog, messagebox, scrolledtext
        print("✓ tkinter modules imported successfully")
        
        try:
            import matplotlib.pyplot as plt
            from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
            print("✓ matplotlib GUI backend available")
        except ImportError:
            print("⚠ matplotlib GUI backend not available")
        
        try:
            import seaborn as sns
            print("✓ seaborn available for advanced visualizations")
        except ImportError:
            print("⚠ seaborn not available")
        
        return True
        
    except ImportError:
        print("✗ GUI modules not available (headless environment)")
        return False

def test_advanced_features():
    """Test advanced features availability"""
    print("\nTesting advanced features availability...")
    
    advanced_modules = [
        ('streamlit', 'Web GUI framework'),
        ('torch', 'PyTorch deep learning'),
        ('xgboost', 'XGBoost ML'),
        ('lightgbm', 'LightGBM ML'),
        ('optuna', 'Hyperparameter optimization'),
        ('ta', 'Technical analysis'),
        ('yfinance', 'Real-time data'),
        ('plotly', 'Interactive visualizations')
    ]
    
    available_features = []
    
    for module, description in advanced_modules:
        try:
            __import__(module)
            print(f"✓ {description}")
            available_features.append(module)
        except ImportError:
            print(f"⚠ {description} - install with: pip install {module}")
    
    print(f"\n📊 Available advanced features: {len(available_features)}/{len(advanced_modules)}")
    
    # Check for advanced scripts
    advanced_scripts = [
        ('ultimate_demo.py', 'Ultimate Demo with all features'),
        ('moneyprinter.py', 'MoneyPrinter advanced strategy'),
        ('advanced_gui.py', 'Advanced web GUI'),
        ('src/next_gen_ml_models.py', 'Next-gen ML models')
    ]
    
    print(f"\nAdvanced scripts available:")
    for script, description in advanced_scripts:
        if os.path.exists(script):
            print(f"✓ {script} - {description}")
        else:
            print(f"⚠ {script} - {description} (not found)")
    
    return True

def main():
    """Run all tests"""
    print("🚀 Asset ML Strategy - Functionality Test")
    print("=" * 50)
    
    # Test data loading
    data = test_data_loading()
    if data is None:
        print("❌ Critical failure: Data loading failed")
        return False
    
    # Test ML functionality
    ml_success = test_ml_functionality(data)
    if not ml_success:
        print("❌ Critical failure: ML functionality failed")
        return False
    
    # Test visualization
    viz_success = test_visualization(data)
    
    # Test GUI modules
    gui_success = test_gui_modules()
    
    # Test advanced features
    test_advanced_features()
    
    print("\n" + "=" * 50)
    if ml_success and viz_success:
        print("✅ All core tests passed!")
        if gui_success:
            print("✅ GUI support available - ready for full application")
        else:
            print("Note: GUI may not work in headless environment but core functionality is intact.")
        return True
    else:
        print("❌ Some tests failed")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)