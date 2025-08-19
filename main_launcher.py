#!/usr/bin/env python3
"""
MASTER LAUNCHER - Asset ML Strategy
Perfect unified interface for all ML models and features
"""

import os
import sys
import subprocess
import argparse
from pathlib import Path

def print_banner():
    """Print the application banner"""
    print("""
🚀 ASSET ML STRATEGY - MASTER LAUNCHER 🚀
═══════════════════════════════════════════
  Perfect Free Local ML Trading System
═══════════════════════════════════════════

📊 6+ ADVANCED ML MODELS AVAILABLE:
  1. Random Forest Regressor
  2. Transformer Neural Networks  
  3. LSTM with Attention
  4. CNN-LSTM Hybrid
  5. WaveNet Architecture
  6. XGBoost & LightGBM
  7. Quantum-Inspired Networks
  8. Ensemble Methods

🎯 FEATURES AVAILABLE:
  ✓ Technical Indicators (SMA, EMA, RSI, MACD)
  ✓ Deep Learning Architectures
  ✓ Reinforcement Learning Agents
  ✓ Portfolio Optimization
  ✓ Risk Management
  ✓ Backtesting Engine
  ✓ Sentiment Analysis
  ✓ Real-time Data Processing
  ✓ GUI Interfaces (Desktop & Web)

🆓 100% FREE & LOCAL - No subscriptions!
""")

def check_dependencies():
    """Check and install core dependencies"""
    print("🔍 Checking dependencies...")
    
    core_packages = ['pandas', 'numpy', 'scikit-learn', 'matplotlib']
    missing = []
    
    for package in core_packages:
        try:
            if package == 'scikit-learn':
                import sklearn
            else:
                __import__(package)
        except ImportError:
            missing.append(package)
    
    if missing:
        print(f"⚠️  Missing core packages: {missing}")
        print("📦 Installing core dependencies...")
        try:
            subprocess.check_call([sys.executable, '-m', 'pip', 'install'] + missing)
            print("✅ Core dependencies installed!")
        except subprocess.CalledProcessError:
            print("❌ Failed to install dependencies. Please run: pip install pandas numpy scikit-learn matplotlib")
            return False
    else:
        print("✅ All core dependencies available!")
    
    return True

def show_menu():
    """Show the main menu"""
    print("""
🎮 SELECT LAUNCH MODE:

BASIC INTERFACES:
  1. Simple GUI (Beginner-Friendly)
  2. Test Functionality (Verify System)
  3. Basic Command Line

ADVANCED INTERFACES:  
  4. Ultimate Demo (All Features)
  5. MoneyPrinter Strategy (Advanced)
  6. Advanced Web GUI (Streamlit)
  7. API Server (REST Endpoints)

SPECIALIZED TOOLS:
  8. Generate Sample Data
  9. Real Data Fetcher
  10. Risk Analysis Tool
  11. Portfolio Optimizer

DEVELOPMENT:
  12. Run Tests
  13. Install All Dependencies
  14. System Diagnostics

  0. Exit

Choice: """)

def launch_simple_gui():
    """Launch the simple GUI"""
    print("🚀 Launching Simple GUI...")
    try:
        from asset_ml_strategy import main
        main()
    except ImportError as e:
        print(f"❌ Failed to import GUI: {e}")
        return False
    except Exception as e:
        print(f"❌ GUI launch failed: {e}")
        return False
    return True

def launch_test():
    """Launch functionality test"""
    print("🧪 Running functionality test...")
    try:
        subprocess.run([sys.executable, 'test_functionality.py'])
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False
    return True

def launch_ultimate_demo():
    """Launch ultimate demo"""
    print("🌟 Launching Ultimate Demo...")
    if os.path.exists('ultimate_demo.py'):
        try:
            subprocess.Popen([sys.executable, 'ultimate_demo.py'])
            print("✅ Ultimate Demo launched!")
        except Exception as e:
            print(f"❌ Launch failed: {e}")
    else:
        print("❌ ultimate_demo.py not found")

def launch_moneyprinter():
    """Launch MoneyPrinter strategy"""
    print("💰 Launching MoneyPrinter Strategy...")
    if os.path.exists('moneyprinter.py'):
        try:
            subprocess.Popen([sys.executable, 'moneyprinter.py'])
            print("✅ MoneyPrinter launched!")
        except Exception as e:
            print(f"❌ Launch failed: {e}")
    else:
        print("❌ moneyprinter.py not found")

def launch_web_gui():
    """Launch advanced web GUI"""
    print("🌐 Launching Advanced Web GUI...")
    if os.path.exists('advanced_gui.py'):
        try:
            # Install streamlit if needed
            try:
                import streamlit
            except ImportError:
                print("📦 Installing Streamlit...")
                subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'streamlit'])
            
            subprocess.Popen([sys.executable, '-m', 'streamlit', 'run', 'advanced_gui.py'])
            print("✅ Web GUI will open in your browser!")
        except Exception as e:
            print(f"❌ Launch failed: {e}")
    else:
        print("❌ advanced_gui.py not found")

def launch_api_server():
    """Launch API server"""
    print("🔌 Launching API Server...")
    if os.path.exists('api_dev.py'):
        try:
            subprocess.Popen([sys.executable, 'api_dev.py'])
            print("✅ API Server launched on http://localhost:8000")
        except Exception as e:
            print(f"❌ Launch failed: {e}")
    else:
        print("❌ API server not found")

def generate_sample_data():
    """Generate sample data"""
    print("📊 Generating sample data...")
    if os.path.exists('generate_synthetic_data.py'):
        try:
            subprocess.run([sys.executable, 'generate_synthetic_data.py'])
        except Exception as e:
            print(f"❌ Generation failed: {e}")
    else:
        print("❌ Sample data generator not found")

def fetch_real_data():
    """Fetch real market data"""
    print("📈 Fetching real market data...")
    if os.path.exists('real_data_fetcher.py'):
        try:
            subprocess.run([sys.executable, 'real_data_fetcher.py'])
        except Exception as e:
            print(f"❌ Data fetch failed: {e}")
    else:
        print("❌ Real data fetcher not found")

def run_risk_analysis():
    """Run risk analysis tool"""
    print("⚠️  Running risk analysis...")
    if os.path.exists('risk_management.py'):
        try:
            exec(open('risk_management.py').read())
        except Exception as e:
            print(f"❌ Risk analysis failed: {e}")
    else:
        print("❌ Risk management tool not found")

def run_portfolio_optimizer():
    """Run portfolio optimizer"""
    print("📈 Running portfolio optimizer...")
    portfolio_path = 'src/portfolio_optimization.py'
    if os.path.exists(portfolio_path):
        try:
            subprocess.run([sys.executable, portfolio_path])
        except Exception as e:
            print(f"❌ Portfolio optimization failed: {e}")
    else:
        print("❌ Portfolio optimizer not found")

def run_tests():
    """Run all tests"""
    print("🧪 Running comprehensive tests...")
    test_files = ['test_functionality.py', 'tests/test_strategy.py', 'tests/test_api.py']
    
    for test_file in test_files:
        if os.path.exists(test_file):
            print(f"Running {test_file}...")
            try:
                subprocess.run([sys.executable, test_file])
            except Exception as e:
                print(f"❌ {test_file} failed: {e}")
        else:
            print(f"⚠️  {test_file} not found")

def install_all_dependencies():
    """Install all dependencies"""
    print("📦 Installing all dependencies...")
    
    requirements_files = ['requirements.txt', 'requirements-free.txt']
    
    for req_file in requirements_files:
        if os.path.exists(req_file):
            print(f"Installing from {req_file}...")
            try:
                subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-r', req_file])
                print(f"✅ {req_file} installed!")
                break
            except subprocess.CalledProcessError:
                print(f"⚠️  Failed to install from {req_file}")
        else:
            print(f"⚠️  {req_file} not found")
    
    # Install additional packages individually
    additional_packages = [
        'streamlit', 'plotly', 'yfinance', 'ta', 'openpyxl',
        'xgboost', 'lightgbm', 'optuna'
    ]
    
    print("Installing additional packages...")
    for package in additional_packages:
        try:
            subprocess.check_call([sys.executable, '-m', 'pip', 'install', package])
            print(f"✅ {package} installed!")
        except subprocess.CalledProcessError:
            print(f"⚠️  Failed to install {package}")

def system_diagnostics():
    """Run system diagnostics"""
    print("🔍 Running system diagnostics...")
    
    print("\n📋 SYSTEM INFO:")
    print(f"Python version: {sys.version}")
    print(f"Working directory: {os.getcwd()}")
    print(f"Platform: {sys.platform}")
    
    print("\n📦 CORE PACKAGES:")
    core_packages = ['pandas', 'numpy', 'scikit-learn', 'matplotlib', 'seaborn']
    for package in core_packages:
        try:
            if package == 'scikit-learn':
                import sklearn
                version = sklearn.__version__
            else:
                mod = __import__(package)
                version = getattr(mod, '__version__', 'Unknown')
            print(f"✅ {package}: {version}")
        except ImportError:
            print(f"❌ {package}: Not installed")
    
    print("\n🔧 ADVANCED PACKAGES:")
    advanced_packages = ['torch', 'streamlit', 'plotly', 'xgboost', 'lightgbm', 'optuna']
    for package in advanced_packages:
        try:
            mod = __import__(package)
            version = getattr(mod, '__version__', 'Unknown')
            print(f"✅ {package}: {version}")
        except ImportError:
            print(f"❌ {package}: Not installed")
    
    print("\n📁 AVAILABLE FILES:")
    key_files = [
        'asset_ml_strategy.py', 'run.py', 'test_functionality.py',
        'ultimate_demo.py', 'moneyprinter.py', 'advanced_gui.py',
        'requirements.txt', 'README.md'
    ]
    
    for file in key_files:
        if os.path.exists(file):
            size = os.path.getsize(file)
            print(f"✅ {file} ({size} bytes)")
        else:
            print(f"❌ {file}: Missing")
    
    print("\n📂 DIRECTORY STRUCTURE:")
    for item in sorted(os.listdir('.')):
        if os.path.isdir(item):
            print(f"📁 {item}/")
    
    print("\nDiagnostics complete!")

def basic_cli():
    """Basic command line interface"""
    print("💻 Basic CLI Mode")
    print("Loading core functionality...")
    
    try:
        # Test basic functionality
        print("Running basic test...")
        import pandas as pd
        import numpy as np
        from sklearn.ensemble import RandomForestRegressor
        
        # Generate test data
        np.random.seed(42)
        X = np.random.randn(100, 5)
        y = np.random.randn(100)
        
        # Train model
        model = RandomForestRegressor(n_estimators=10, random_state=42)
        model.fit(X, y)
        
        score = model.score(X, y)
        print(f"✅ Basic ML test passed - R² Score: {score:.4f}")
        
        print("\n📊 For full features, use the GUI modes!")
        print("💡 Tip: Try option 1 (Simple GUI) or 4 (Ultimate Demo)")
        
    except Exception as e:
        print(f"❌ CLI test failed: {e}")

def main():
    """Main launcher function"""
    print_banner()
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Asset ML Strategy Master Launcher')
    parser.add_argument('--mode', '-m', type=int, help='Launch mode (1-14)')
    parser.add_argument('--install', action='store_true', help='Install all dependencies')
    parser.add_argument('--test', action='store_true', help='Run functionality test')
    parser.add_argument('--diagnostics', action='store_true', help='Run system diagnostics')
    
    args = parser.parse_args()
    
    # Handle command line arguments
    if args.install:
        install_all_dependencies()
        return
    
    if args.test:
        launch_test()
        return
    
    if args.diagnostics:
        system_diagnostics()
        return
    
    if args.mode:
        choice = str(args.mode)
    else:
        # Check dependencies first
        if not check_dependencies():
            print("❌ Cannot continue without core dependencies")
            return
        
        # Interactive menu
        show_menu()
        choice = input().strip()
    
    # Handle menu choices
    if choice == '1':
        launch_simple_gui()
    elif choice == '2':
        launch_test()
    elif choice == '3':
        basic_cli()
    elif choice == '4':
        launch_ultimate_demo()
    elif choice == '5':
        launch_moneyprinter()
    elif choice == '6':
        launch_web_gui()
    elif choice == '7':
        launch_api_server()
    elif choice == '8':
        generate_sample_data()
    elif choice == '9':
        fetch_real_data()
    elif choice == '10':
        run_risk_analysis()
    elif choice == '11':
        run_portfolio_optimizer()
    elif choice == '12':
        run_tests()
    elif choice == '13':
        install_all_dependencies()
    elif choice == '14':
        system_diagnostics()
    elif choice == '0':
        print("👋 Goodbye!")
        return
    else:
        print("❌ Invalid choice. Please select 1-14 or 0 to exit.")

if __name__ == "__main__":
    main()