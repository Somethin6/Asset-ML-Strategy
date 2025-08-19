#!/usr/bin/env python3
"""
Asset ML Strategy Launcher
Simple launcher script that handles different environments
"""

import sys
import os

def check_requirements():
    """Check if core requirements are installed"""
    core_packages = [
        ('pandas', 'pandas'),
        ('numpy', 'numpy'),
        ('sklearn', 'scikit-learn'),
        ('matplotlib', 'matplotlib'),
        ('openpyxl', 'openpyxl (Excel support)')
    ]
    
    optional_packages = [
        ('seaborn', 'seaborn (enhanced visualizations)'),
        ('tkinter', 'tkinter (Desktop GUI)')
    ]
    
    missing_core = []
    missing_optional = []
    
    print("📦 Checking core requirements...")
    for package, display_name in core_packages:
        try:
            if package == 'sklearn':
                import sklearn
            else:
                __import__(package)
            print(f"✓ {display_name}")
        except ImportError:
            print(f"✗ {display_name}")
            missing_core.append(display_name)
    
    print("\n📦 Checking optional components...")
    for package, display_name in optional_packages:
        try:
            __import__(package)
            print(f"✓ {display_name}")
        except ImportError:
            print(f"⚠ {display_name} (optional)")
            missing_optional.append(display_name)
    
    if missing_core:
        print(f"\n❌ Missing REQUIRED packages: {', '.join(missing_core)}")
        print("Please install them with: pip install pandas numpy scikit-learn matplotlib openpyxl")
        return False
    
    if missing_optional:
        print(f"\n💡 Optional components not available: {', '.join(missing_optional)}")
        print("   This is fine - the application will use alternative interfaces.")
    
    print("✅ All core requirements satisfied!")
    return True

def check_gui_support():
    """Check if GUI (tkinter) is available"""
    try:
        import tkinter as tk
        # Try to create a test window
        root = tk.Tk()
        root.withdraw()  # Hide the window
        root.destroy()
        return True
    except Exception:
        return False

def check_display_environment():
    """Check if we're in a display environment"""
    return bool(os.environ.get('DISPLAY') or os.environ.get('WAYLAND_DISPLAY') or sys.platform == 'win32')

def run_core_test():
    """Run core functionality test"""
    print("\n🧪 Running core functionality test...")
    import subprocess
    result = subprocess.run([sys.executable, "test_functionality.py"])
    if result.returncode == 0:
        print("\n✅ Core functionality test completed successfully!")
    else:
        print("\n❌ Core functionality test had issues")
    input("\nPress Enter to continue...")

def offer_interface_options():
    """Offer different interface options to the user"""
    print("\n" + "=" * 50)
    print("🚀 ASSET ML STRATEGY - INTERFACE OPTIONS")
    print("=" * 50)
    print("\nAvailable interfaces:")
    print("1. 🌐 Web Interface (Streamlit) - Recommended for analysis")
    print("2. 🔌 API Server (FastAPI) - For developers and integrations")
    print("3. 🧪 Test Core Functionality - Verify everything works")
    print("4. 📋 System Information - Check dependencies and troubleshoot")
    print("5. 📊 Quick Demo - Run a simple ML demonstration")
    print("6. 🚪 Exit")
    
    while True:
        try:
            choice = input("\n👉 Select an option (1-6): ").strip()
            
            if choice == '1':
                # Launch web interface
                launch_web_interface()
                break
                
            elif choice == '2':
                # Launch API server
                launch_api_server()
                break
                
            elif choice == '3':
                # Run functionality test
                run_core_test()
                
            elif choice == '4':
                # Show system info
                show_system_info()
                
            elif choice == '5':
                # Quick demo
                run_quick_demo()
                
            elif choice == '6':
                print("👋 Goodbye!")
                sys.exit(0)
                
            else:
                print("❌ Invalid choice. Please enter 1-6.")
                
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            sys.exit(0)

def launch_web_interface():
    """Launch web interface"""
    print("\n🌐 Launching Web Interface...")
    
    # First, try to install streamlit if needed
    try:
        import streamlit
        print("✅ Streamlit already available")
    except ImportError:
        print("📥 Installing Streamlit...")
        import subprocess
        result = subprocess.run([sys.executable, '-m', 'pip', 'install', '--user', 'streamlit>=1.25.0'])
        if result.returncode != 0:
            print("❌ Failed to install Streamlit")
            return
    
    # Look for available web applications
    web_apps = [
        ('dashboard.py', 'Main Dashboard'),
        ('advanced_gui.py', 'Advanced GUI'),
        ('simple_web_gui.py', 'Simple Web GUI'),
        ('web_launcher.py', 'Web Launcher'),
        ('main_launcher.py', 'Main Launcher')
    ]
    
    for app_file, app_name in web_apps:
        if os.path.exists(app_file):
            print(f"🚀 Starting {app_name}...")
            print("🌐 The web interface will be available at http://localhost:8501")
            print("🌍 Or from another machine at: http://YOUR_SERVER_IP:8501")
            print("\n" + "="*50)
            print("📝 Application is starting...")
            print("📝 Press Ctrl+C to stop the application")
            print("="*50 + "\n")
            import subprocess
            try:
                subprocess.run([
                    'streamlit', 'run', app_file,
                    '--server.address', '0.0.0.0',
                    '--server.port', '8501',
                    '--server.headless', 'true',
                    '--browser.gatherUsageStats', 'false'
                ])
            except FileNotFoundError:
                print("❌ Streamlit command not found. Please ensure it's installed correctly.")
            except KeyboardInterrupt:
                print("\n👋 Web interface stopped by user")
            return
    
    print("❌ No web applications found")
    print("💡 You can create a simple web interface or use the API server instead")

def run_quick_demo():
    """Run a quick ML demonstration"""
    print("\n📊 Running Quick ML Demo...")
    
    # Test different demo options in order of preference
    demo_options = [
        ('test_functionality.py', 'Core Functionality Test', True),
        ('enhanced_ml_test.py', 'Enhanced ML Test', False),
        ('ultimate_demo.py', 'Ultimate Demo', False),
        ('quick_demo.py', 'Quick Demo', False),
    ]
    
    print("Available demo options:")
    for i, (script, name, is_core) in enumerate(demo_options):
        if os.path.exists(script):
            status = "✅ CORE" if is_core else "✅"
            print(f"  {i+1}. {status} {name}")
        else:
            print(f"  {i+1}. ❌ {name} (not found)")
    
    # Try to run the first available demo
    for script, name, is_core in demo_options:
        if os.path.exists(script):
            print(f"\n🎯 Running {name}...")
            try:
                import subprocess
                result = subprocess.run([sys.executable, script], timeout=120)
                if result.returncode == 0:
                    print(f"✅ {name} completed successfully!")
                else:
                    print(f"⚠️ {name} completed with return code {result.returncode}")
            except subprocess.TimeoutExpired:
                print(f"⏰ {name} timed out (>120s)")
            except Exception as e:
                print(f"❌ {name} failed: {e}")
                continue
            
            print("\nDemo completed. Would you like to:")
            print("1. Run another demo")
            print("2. Return to main menu")
            
            try:
                choice = input("Choose (1-2): ").strip()
                if choice == '1':
                    continue
                else:
                    return
            except (EOFError, KeyboardInterrupt):
                return
            
            return
    
    # If no demo scripts found, create a simple inline demo
    print("🎯 No demo scripts found - running inline ML demo...")
    try:
        run_inline_demo()
    except Exception as e:
        print(f"❌ Demo failed: {e}")

def run_inline_demo():
    """Run a simple inline demo"""
    import pandas as pd
    import numpy as np
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import r2_score
    
    print("📊 Generating sample data...")
    # Generate sample data
    dates = pd.date_range('2020-01-01', periods=100)
    np.random.seed(42)
    
    base_price = 100
    prices = [base_price]
    
    # Generate realistic price movements
    for i in range(1, 100):
        change = np.random.normal(0.001, 0.02)  # Small daily changes
        new_price = prices[-1] * (1 + change)
        prices.append(max(new_price, 10))  # Ensure positive prices
    
    data = {
        'Date': dates,
        'Price': prices,
        'Volume': np.random.randint(50000, 200000, 100),
        'MA_5': pd.Series(prices).rolling(5).mean(),
        'Volatility': pd.Series(prices).pct_change().rolling(5).std() * np.sqrt(252)
    }
    df = pd.DataFrame(data).dropna()
    
    print(f"✅ Generated {len(df)} samples")
    print(f"   📈 Price range: ${df['Price'].min():.2f} - ${df['Price'].max():.2f}")
    
    # Prepare ML model
    X = df[['Volume', 'MA_5', 'Volatility']]
    y = df['Price']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    print("🤖 Training Random Forest model...")
    # Train model
    model = RandomForestRegressor(n_estimators=50, random_state=42)
    model.fit(X_train, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    
    print(f"✅ Demo Results:")
    print(f"   📊 Training samples: {len(X_train)}")
    print(f"   📊 Testing samples: {len(X_test)}")
    print(f"   🎯 R² Score: {r2:.4f}")
    print(f"   📉 Mean Absolute Error: ${abs(y_test - y_pred).mean():.2f}")
    
    # Feature importance
    importance = pd.DataFrame({
        'feature': X.columns,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print(f"   🔝 Most important feature: {importance.iloc[0]['feature']} ({importance.iloc[0]['importance']:.3f})")
    print(f"   🎉 Inline demo completed successfully!")

def launch_api_server():
    """Launch API server"""
    print("\n🔌 Looking for API server...")
    
    api_files = [
        ('api/main.py', 'Main API Server'),
        ('api_dev.py', 'Development API Server')
    ]
    
    for api_file, name in api_files:
        if os.path.exists(api_file):
            print(f"🚀 Starting {name}: {api_file}")
            print("🔌 The API will be available at http://localhost:8000")
            print("📚 API documentation at http://localhost:8000/docs")
            
            # Try to install uvicorn if needed
            try:
                import uvicorn
            except ImportError:
                print("📥 Installing uvicorn...")
                import subprocess
                subprocess.run([sys.executable, '-m', 'pip', 'install', '--user', 'uvicorn[standard]'])
            
            import subprocess
            try:
                module_path = api_file.replace('/', '.').replace('.py', '') + ':app'
                subprocess.run([
                    'uvicorn', module_path,
                    '--host', '0.0.0.0',
                    '--port', '8000',
                    '--reload'
                ])
            except KeyboardInterrupt:
                print("\n👋 API server stopped by user")
            return
    
    print("❌ No API server found")
    print("💡 You can create a simple API or use the web interface instead")

def show_system_info():
    """Show system information"""
    print("\n" + "=" * 50)
    print("🔍 SYSTEM INFORMATION")
    print("=" * 50)
    print(f"🐍 Python: {sys.version}")
    print(f"💻 Platform: {sys.platform}")
    print(f"🖥️  Display Environment: {'✅ Available' if check_display_environment() else '❌ Headless'}")
    print(f"🎨 Desktop GUI Support: {'✅ Available' if check_gui_support() else '❌ Not Available'}")
    
    print(f"\n📦 Package Status:")
    packages_to_check = [
        ('pandas', 'Data processing'),
        ('numpy', 'Numerical computing'),
        ('sklearn', 'Machine learning'),
        ('matplotlib', 'Plotting'),
        ('seaborn', 'Statistical visualization'),
        ('openpyxl', 'Excel file support'),
        ('streamlit', 'Web interface'),
        ('fastapi', 'API server'),
        ('uvicorn', 'Web server')
    ]
    
    for package, description in packages_to_check:
        try:
            if package == 'sklearn':
                import sklearn
            else:
                __import__(package)
            print(f"   ✅ {package} - {description}")
        except ImportError:
            print(f"   ❌ {package} - {description} (not installed)")
    
    print(f"\n📁 Available Application Files:")
    files = [
        ('asset_ml_strategy.py', 'Desktop GUI application'),
        ('test_functionality.py', 'Core functionality test'),
        ('dashboard.py', 'Web dashboard'),
        ('advanced_gui.py', 'Advanced web GUI'), 
        ('simple_web_gui.py', 'Simple web GUI'),
        ('web_launcher.py', 'Web launcher'),
        ('api_dev.py', 'API development server'),
        ('quick_demo.py', 'Quick demonstration'),
        ('ultimate_demo.py', 'Ultimate demonstration')
    ]
    
    for file, description in files:
        status = "✅" if os.path.exists(file) else "❌"
        print(f"   {status} {file} - {description}")
    
    print(f"\n📊 Sample Data:")
    data_files = [
        'sample_data/AAPL_sample.xlsx',
        'sample_data/AAPL_sample.csv'
    ]
    
    for file in data_files:
        if os.path.exists(file):
            try:
                import pandas as pd
                if file.endswith('.xlsx'):
                    df = pd.read_excel(file)
                else:
                    df = pd.read_csv(file)
                print(f"   ✅ {file} ({len(df)} rows, {len(df.columns)} columns)")
            except Exception as e:
                print(f"   ⚠️  {file} (exists but couldn't read: {e})")
        else:
            print(f"   ❌ {file}")
    
    input("\nPress Enter to continue...")

def main():
    """Main launcher"""
    print("🚀 Asset ML Strategy - Smart Launcher")
    print("=" * 50)
    
    # Check requirements
    if not check_requirements():
        print("\n💡 TIP: Install missing packages with:")
        print("   pip install pandas numpy scikit-learn matplotlib openpyxl")
        sys.exit(1)
    
    # Determine environment and interface options
    has_display = check_display_environment()
    has_gui = check_gui_support()
    
    print(f"\n🔍 Environment Detection:")
    print(f"   Display Environment: {'✅ Available' if has_display else '❌ Headless'}")
    print(f"   Desktop GUI Support: {'✅ Available' if has_gui else '❌ Not Available'}")
    
    # If we have full GUI support and display, offer desktop GUI as primary option
    if has_display and has_gui:
        print("\n🎯 Full GUI environment detected!")
        print("1. 🖥️  Launch Desktop GUI (recommended)")
        print("2. 🌐 Use Web Interface instead")
        print("3. 🧪 Test Core Functionality")
        
        choice = input("\nChoose interface (1-3): ").strip()
        if choice == '1':
            try:
                print("🚀 Launching Desktop GUI...")
                from asset_ml_strategy import main as app_main
                app_main()
            except Exception as e:
                print(f"❌ Error launching Desktop GUI: {e}")
                print("🔄 Falling back to web interface...")
                offer_interface_options()
        elif choice == '3':
            run_core_test()
        else:
            offer_interface_options()
    else:
        # Headless or no GUI support - this is fine, use alternatives
        if not has_display:
            print("\n🌐 Headless environment detected - perfect for web interfaces!")
        elif not has_gui:
            print("\n🌐 Desktop GUI unavailable - using web interfaces instead")
        
        offer_interface_options()

if __name__ == "__main__":
    main()