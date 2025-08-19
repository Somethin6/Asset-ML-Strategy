#!/usr/bin/env python3
"""
Enhanced Asset ML Strategy Launcher
Improved launcher with better error handling and feature detection
"""

import sys
import os
import subprocess
from datetime import datetime

def print_banner():
    """Print an attractive banner"""
    print("\n" + "="*70)
    print("🚀 ASSET ML STRATEGY - ENHANCED LAUNCHER")
    print("="*70)
    print("💰 Professional Machine Learning for Financial Assets")
    print("🔥 Enhanced with XGBoost, LightGBM, Technical Analysis & More!")
    print("="*70 + "\n")

def check_package_availability():
    """Check which packages are available"""
    packages = {
        'pandas': ('pandas', 'Core data processing'),
        'numpy': ('numpy', 'Numerical computations'),
        'sklearn': ('scikit-learn', 'Basic ML algorithms'),
        'matplotlib': ('matplotlib', 'Basic plotting'),
        'seaborn': ('seaborn', 'Statistical plotting'),
        'openpyxl': ('openpyxl', 'Excel file support'),
        'streamlit': ('streamlit', 'Web interface framework'),
        'fastapi': ('fastapi', 'API server framework'),
        'uvicorn': ('uvicorn', 'API server'),
        'plotly': ('plotly', 'Interactive charts'),
        'xgboost': ('xgboost', 'Advanced ML - XGBoost'),
        'lightgbm': ('lightgbm', 'Advanced ML - LightGBM'),
        'yfinance': ('yfinance', 'Real-time market data'),
        'ta': ('ta', 'Technical analysis indicators'),
        'optuna': ('optuna', 'Hyperparameter optimization')
    }
    
    available = {}
    missing = []
    
    print("📦 Checking package availability...")
    
    for package, (pip_name, description) in packages.items():
        try:
            if package == 'sklearn':
                import sklearn
            else:
                __import__(package)
            print(f"  ✅ {description}")
            available[package] = True
        except ImportError:
            print(f"  ❌ {description}")
            available[package] = False
            missing.append(pip_name)
    
    return available, missing

def check_gui_support():
    """Check if GUI (tkinter) is available"""
    try:
        import tkinter as tk
        root = tk.Tk()
        root.withdraw()
        root.destroy()
        return True
    except:
        return False

def check_available_scripts():
    """Check which application scripts are available"""
    scripts = {
        'asset_ml_strategy.py': 'Desktop GUI Application',
        'simple_web_gui.py': 'Simple Web Interface',
        'advanced_gui.py': 'Advanced Web Interface',
        'dashboard.py': 'Interactive Dashboard',
        'api_dev.py': 'API Server',
        'test_functionality.py': 'Core Functionality Test',
        'enhanced_ml_test.py': 'Enhanced ML Capabilities Test',
        'robust_ml_test.py': 'Robust ML Strategy Test',
        'ultimate_demo.py': 'Ultimate Feature Demo',
        'moneyprinter.py': 'MoneyPrinter Strategy'
    }
    
    available_scripts = {}
    for script, description in scripts.items():
        if os.path.exists(script):
            available_scripts[script] = description
    
    return available_scripts

def install_missing_packages(missing_packages):
    """Install missing packages"""
    if not missing_packages:
        return True
    
    print(f"\n📦 Missing packages detected: {', '.join(missing_packages)}")
    print("Would you like to install them? (y/n): ", end="")
    
    try:
        response = input().strip().lower()
        if response in ['y', 'yes']:
            print(f"\n🔄 Installing packages...")
            try:
                subprocess.check_call([
                    sys.executable, '-m', 'pip', 'install', '--user'
                ] + missing_packages)
                print("✅ Packages installed successfully!")
                return True
            except subprocess.CalledProcessError:
                print("❌ Failed to install some packages")
                return False
        else:
            print("📝 Continuing with available packages...")
            return True
    except KeyboardInterrupt:
        print("\n👋 Installation cancelled.")
        return False

def launch_web_interface(script_name):
    """Launch a Streamlit web interface"""
    print(f"\n🌐 Launching {script_name}...")
    print("🔗 Interface will be available at: http://localhost:8501")
    print("🌍 Or from another machine at: http://YOUR_SERVER_IP:8501")
    print("\n" + "="*50)
    print("📝 Starting application...")
    print("📝 Press Ctrl+C to stop")
    print("="*50 + "\n")
    
    try:
        subprocess.run([
            'streamlit', 'run', script_name,
            '--server.address', '0.0.0.0',
            '--server.port', '8501',
            '--server.headless', 'true',
            '--browser.gatherUsageStats', 'false'
        ])
    except KeyboardInterrupt:
        print("\n👋 Application stopped.")
    except FileNotFoundError:
        print("❌ Streamlit not found. Please install with: pip install streamlit")

def launch_api_server():
    """Launch FastAPI server"""
    print(f"\n🔌 Launching API Server...")
    print("🔗 API will be available at: http://localhost:8000")
    print("📖 API docs at: http://localhost:8000/docs")
    
    try:
        subprocess.run([
            'uvicorn', 'api_dev:app',
            '--host', '0.0.0.0',
            '--port', '8000'
        ])
    except KeyboardInterrupt:
        print("\n👋 API server stopped.")
    except FileNotFoundError:
        print("❌ Uvicorn not found. Please install with: pip install uvicorn")

def run_test_script(script_name):
    """Run a test script"""
    print(f"\n🧪 Running {script_name}...")
    print("="*50)
    try:
        result = subprocess.run([sys.executable, script_name])
        print("="*50)
        if result.returncode == 0:
            print("✅ Test completed successfully!")
        else:
            print("⚠️ Test completed with warnings.")
    except Exception as e:
        print(f"❌ Error running test: {e}")

def show_main_menu(available_packages, available_scripts):
    """Show the main menu"""
    print("\n🎯 MAIN MENU - Choose your experience:")
    print("="*50)
    
    # Calculate capability scores
    core_score = sum([available_packages.get(pkg, 0) for pkg in ['pandas', 'numpy', 'sklearn']]) / 3
    web_score = sum([available_packages.get(pkg, 0) for pkg in ['streamlit', 'plotly']]) / 2
    advanced_ml_score = sum([available_packages.get(pkg, 0) for pkg in ['xgboost', 'lightgbm', 'optuna']]) / 3
    
    # Menu options based on available features
    menu_options = []
    
    if available_packages.get('streamlit') and 'simple_web_gui.py' in available_scripts:
        menu_options.append(("1", "🌐 Simple Web Interface", "launch_web", "simple_web_gui.py"))
    
    if available_packages.get('streamlit') and 'advanced_gui.py' in available_scripts:
        menu_options.append(("2", "🚀 Advanced Web Interface", "launch_web", "advanced_gui.py"))
    
    if available_packages.get('fastapi') and 'api_dev.py' in available_scripts:
        menu_options.append(("3", "🔌 API Server", "launch_api", None))
    
    if check_gui_support() and 'asset_ml_strategy.py' in available_scripts:
        menu_options.append(("4", "🖥️ Desktop GUI", "launch_desktop", "asset_ml_strategy.py"))
    
    # Test options
    test_options = []
    if 'test_functionality.py' in available_scripts:
        test_options.append(("5", "🧪 Core Functionality Test", "run_test", "test_functionality.py"))
    
    if 'enhanced_ml_test.py' in available_scripts:
        test_options.append(("6", "🎯 Enhanced ML Test", "run_test", "enhanced_ml_test.py"))
    
    if 'robust_ml_test.py' in available_scripts:
        test_options.append(("7", "🔬 Robust ML Strategy Test", "run_test", "robust_ml_test.py"))
    
    # Advanced demos
    demo_options = []
    if 'ultimate_demo.py' in available_scripts:
        demo_options.append(("8", "🌟 Ultimate Demo", "run_test", "ultimate_demo.py"))
    
    # Print menu sections
    if menu_options:
        print("📱 MAIN INTERFACES:")
        for code, desc, action, param in menu_options:
            print(f"  {code}. {desc}")
    
    if test_options:
        print("\n🧪 TESTING & VALIDATION:")
        for code, desc, action, param in test_options:
            print(f"  {code}. {desc}")
    
    if demo_options:
        print("\n🎭 ADVANCED DEMOS:")
        for code, desc, action, param in demo_options:
            print(f"  {code}. {desc}")
    
    print("\n🔧 UTILITIES:")
    print("  9. 📊 System Information")
    print("  10. 📦 Install Missing Packages")
    print("  0. 🚪 Exit")
    
    # Show capability summary
    print(f"\n📈 CAPABILITY SUMMARY:")
    print(f"  Core ML: {'🟢' if core_score > 0.8 else '🟡' if core_score > 0.5 else '🔴'} {core_score:.0%}")
    print(f"  Web Interface: {'🟢' if web_score > 0.8 else '🟡' if web_score > 0.5 else '🔴'} {web_score:.0%}")
    print(f"  Advanced ML: {'🟢' if advanced_ml_score > 0.8 else '🟡' if advanced_ml_score > 0.5 else '🔴'} {advanced_ml_score:.0%}")
    
    # Combine all options
    all_options = menu_options + test_options + demo_options
    return all_options

def show_system_info(available_packages, available_scripts):
    """Show detailed system information"""
    print("\n" + "="*60)
    print("🔍 SYSTEM INFORMATION")
    print("="*60)
    
    print(f"🐍 Python: {sys.version}")
    print(f"💻 Platform: {sys.platform}")
    print(f"📅 Current Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Package summary
    available_count = sum(available_packages.values())
    total_count = len(available_packages)
    print(f"\n📦 Packages: {available_count}/{total_count} available ({available_count/total_count:.0%})")
    
    # Script summary
    print(f"🎯 Available Scripts: {len(available_scripts)}")
    for script, description in available_scripts.items():
        print(f"  ✅ {script} - {description}")
    
    # Recommendations
    print(f"\n💡 RECOMMENDATIONS:")
    if not available_packages.get('streamlit'):
        print("  📝 Install Streamlit for web interface: pip install streamlit")
    if not available_packages.get('xgboost'):
        print("  🚀 Install XGBoost for advanced ML: pip install xgboost")
    if not available_packages.get('yfinance'):
        print("  💰 Install yfinance for real data: pip install yfinance")
    
    input("\nPress Enter to continue...")

def main():
    """Enhanced main launcher"""
    print_banner()
    
    # Check package availability
    available_packages, missing_packages = check_package_availability()
    
    # Check available scripts
    available_scripts = check_available_scripts()
    
    # Main loop
    while True:
        try:
            all_options = show_main_menu(available_packages, available_scripts)
            
            print(f"\n👉 Select an option (0-10): ", end="")
            choice = input().strip()
            
            # Handle menu selections
            selected_option = None
            for code, desc, action, param in all_options:
                if choice == code:
                    selected_option = (action, param)
                    break
            
            if selected_option:
                action, param = selected_option
                
                if action == "launch_web":
                    launch_web_interface(param)
                elif action == "launch_api":
                    launch_api_server()
                elif action == "launch_desktop":
                    print(f"\n🖥️ Launching Desktop GUI...")
                    try:
                        from asset_ml_strategy import main as app_main
                        app_main()
                    except Exception as e:
                        print(f"❌ Error launching Desktop GUI: {e}")
                elif action == "run_test":
                    run_test_script(param)
            
            elif choice == "9":
                show_system_info(available_packages, available_scripts)
            
            elif choice == "10":
                if install_missing_packages(missing_packages):
                    # Refresh package availability
                    print("\n🔄 Refreshing package availability...")
                    available_packages, missing_packages = check_package_availability()
            
            elif choice == "0":
                print("\n👋 Thank you for using Asset ML Strategy!")
                print("💰 May your investments be profitable!")
                break
            
            else:
                print("❌ Invalid choice. Please select a valid option.")
                
        except KeyboardInterrupt:
            print("\n\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"\n❌ Unexpected error: {e}")
            print("Please try again or contact support.")

if __name__ == "__main__":
    main()