#!/usr/bin/env python3
"""
Asset ML Strategy Launcher
Simple launcher script that handles different environments
"""

import sys
import os

def check_requirements():
    """Check if all requirements are installed"""
    required_packages = [
        ('pandas', 'pandas'),
        ('numpy', 'numpy'),
        ('sklearn', 'scikit-learn'),
        ('matplotlib', 'matplotlib'),
        ('seaborn', 'seaborn'),
        ('tkinter', 'tkinter (GUI)')
    ]
    
    missing_packages = []
    
    for package, display_name in required_packages:
        try:
            if package == 'sklearn':
                import sklearn
            else:
                __import__(package)
            print(f"✓ {display_name}")
        except ImportError:
            print(f"✗ {display_name}")
            missing_packages.append(display_name)
    
    if missing_packages:
        print(f"\n⚠ Missing packages: {', '.join(missing_packages)}")
        print("Please install them with: pip install -r requirements.txt")
        return False
    
    print("✓ All required packages available")
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

def offer_interface_options():
    """Offer different interface options to the user"""
    print("\n" + "=" * 50)
    print("🚀 ASSET ML STRATEGY - INTERFACE OPTIONS")
    print("=" * 50)
    print("\nAvailable interfaces:")
    print("1. 🌐 Web Interface (Streamlit) - Recommended for all environments")
    print("2. 🔌 API Server (FastAPI) - For developers and integrations")
    print("3. 🧪 Test Core Functionality - Verify everything works")
    print("4. 📋 System Information - Check dependencies and troubleshoot")
    print("5. 🚪 Exit")
    
    while True:
        try:
            choice = input("\n👉 Select an option (1-5): ").strip()
            
            if choice == '1':
                # Launch web interface
                print("\n🌐 Launching Web Interface...")
                if os.path.exists('web_launcher.py'):
                    import subprocess
                    result = subprocess.run([sys.executable, "web_launcher.py"])
                    sys.exit(result.returncode)
                else:
                    print("❌ Web launcher not found. Using basic Streamlit launch...")
                    launch_basic_web()
                break
                
            elif choice == '2':
                # Launch API server
                launch_api_server()
                break
                
            elif choice == '3':
                # Run functionality test
                print("\n🧪 Running functionality test...")
                import subprocess
                result = subprocess.run([sys.executable, "test_functionality.py"])
                sys.exit(result.returncode)
                
            elif choice == '4':
                # Show system info
                show_system_info()
                
            elif choice == '5':
                print("👋 Goodbye!")
                sys.exit(0)
                
            else:
                print("❌ Invalid choice. Please enter 1-5.")
                
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            sys.exit(0)

def launch_basic_web():
    """Launch basic web interface"""
    web_apps = ['simple_web_gui.py', 'dashboard.py', 'advanced_gui.py', 'main_launcher.py']
    
    for app in web_apps:
        if os.path.exists(app):
            print(f"🚀 Starting {app}...")
            print("🌐 The web interface will be available at http://localhost:8501")
            print("🌍 Or from another machine at: http://YOUR_SERVER_IP:8501")
            print("\n" + "="*50)
            print("📝 Application is starting...")
            print("📝 Press Ctrl+C to stop the application")
            print("="*50 + "\n")
            import subprocess
            try:
                subprocess.run([
                    'streamlit', 'run', app,
                    '--server.address', '0.0.0.0',
                    '--server.port', '8501',
                    '--server.headless', 'true',
                    '--browser.gatherUsageStats', 'false'
                ])
            except FileNotFoundError:
                print("❌ Streamlit not found. Installing...")
                subprocess.check_call([sys.executable, '-m', 'pip', 'install', '--user', 'streamlit>=1.25.0'])
                print("✅ Streamlit installed. Please run the command again.")
            return
    
    print("❌ No web applications found")

def launch_api_server():
    """Launch API server"""
    api_files = ['api/main.py', 'api_dev.py']
    
    for api_file in api_files:
        if os.path.exists(api_file):
            print(f"🚀 Starting API server: {api_file}")
            print("🔌 The API will be available at http://localhost:8000")
            import subprocess
            subprocess.run([
                'uvicorn', f"{api_file.replace('/', '.').replace('.py', '')}:app",
                '--host', '0.0.0.0',
                '--port', '8000'
            ])
            return
    
    print("❌ No API server found")

def show_system_info():
    """Show system information"""
    print("\n" + "=" * 50)
    print("🔍 SYSTEM INFORMATION")
    print("=" * 50)
    print(f"🐍 Python: {sys.version}")
    print(f"💻 Platform: {sys.platform}")
    print(f"🖥️ Display Environment: {'✅ Available' if check_display_environment() else '❌ Headless'}")
    print(f"🎨 GUI Support: {'✅ Available' if check_gui_support() else '❌ Not Available'}")
    
    print(f"\n📁 Available Files:")
    files = ['asset_ml_strategy.py', 'dashboard.py', 'advanced_gui.py', 'web_launcher.py', 'api_dev.py']
    for file in files:
        status = "✅" if os.path.exists(file) else "❌"
        print(f"   {status} {file}")
    
    input("\nPress Enter to continue...")

def main():
    """Main launcher"""
    print("🚀 Asset ML Strategy - Smart Launcher")
    print("=" * 50)
    
    # Check requirements
    if not check_requirements():
        print("\n💡 TIP: Install missing packages with:")
        print("   pip install -r requirements.txt")
        sys.exit(1)
    
    # Determine environment and interface options
    has_display = check_display_environment()
    has_gui = check_gui_support()
    
    print(f"\n🔍 Environment Detection:")
    print(f"   Display Environment: {'✅ Available' if has_display else '❌ Headless'}")
    print(f"   GUI Support: {'✅ Available' if has_gui else '❌ Not Available'}")
    
    # If we have full GUI support and display, offer desktop GUI as primary option
    if has_display and has_gui:
        print("\n🎯 Full GUI environment detected!")
        print("1. Launch Desktop GUI (recommended)")
        print("2. Use Web Interface instead")
        
        choice = input("\nChoose interface (1-2): ").strip()
        if choice == '1':
            try:
                print("🚀 Launching Desktop GUI...")
                from asset_ml_strategy import main as app_main
                app_main()
            except Exception as e:
                print(f"❌ Error launching Desktop GUI: {e}")
                print("🔄 Falling back to web interface...")
                offer_interface_options()
        else:
            offer_interface_options()
    else:
        # Headless or no GUI support - offer web options
        if not has_display:
            print("\n🌐 Headless environment detected - using web interfaces")
        elif not has_gui:
            print("\n⚠️ GUI support unavailable - using web interfaces")
        
        offer_interface_options()

if __name__ == "__main__":
    main()