#!/usr/bin/env python3
"""
Web Interface Launcher for Asset ML Strategy
Provides multiple web interface options with automatic setup and dependency management
"""

import subprocess
import sys
import os
import socket
import time
from pathlib import Path

def check_port(port):
    """Check if a port is available"""
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        sock.bind(('localhost', port))
        sock.close()
        return True
    except OSError:
        return False

def find_available_port(start_port=8501):
    """Find the next available port starting from start_port"""
    port = start_port
    while port < start_port + 50:  # Check up to 50 ports
        if check_port(port):
            return port
        port += 1
    return None

def install_web_dependencies():
    """Install required web dependencies"""
    print("🔧 Checking and installing web dependencies...")
    
    required_packages = [
        'streamlit>=1.25.0',
        'plotly>=5.15.0',
        'fastapi>=0.100.0',
        'uvicorn[standard]>=0.23.0',
    ]
    
    missing_packages = []
    
    for package in required_packages:
        package_name = package.split('>=')[0].split('[')[0]
        try:
            __import__(package_name)
            print(f"✓ {package_name}")
        except ImportError:
            missing_packages.append(package)
            print(f"✗ {package_name}")
    
    if missing_packages:
        print(f"\n📦 Installing missing packages: {', '.join(missing_packages)}")
        try:
            subprocess.check_call([
                sys.executable, '-m', 'pip', 'install', '--user'
            ] + missing_packages)
            print("✅ All dependencies installed successfully!")
        except subprocess.CalledProcessError as e:
            print(f"❌ Failed to install dependencies: {e}")
            return False
    
    return True

def launch_streamlit_app(app_file, port=None):
    """Launch a Streamlit application"""
    if not os.path.exists(app_file):
        print(f"❌ Application file not found: {app_file}")
        return False
    
    if port is None:
        port = find_available_port()
    
    if port is None:
        print("❌ No available ports found")
        return False
    
    print(f"🚀 Starting {app_file} on port {port}...")
    print(f"🌐 Access your application at: http://localhost:{port}")
    print(f"🌍 Or from another machine at: http://YOUR_SERVER_IP:{port}")
    print("\n" + "="*60)
    print("📝 Application is starting...")
    print("📝 Press Ctrl+C to stop the application")
    print("="*60 + "\n")
    
    try:
        subprocess.run([
            'streamlit', 'run', app_file,
            '--server.port', str(port),
            '--server.address', '0.0.0.0',
            '--server.headless', 'true',
            '--browser.gatherUsageStats', 'false'
        ])
    except KeyboardInterrupt:
        print("\n🛑 Application stopped by user")
    except FileNotFoundError:
        print("❌ Streamlit not found. Installing...")
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', '--user', 'streamlit'])
        print("✅ Streamlit installed. Please run the command again.")
    
    return True

def launch_fastapi_app():
    """Launch FastAPI application if available"""
    api_files = ['api/main.py', 'api_dev.py']
    
    for api_file in api_files:
        if os.path.exists(api_file):
            port = find_available_port(8000)
            print(f"🚀 Starting FastAPI server: {api_file} on port {port}...")
            try:
                subprocess.run([
                    'uvicorn', f"{api_file.replace('/', '.').replace('.py', '')}:app",
                    '--host', '0.0.0.0',
                    '--port', str(port),
                    '--reload'
                ])
            except FileNotFoundError:
                print("❌ Uvicorn not found. Installing...")
                subprocess.check_call([sys.executable, '-m', 'pip', 'install', '--user', 'uvicorn[standard]'])
            return True
    
    print("❌ No FastAPI application found")
    return False

def show_menu():
    """Show the main menu"""
    print("\n" + "="*60)
    print("🚀 ASSET ML STRATEGY - WEB INTERFACE LAUNCHER")
    print("="*60)
    print("\n📊 Available Web Interfaces:")
    
    interfaces = []
    
    # Check for Streamlit apps
    streamlit_apps = {
        'simple_web_gui.py': '💰 Asset ML Strategy Web Interface - Reliable core functionality',
        'dashboard.py': '🎯 MoneyPrinter Dashboard - Advanced features (requires additional dependencies)',
        'advanced_gui.py': '🔬 Advanced ML Trading Strategy - Full feature suite (requires additional dependencies)',
        'main_launcher.py': '🚀 Master Launcher - All models and features (requires additional dependencies)',
    }
    
    for i, (filename, description) in enumerate(streamlit_apps.items(), 1):
        if os.path.exists(filename):
            interfaces.append((filename, description, 'streamlit'))
            print(f"{i}. {description}")
        else:
            print(f"{i}. {description} [NOT AVAILABLE]")
    
    # Check for API endpoints
    if os.path.exists('api') or os.path.exists('api_dev.py'):
        interfaces.append(('api', '🔌 FastAPI Backend - REST API interface', 'fastapi'))
        print(f"{len(interfaces)}. 🔌 FastAPI Backend - REST API interface")
    
    # Additional options
    print(f"\n{len(interfaces)+1}. 🔧 Install/Update All Dependencies")
    print(f"{len(interfaces)+2}. ❓ System Information & Troubleshooting")
    print(f"{len(interfaces)+3}. 🚪 Exit")
    
    return interfaces

def show_system_info():
    """Show system information and troubleshooting"""
    print("\n" + "="*60)
    print("🔍 SYSTEM INFORMATION & TROUBLESHOOTING")
    print("="*60)
    
    # Python version
    print(f"🐍 Python Version: {sys.version}")
    
    # Check key dependencies
    print("\n📦 Dependency Status:")
    deps = ['pandas', 'numpy', 'sklearn', 'matplotlib', 'streamlit', 'plotly', 'fastapi']
    for dep in deps:
        try:
            module = __import__(dep if dep != 'sklearn' else 'sklearn')
            version = getattr(module, '__version__', 'Unknown')
            print(f"✅ {dep}: {version}")
        except ImportError:
            print(f"❌ {dep}: Not installed")
    
    # Check available ports
    print(f"\n🔌 Available Ports:")
    for port in [8501, 8502, 8503, 8000, 8080]:
        status = "✅ Available" if check_port(port) else "❌ In use"
        print(f"   Port {port}: {status}")
    
    # Check files
    print(f"\n📁 Available Interface Files:")
    files = ['dashboard.py', 'advanced_gui.py', 'main_launcher.py', 'api_dev.py']
    for file in files:
        status = "✅ Found" if os.path.exists(file) else "❌ Missing"
        print(f"   {file}: {status}")
    
    print(f"\n💡 Troubleshooting Tips:")
    print("   • If ports are in use, the launcher will find alternative ports")
    print("   • Missing dependencies will be installed automatically")
    print("   • For 'site can't be reached' errors, check firewall settings")
    print("   • Use 0.0.0.0 address to allow external connections")

def main():
    """Main launcher function"""
    print("🎬 Initializing Asset ML Strategy Web Launcher...")
    
    # Ensure we're in the right directory
    os.chdir(Path(__file__).parent)
    
    while True:
        interfaces = show_menu()
        
        try:
            choice = input(f"\n👉 Select an option (1-{len(interfaces)+3}): ").strip()
            
            if not choice.isdigit():
                print("❌ Please enter a valid number")
                continue
                
            choice = int(choice)
            
            if 1 <= choice <= len(interfaces):
                filename, description, app_type = interfaces[choice-1]
                print(f"\n🎯 Launching: {description}")
                
                if app_type == 'streamlit':
                    if install_web_dependencies():
                        launch_streamlit_app(filename)
                    else:
                        print("❌ Failed to install dependencies. Please try again.")
                elif app_type == 'fastapi':
                    if install_web_dependencies():
                        launch_fastapi_app()
                    else:
                        print("❌ Failed to install dependencies. Please try again.")
            
            elif choice == len(interfaces) + 1:
                print("\n🔧 Installing/Updating all dependencies...")
                install_web_dependencies()
                print("✅ Dependencies updated!")
                input("Press Enter to continue...")
            
            elif choice == len(interfaces) + 2:
                show_system_info()
                input("\nPress Enter to continue...")
            
            elif choice == len(interfaces) + 3:
                print("👋 Goodbye!")
                break
            
            else:
                print("❌ Invalid choice. Please try again.")
                
        except KeyboardInterrupt:
            print("\n\n👋 Goodbye!")
            break
        except ValueError:
            print("❌ Please enter a valid number")
        except Exception as e:
            print(f"❌ An error occurred: {e}")

if __name__ == "__main__":
    main()