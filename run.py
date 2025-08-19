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

def main():
    """Main launcher"""
    print("Asset ML Strategy - Launcher")
    print("=" * 40)
    
    # Check requirements
    if not check_requirements():
        sys.exit(1)
    
    # Check GUI support
    if check_gui_support():
        print("✓ GUI support detected")
        print("Launching full GUI application...")
        
        # Import and run the main application
        try:
            from asset_ml_strategy import main as app_main
            app_main()
        except Exception as e:
            print(f"Error launching GUI: {e}")
            print("Try running the test script instead: python test_functionality.py")
            sys.exit(1)
    else:
        print("⚠ GUI not available (headless environment)")
        print("Running functionality test instead...")
        
        # Run the test script
        try:
            import subprocess
            result = subprocess.run([sys.executable, "test_functionality.py"], 
                                  cwd=os.path.dirname(__file__))
            sys.exit(result.returncode)
        except Exception as e:
            print(f"Error running test: {e}")
            sys.exit(1)

if __name__ == "__main__":
    main()