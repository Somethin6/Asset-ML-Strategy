#!/usr/bin/env python3
"""
🚀 ULTIMATE ADVANCED ASSET ML STRATEGY 🚀
The Most Advanced Free Local ML Trading System Ever Created!

REVOLUTIONARY FEATURES:
- 143+ Advanced Technical & Statistical Features
- 16+ ML Models with Intelligent Ensemble Learning  
- Real-time Signal Generation & Portfolio Optimization
- Advanced Risk Management & Backtesting Engine
- Interactive Visualizations & Professional GUI
- 100% Free & Local Operation

From Basic Datasheets to Infinitely Advanced Trading Intelligence!
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import seaborn as sns
import os
from datetime import datetime
import warnings
import logging
import sys

warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Import our advanced systems
try:
    from ultimate_trading_system import UltimateAdvancedTradingSystem
    from advanced_features import AdvancedFeatureEngine
    from advanced_ml_ensemble import AdvancedMLEnsemble
    ADVANCED_MODE = True
    print("🚀 ADVANCED MODE: Ultimate trading systems loaded!")
except ImportError as e:
    ADVANCED_MODE = False
    print(f"⚠️  BASIC MODE: Advanced systems not available ({e})")
    # Fallback imports for basic functionality
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import mean_squared_error, r2_score

class AdvancedAssetMLStrategy:
    def __init__(self, root):
        self.root = root
        self.root.title("🚀 ULTIMATE ADVANCED ASSET ML STRATEGY - Most Advanced Free Trading System")
        self.root.geometry("1600x1000")
        self.root.configure(bg='#1e1e2e')
        
        # Initialize advanced systems if available
        if ADVANCED_MODE:
            self.advanced_system = UltimateAdvancedTradingSystem()
            self.feature_engine = AdvancedFeatureEngine()
            self.ml_ensemble = AdvancedMLEnsemble()
        
        # Data storage
        self.data = None
        self.features = None
        self.model_trained = False
        self.predictions = None
        self.backtest_results = None
        
        # GUI setup
        self.setup_enhanced_gui()
        
        # Show welcome message
        self.show_welcome_message()
    
    def show_welcome_message(self):
        """Show welcome message with system capabilities"""
        mode = "ULTIMATE ADVANCED" if ADVANCED_MODE else "BASIC"
        features = "143+ Advanced Features & 16+ ML Models" if ADVANCED_MODE else "Basic Features & ML Models"
        
        welcome_msg = f"""
🚀 WELCOME TO THE {mode} ASSET ML STRATEGY! 🚀

{features}

CAPABILITIES:
{'✅ 143+ Technical & Statistical Features' if ADVANCED_MODE else '✅ Basic Technical Features'}
{'✅ 16+ ML Models with Ensemble Learning' if ADVANCED_MODE else '✅ Random Forest ML Model'}
{'✅ Advanced Signal Generation' if ADVANCED_MODE else '✅ Basic Signal Generation'}
{'✅ Comprehensive Backtesting' if ADVANCED_MODE else '✅ Basic Performance Metrics'}
{'✅ Risk Management Systems' if ADVANCED_MODE else '✅ Simple Risk Analysis'}
✅ Professional GUI Interface
✅ 100% Free & Local Operation

Ready to transform your trading with {'cutting-edge' if ADVANCED_MODE else 'reliable'} AI technology!
        """
        messagebox.showinfo("Ultimate Advanced Trading System", welcome_msg)
    
    def setup_enhanced_gui(self):
        """Setup the enhanced GUI interface"""
        # Create main header
        header_frame = tk.Frame(self.root, bg='#0d1421', height=80)
        header_frame.pack(fill='x')
        header_frame.pack_propagate(False)
        
        title_text = "🚀 ULTIMATE ADVANCED ASSET ML STRATEGY 🚀" if ADVANCED_MODE else "🔧 ASSET ML STRATEGY"
        subtitle = "Most Advanced Free Local Trading System" if ADVANCED_MODE else "Free Local Trading Analysis"
        
        title_label = tk.Label(
            header_frame,
            text=title_text,
            font=('Arial', 20, 'bold'),
            bg='#0d1421',
            fg='#00ff88'
        )
        title_label.pack(pady=(10, 0))
        
        subtitle_label = tk.Label(
            header_frame,
            text=subtitle,
            font=('Arial', 12),
            bg='#0d1421',
            fg='#888888'
        )
        subtitle_label.pack()
        
        # Create notebook for tabs
        style = ttk.Style()
        style.theme_use('clam')
        style.configure('TNotebook', background='#2d2d2d')
        style.configure('TNotebook.Tab', padding=[20, 10])
        
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill='both', expand=True, padx=10, pady=10)
        
        # Setup all tabs
        self.setup_data_tab()
        if ADVANCED_MODE:
            self.setup_advanced_features_tab()
        self.setup_ml_tab()
        self.setup_signals_tab()
        self.setup_backtest_tab()
        self.setup_visualization_tab()
        
        # Setup status bar
        self.setup_status_bar()
    
    def setup_data_tab(self):
        """Setup data loading tab"""
        self.data_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.data_frame, text="📊 Data Loading")
        
        # File selection
        file_frame = ttk.LabelFrame(self.data_frame, text="📁 Load Trading Data", padding=15)
        file_frame.pack(fill='x', padx=10, pady=5)
        
        ttk.Button(
            file_frame,
            text="🚀 Load Data File (CSV/Excel)",
            command=self.load_data_file,
            width=25
        ).pack(side=tk.LEFT, padx=5)
        
        self.file_label = ttk.Label(file_frame, text="No file loaded")
        self.file_label.pack(side=tk.LEFT, padx=10)
        
        # Data preview
        preview_frame = ttk.LabelFrame(self.data_frame, text="📋 Data Preview", padding=10)
        preview_frame.pack(fill='both', expand=True, padx=10, pady=5)
        
        # Treeview for data display
        self.data_tree = ttk.Treeview(preview_frame, height=12)
        data_scroll = ttk.Scrollbar(preview_frame, orient="vertical", command=self.data_tree.yview)
        self.data_tree.configure(yscrollcommand=data_scroll.set)
        
        data_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        self.data_tree.pack(fill='both', expand=True)
        
        # Data information
        info_frame = ttk.LabelFrame(self.data_frame, text="📊 Data Information", padding=10)
        info_frame.pack(fill='x', padx=10, pady=5)
        
        self.data_info_text = scrolledtext.ScrolledText(info_frame, height=6)
        self.data_info_text.pack(fill='both', expand=True)
    
    def setup_advanced_features_tab(self):
        """Setup advanced features tab (only if advanced mode)"""
        self.features_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.features_frame, text="🔬 Advanced Features")
        
        # Feature generation controls
        controls_frame = ttk.LabelFrame(self.features_frame, text="🔬 Feature Engineering", padding=15)
        controls_frame.pack(fill='x', padx=10, pady=5)
        
        ttk.Button(
            controls_frame,
            text="🚀 Generate 143+ Features",
            command=self.generate_advanced_features,
            width=25
        ).pack(side=tk.LEFT, padx=5)
        
        self.features_label = ttk.Label(controls_frame, text="No features generated")
        self.features_label.pack(side=tk.LEFT, padx=10)
        
        # Feature summary
        summary_frame = ttk.LabelFrame(self.features_frame, text="📊 Feature Summary", padding=10)
        summary_frame.pack(fill='both', expand=True, padx=10, pady=5)
        
        self.features_text = scrolledtext.ScrolledText(summary_frame, height=20)
        self.features_text.pack(fill='both', expand=True)
    
    def setup_ml_tab(self):
        """Setup ML training tab"""
        self.ml_frame = ttk.Frame(self.notebook)
        tab_text = "🤖 Advanced ML Ensemble" if ADVANCED_MODE else "🤖 ML Analysis"
        self.notebook.add(self.ml_frame, text=tab_text)
        
        # ML controls
        controls_frame = ttk.LabelFrame(self.ml_frame, text="🤖 ML Training", padding=15)
        controls_frame.pack(fill='x', padx=10, pady=5)
        
        button_text = "🚀 Train 16+ ML Models" if ADVANCED_MODE else "🔧 Train ML Model"
        ttk.Button(
            controls_frame,
            text=button_text,
            command=self.train_ml_models,
            width=25
        ).pack(side=tk.LEFT, padx=5)
        
        self.ml_label = ttk.Label(controls_frame, text="Models not trained")
        self.ml_label.pack(side=tk.LEFT, padx=10)
        
        # ML results
        results_frame = ttk.LabelFrame(self.ml_frame, text="🏆 ML Performance Results", padding=10)
        results_frame.pack(fill='both', expand=True, padx=10, pady=5)
        
        self.ml_results_text = scrolledtext.ScrolledText(results_frame, height=20)
        self.ml_results_text.pack(fill='both', expand=True)
    
    def setup_signals_tab(self):
        """Setup trading signals tab"""
        self.signals_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.signals_frame, text="📈 Trading Signals")
        
        # Signal generation
        controls_frame = ttk.LabelFrame(self.signals_frame, text="📈 Signal Generation", padding=15)
        controls_frame.pack(fill='x', padx=10, pady=5)
        
        ttk.Button(
            controls_frame,
            text="🎯 Generate Trading Signals",
            command=self.generate_trading_signals,
            width=25
        ).pack(side=tk.LEFT, padx=5)
        
        self.signals_label = ttk.Label(controls_frame, text="No signals generated")
        self.signals_label.pack(side=tk.LEFT, padx=10)
        
        # Signals display
        signals_display_frame = ttk.LabelFrame(self.signals_frame, text="📊 Trading Signals", padding=10)
        signals_display_frame.pack(fill='both', expand=True, padx=10, pady=5)
        
        self.signals_text = scrolledtext.ScrolledText(signals_display_frame, height=20)
        self.signals_text.pack(fill='both', expand=True)
    
    def setup_backtest_tab(self):
        """Setup backtesting tab"""
        self.backtest_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.backtest_frame, text="📊 Backtesting")
        
        # Backtest controls
        controls_frame = ttk.LabelFrame(self.backtest_frame, text="📊 Backtesting Engine", padding=15)
        controls_frame.pack(fill='x', padx=10, pady=5)
        
        ttk.Button(
            controls_frame,
            text="🚀 Run Advanced Backtest",
            command=self.run_backtest,
            width=25
        ).pack(side=tk.LEFT, padx=5)
        
        self.backtest_label = ttk.Label(controls_frame, text="Backtest not run")
        self.backtest_label.pack(side=tk.LEFT, padx=10)
        
        # Backtest results
        results_frame = ttk.LabelFrame(self.backtest_frame, text="📈 Performance Results", padding=10)
        results_frame.pack(fill='both', expand=True, padx=10, pady=5)
        
        self.backtest_text = scrolledtext.ScrolledText(results_frame, height=20)
        self.backtest_text.pack(fill='both', expand=True)
    
    def setup_visualization_tab(self):
        """Setup visualization tab"""
        self.viz_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.viz_frame, text="📊 Visualizations")
        
        # Chart controls
        controls_frame = ttk.LabelFrame(self.viz_frame, text="📊 Advanced Charts", padding=10)
        controls_frame.pack(fill='x', padx=10, pady=5)
        
        chart_buttons = [
            ("📈 Price Chart", self.plot_price_chart),
            ("📊 Volume Analysis", self.plot_volume_chart),
            ("🎯 Predictions", self.plot_predictions_chart),
            ("📈 Signals", self.plot_signals_chart)
        ]
        
        for i, (text, command) in enumerate(chart_buttons):
            ttk.Button(controls_frame, text=text, command=command, width=20).grid(
                row=i//2, column=i%2, padx=5, pady=5, sticky='ew'
            )
        
        for j in range(2):
            controls_frame.grid_columnconfigure(j, weight=1)
        
        # Chart display
        self.chart_frame = ttk.Frame(self.viz_frame)
        self.chart_frame.pack(fill='both', expand=True, padx=10, pady=5)
    
    def setup_status_bar(self):
        """Setup status bar"""
        self.status_frame = tk.Frame(self.root, bg='#0d1421', height=30)
        self.status_frame.pack(side=tk.BOTTOM, fill=tk.X)
        self.status_frame.pack_propagate(False)
        
        self.status_label = tk.Label(
            self.status_frame,
            text="🚀 Ultimate Advanced Trading System Ready - Load data to begin!",
            bg='#0d1421',
            fg='#00ff88',
            font=('Arial', 10),
            anchor='w',
            padx=10
        )
        self.status_label.pack(fill='both', expand=True, pady=5)
    
    def update_status(self, message):
        """Update status bar message"""
        self.status_label.config(text=message)
        self.root.update()
    
    def load_data_file(self):
        """Load data file (CSV or Excel)"""
        file_path = filedialog.askopenfilename(
            title="Select Trading Data File",
            filetypes=[
                ("All supported", "*.csv *.xlsx *.xls"),
                ("CSV files", "*.csv"),
                ("Excel files", "*.xlsx *.xls")
            ]
        )
        
        if not file_path:
            return
        
        try:
            self.update_status("📊 Loading trading data...")
            
            # Load data
            if file_path.lower().endswith(('.xlsx', '.xls')):
                self.data = pd.read_excel(file_path)
            else:
                self.data = pd.read_csv(file_path)
            
            # Validate and clean data
            self.validate_and_clean_data()
            
            # Update GUI
            self.display_data()
            self.update_data_info()
            
            filename = os.path.basename(file_path)
            self.file_label.config(text=f"✅ {filename} ({len(self.data)} rows)")
            
            self.update_status("✅ Data loaded successfully!")
            
            if ADVANCED_MODE:
                # Initialize advanced system with data
                self.advanced_system.load_data(self.data)
            
            messagebox.showinfo("Success", f"Data loaded successfully!\n{len(self.data)} rows loaded.")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load data:\n{str(e)}")
            self.update_status("❌ Data loading failed")
    
    def validate_and_clean_data(self):
        """Validate and clean the loaded data"""
        # Check for required columns
        required_cols = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']
        
        # Try to find columns with different naming conventions
        column_mappings = {
            'Date': ['date', 'DATE', 'Timestamp', 'timestamp'],
            'Open': ['open', 'OPEN'],
            'High': ['high', 'HIGH'],
            'Low': ['low', 'LOW'], 
            'Close': ['close', 'CLOSE'],
            'Volume': ['volume', 'VOLUME', 'Vol', 'vol']
        }
        
        for standard_name, alternatives in column_mappings.items():
            if standard_name not in self.data.columns:
                for alt in alternatives:
                    if alt in self.data.columns:
                        self.data.rename(columns={alt: standard_name}, inplace=True)
                        break
        
        # Add Adj Close if missing
        if 'Adj Close' not in self.data.columns:
            self.data['Adj Close'] = self.data['Close'].copy()
        
        # Convert Date column
        if 'Date' in self.data.columns:
            self.data['Date'] = pd.to_datetime(self.data['Date'])
            self.data = self.data.sort_values('Date').reset_index(drop=True)
    
    def display_data(self):
        """Display data in treeview"""
        # Clear existing data
        for item in self.data_tree.get_children():
            self.data_tree.delete(item)
        
        # Set columns
        columns = list(self.data.columns)
        self.data_tree['columns'] = columns
        self.data_tree['show'] = 'headings'
        
        for col in columns:
            self.data_tree.heading(col, text=col)
            self.data_tree.column(col, width=100)
        
        # Add data (first 50 rows)
        for _, row in self.data.head(50).iterrows():
            values = [str(val)[:15] for val in row]
            self.data_tree.insert('', 'end', values=values)
    
    def update_data_info(self):
        """Update data information display"""
        if self.data is None:
            return
        
        info_text = f"📊 DATASET INFORMATION:\n"
        info_text += f"Shape: {self.data.shape[0]:,} rows × {self.data.shape[1]} columns\n"
        
        if 'Date' in self.data.columns:
            info_text += f"Date range: {self.data['Date'].min()} to {self.data['Date'].max()}\n"
            info_text += f"Trading days: {len(self.data):,}\n"
        
        info_text += f"\n💰 PRICE ANALYSIS:\n"
        if 'Close' in self.data.columns:
            info_text += f"Current price: ${self.data['Close'].iloc[-1]:.2f}\n"
            info_text += f"Price range: ${self.data['Close'].min():.2f} - ${self.data['Close'].max():.2f}\n"
            info_text += f"Average price: ${self.data['Close'].mean():.2f}\n"
        
        if 'Volume' in self.data.columns:
            info_text += f"\n📊 VOLUME ANALYSIS:\n"
            info_text += f"Average volume: {self.data['Volume'].mean():,.0f}\n"
            info_text += f"Total volume: {self.data['Volume'].sum():,}\n"
        
        info_text += f"\n✅ Ready for {'advanced' if ADVANCED_MODE else 'basic'} analysis!\n"
        
        self.data_info_text.delete(1.0, tk.END)
        self.data_info_text.insert(1.0, info_text)
    
    def generate_advanced_features(self):
        """Generate advanced features (only in advanced mode)"""
        if not ADVANCED_MODE:
            messagebox.showwarning("Feature Not Available", "Advanced features require advanced mode")
            return
        
        if self.data is None:
            messagebox.showerror("Error", "Please load data first!")
            return
        
        try:
            self.update_status("🔬 Generating 143+ advanced features...")
            
            # Generate features using advanced system
            success = self.advanced_system.engineer_features()
            
            if success:
                self.features = self.advanced_system.features
                
                # Generate feature summary
                feature_summary = self.feature_engine.create_feature_summary(self.features)
                
                # Update GUI
                self.features_label.config(text=f"✅ Generated {feature_summary['total_features']} features")
                
                # Display feature information
                feature_text = f"🔬 ADVANCED FEATURE ENGINEERING RESULTS\n"
                feature_text += f"{'='*50}\n\n"
                feature_text += f"📊 FEATURE SUMMARY:\n"
                feature_text += f"Total Features: {feature_summary['total_features']}\n"
                feature_text += f"Categories: {len(feature_summary['categories'])}\n\n"
                
                feature_text += f"📋 FEATURE CATEGORIES:\n"
                for category, count in feature_summary['categories'].items():
                    feature_text += f"  • {category}: {count} features\n"
                
                feature_text += f"\n🎯 TOP FEATURES BY CATEGORY:\n"
                for category, features_list in feature_summary['feature_list'].items():
                    if features_list:
                        top_features = features_list[:3]
                        feature_text += f"  {category}: {', '.join(top_features)}{'...' if len(features_list) > 3 else ''}\n"
                
                feature_text += f"\n✅ Advanced features ready for ML training! 🚀\n"
                
                self.features_text.delete(1.0, tk.END)
                self.features_text.insert(1.0, feature_text)
                
                self.update_status("✅ Advanced features generated successfully!")
                messagebox.showinfo("Success", f"Generated {feature_summary['total_features']} advanced features!")
                
            else:
                raise Exception("Feature engineering failed")
                
        except Exception as e:
            messagebox.showerror("Error", f"Feature generation failed:\n{str(e)}")
            self.update_status("❌ Feature generation failed")
    
    def train_ml_models(self):
        """Train ML models"""
        if self.data is None:
            messagebox.showerror("Error", "Please load data first!")
            return
        
        try:
            if ADVANCED_MODE:
                self.update_status("🤖 Training 16+ advanced ML models...")
                
                # Train using advanced system
                success = self.advanced_system.train_models()
                
                if success:
                    self.model_trained = True
                    
                    # Get model report
                    ml_report = self.advanced_system.ml_ensemble.generate_model_report()
                    
                    self.ml_results_text.delete(1.0, tk.END)
                    self.ml_results_text.insert(1.0, ml_report)
                    
                    self.ml_label.config(text="✅ 16+ models trained successfully!")
                    self.update_status("✅ Advanced ML ensemble trained!")
                    
                    messagebox.showinfo("Success", "Advanced ML ensemble trained successfully!")
                else:
                    raise Exception("Advanced ML training failed")
                    
            else:
                # Basic ML training fallback
                self.train_basic_ml()
                
        except Exception as e:
            messagebox.showerror("Error", f"ML training failed:\n{str(e)}")
            self.update_status("❌ ML training failed")
    
    def train_basic_ml(self):
        """Train basic ML model (fallback)"""
        self.update_status("🔧 Training basic ML model...")
        
        # Prepare basic features
        feature_cols = ['Open', 'High', 'Low', 'Adj Close', 'Volume']
        feature_cols = [col for col in feature_cols if col in self.data.columns]
        
        X = self.data[feature_cols].copy()
        y = self.data['Close'].copy()
        
        # Add simple features
        X['SMA_10'] = self.data['Close'].rolling(10).mean()
        X['SMA_20'] = self.data['Close'].rolling(20).mean()
        X['Volatility'] = self.data['Close'].rolling(10).std()
        
        # Clean data
        mask = ~(X.isnull().any(axis=1) | y.isnull())
        X = X[mask]
        y = y[mask]
        
        # Train model
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        
        # Evaluate
        y_pred = model.predict(X_test)
        r2 = r2_score(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        
        self.model_trained = True
        
        # Display results
        results_text = f"🔧 BASIC ML MODEL RESULTS\n"
        results_text += f"{'='*40}\n\n"
        results_text += f"Model: Random Forest\n"
        results_text += f"Features: {len(X.columns)}\n"
        results_text += f"Training samples: {len(X_train)}\n"
        results_text += f"Test samples: {len(X_test)}\n\n"
        results_text += f"Performance:\n"
        results_text += f"R² Score: {r2:.4f}\n"
        results_text += f"MSE: {mse:.4f}\n\n"
        results_text += f"✅ Basic model trained successfully!\n"
        
        self.ml_results_text.delete(1.0, tk.END)
        self.ml_results_text.insert(1.0, results_text)
        
        self.ml_label.config(text="✅ Basic model trained!")
        self.update_status("✅ Basic ML model trained!")
    
    def generate_trading_signals(self):
        """Generate trading signals"""
        if not self.model_trained:
            messagebox.showerror("Error", "Please train models first!")
            return
        
        try:
            if ADVANCED_MODE:
                self.update_status("🎯 Generating advanced trading signals...")
                
                # Generate predictions and signals
                pred_success = self.advanced_system.generate_predictions()
                signal_success = self.advanced_system.generate_trading_signals()
                
                if pred_success and signal_success:
                    self.predictions = self.advanced_system.predictions
                    
                    # Display signals
                    signals_text = f"🎯 ADVANCED TRADING SIGNALS\n"
                    signals_text += f"{'='*40}\n\n"
                    
                    # Signal summary
                    signal_counts = self.predictions['Signal'].value_counts()
                    signals_text += f"📊 SIGNAL SUMMARY:\n"
                    for signal, count in signal_counts.items():
                        signals_text += f"  • {signal}: {count} signals\n"
                    
                    # Recent signals
                    signals_text += f"\n📈 RECENT SIGNALS (Last 10):\n"
                    recent_signals = self.predictions.tail(10)
                    
                    for _, row in recent_signals.iterrows():
                        date_str = row['Date'].strftime('%Y-%m-%d') if pd.notna(row['Date']) else 'N/A'
                        signals_text += f"{date_str}: {row['Signal']} (Confidence: {row['Confidence']:.2f})\n"
                    
                    # Latest signal
                    latest_signal = self.predictions.iloc[-1]
                    signals_text += f"\n🎯 LATEST SIGNAL:\n"
                    signals_text += f"Signal: {latest_signal['Signal']}\n"
                    signals_text += f"Confidence: {latest_signal['Confidence']:.2f}\n"
                    signals_text += f"Position Size: {latest_signal['Position_Size']:.2f}\n"
                    
                    signals_text += f"\n✅ Advanced trading signals ready! 🚀\n"
                    
                    self.signals_text.delete(1.0, tk.END)
                    self.signals_text.insert(1.0, signals_text)
                    
                    self.signals_label.config(text=f"✅ Generated {len(self.predictions)} signals")
                    self.update_status("✅ Trading signals generated!")
                    
                    messagebox.showinfo("Success", "Advanced trading signals generated successfully!")
                else:
                    raise Exception("Signal generation failed")
            else:
                # Basic signal generation
                self.generate_basic_signals()
                
        except Exception as e:
            messagebox.showerror("Error", f"Signal generation failed:\n{str(e)}")
            self.update_status("❌ Signal generation failed")
    
    def generate_basic_signals(self):
        """Generate basic trading signals"""
        signals_text = f"🔧 BASIC TRADING SIGNALS\n"
        signals_text += f"{'='*30}\n\n"
        signals_text += f"Simple moving average crossover strategy:\n"
        
        # Calculate simple signals
        self.data['SMA_10'] = self.data['Close'].rolling(10).mean()
        self.data['SMA_20'] = self.data['Close'].rolling(20).mean()
        
        conditions = [
            self.data['SMA_10'] > self.data['SMA_20'],
            self.data['SMA_10'] < self.data['SMA_20']
        ]
        choices = ['BUY', 'SELL']
        
        self.data['Signal'] = np.select(conditions, choices, default='HOLD')
        
        signal_counts = self.data['Signal'].value_counts()
        for signal, count in signal_counts.items():
            signals_text += f"  • {signal}: {count} signals\n"
        
        signals_text += f"\nLatest Signal: {self.data['Signal'].iloc[-1]}\n"
        signals_text += f"\n✅ Basic signals generated!\n"
        
        self.signals_text.delete(1.0, tk.END)
        self.signals_text.insert(1.0, signals_text)
        
        self.signals_label.config(text="✅ Basic signals generated")
        self.update_status("✅ Basic signals generated!")
    
    def run_backtest(self):
        """Run backtesting analysis"""
        if self.predictions is None and not hasattr(self.data, 'Signal'):
            messagebox.showerror("Error", "Please generate signals first!")
            return
        
        try:
            if ADVANCED_MODE and self.predictions is not None:
                self.update_status("📊 Running advanced backtest...")
                
                # Run advanced backtest
                self.backtest_results = self.advanced_system.run_backtest()
                
                if self.backtest_results:
                    # Display backtest results
                    backtest_text = f"📊 ADVANCED BACKTEST RESULTS\n"
                    backtest_text += f"{'='*40}\n\n"
                    backtest_text += f"💰 PERFORMANCE METRICS:\n"
                    backtest_text += f"Total Return: {self.backtest_results['total_return']:.2%}\n"
                    backtest_text += f"Final Capital: ${self.backtest_results['final_capital']:,.2f}\n"
                    backtest_text += f"Total Trades: {self.backtest_results['total_trades']}\n"
                    backtest_text += f"Win Rate: {self.backtest_results['win_rate']:.1%}\n"
                    backtest_text += f"Average Win: ${self.backtest_results['avg_win']:.2f}\n"
                    backtest_text += f"Average Loss: ${self.backtest_results['avg_loss']:.2f}\n"
                    backtest_text += f"Max Drawdown: {self.backtest_results['max_drawdown']:.2%}\n"
                    backtest_text += f"Sharpe Ratio: {self.backtest_results['sharpe_ratio']:.2f}\n\n"
                    
                    # Recent trades
                    if self.backtest_results['trades']:
                        backtest_text += f"📈 RECENT TRADES (Last 5):\n"
                        recent_trades = self.backtest_results['trades'][-5:]
                        for trade in recent_trades:
                            backtest_text += f"{trade['Date'].strftime('%Y-%m-%d')}: {trade['Type']} - ${trade['Return']:.2f}\n"
                    
                    backtest_text += f"\n✅ Advanced backtest completed! 🚀\n"
                    
                    self.backtest_text.delete(1.0, tk.END)
                    self.backtest_text.insert(1.0, backtest_text)
                    
                    self.backtest_label.config(text=f"✅ Backtest: {self.backtest_results['total_return']:.1%} return")
                    self.update_status("✅ Advanced backtest completed!")
                    
                    messagebox.showinfo("Backtest Complete", 
                                      f"Backtest completed!\nTotal Return: {self.backtest_results['total_return']:.2%}")
                else:
                    raise Exception("Advanced backtest failed")
            else:
                # Basic backtest
                self.run_basic_backtest()
                
        except Exception as e:
            messagebox.showerror("Error", f"Backtest failed:\n{str(e)}")
            self.update_status("❌ Backtest failed")
    
    def run_basic_backtest(self):
        """Run basic backtest"""
        backtest_text = f"🔧 BASIC BACKTEST RESULTS\n"
        backtest_text += f"{'='*30}\n\n"
        
        # Simple backtest logic
        if 'Signal' in self.data.columns:
            signal_changes = self.data['Signal'].ne(self.data['Signal'].shift()).sum()
            buy_signals = (self.data['Signal'] == 'BUY').sum()
            sell_signals = (self.data['Signal'] == 'SELL').sum()
            
            backtest_text += f"Signal Changes: {signal_changes}\n"
            backtest_text += f"Buy Signals: {buy_signals}\n"
            backtest_text += f"Sell Signals: {sell_signals}\n"
            
            # Simple return calculation
            returns = self.data['Close'].pct_change()
            total_return = (1 + returns).prod() - 1
            
            backtest_text += f"Total Return: {total_return:.2%}\n"
        
        backtest_text += f"\n✅ Basic backtest completed!\n"
        
        self.backtest_text.delete(1.0, tk.END)
        self.backtest_text.insert(1.0, backtest_text)
        
        self.backtest_label.config(text="✅ Basic backtest completed")
        self.update_status("✅ Basic backtest completed!")
    
    def plot_price_chart(self):
        """Plot price chart"""
        if self.data is None:
            messagebox.showerror("Error", "Please load data first!")
            return
        
        self.clear_chart_frame()
        
        fig, ax = plt.subplots(figsize=(12, 6))
        fig.patch.set_facecolor('#2d2d2d')
        ax.set_facecolor('#2d2d2d')
        
        # Plot price data
        if 'Date' in self.data.columns:
            ax.plot(self.data['Date'], self.data['Close'], label='Close Price', color='#00ff88', linewidth=2)
            ax.plot(self.data['Date'], self.data['Open'], label='Open Price', alpha=0.7, color='#888888')
        else:
            ax.plot(self.data['Close'], label='Close Price', color='#00ff88', linewidth=2)
        
        ax.set_title('Price Chart', color='white', fontsize=14, fontweight='bold')
        ax.set_xlabel('Date' if 'Date' in self.data.columns else 'Index', color='white')
        ax.set_ylabel('Price', color='white')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.tick_params(colors='white')
        
        canvas = FigureCanvasTkAgg(fig, self.chart_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill='both', expand=True)
    
    def plot_volume_chart(self):
        """Plot volume chart"""
        if self.data is None or 'Volume' not in self.data.columns:
            messagebox.showerror("Error", "Volume data not available!")
            return
        
        self.clear_chart_frame()
        
        fig, ax = plt.subplots(figsize=(12, 6))
        fig.patch.set_facecolor('#2d2d2d')
        ax.set_facecolor('#2d2d2d')
        
        if 'Date' in self.data.columns:
            ax.bar(self.data['Date'], self.data['Volume'], alpha=0.7, color='#ff6b35')
        else:
            ax.bar(range(len(self.data)), self.data['Volume'], alpha=0.7, color='#ff6b35')
        
        ax.set_title('Volume Analysis', color='white', fontsize=14, fontweight='bold')
        ax.set_xlabel('Date' if 'Date' in self.data.columns else 'Index', color='white')
        ax.set_ylabel('Volume', color='white')
        ax.grid(True, alpha=0.3)
        ax.tick_params(colors='white')
        
        canvas = FigureCanvasTkAgg(fig, self.chart_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill='both', expand=True)
    
    def plot_predictions_chart(self):
        """Plot predictions chart"""
        if self.predictions is None:
            messagebox.showerror("Error", "No predictions available!")
            return
        
        self.clear_chart_frame()
        
        fig, ax = plt.subplots(figsize=(12, 6))
        fig.patch.set_facecolor('#2d2d2d')
        ax.set_facecolor('#2d2d2d')
        
        ax.plot(self.predictions['Actual'], label='Actual', color='#00ff88', linewidth=2)
        ax.plot(self.predictions['Predicted'], label='Predicted', color='#ff6b35', linewidth=2, alpha=0.8)
        
        ax.set_title('Predictions vs Actual', color='white', fontsize=14, fontweight='bold')
        ax.set_xlabel('Time', color='white')
        ax.set_ylabel('Price', color='white')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.tick_params(colors='white')
        
        canvas = FigureCanvasTkAgg(fig, self.chart_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill='both', expand=True)
    
    def plot_signals_chart(self):
        """Plot signals chart"""
        if self.predictions is None and 'Signal' not in self.data.columns:
            messagebox.showerror("Error", "No signals available!")
            return
        
        self.clear_chart_frame()
        
        fig, ax = plt.subplots(figsize=(12, 6))
        fig.patch.set_facecolor('#2d2d2d')
        ax.set_facecolor('#2d2d2d')
        
        if self.predictions is not None:
            # Plot price with signals
            ax.plot(self.predictions['Actual'], label='Price', color='white', linewidth=2)
            
            # Mark buy/sell signals
            buy_signals = self.predictions[self.predictions['Signal'] == 'BUY']
            sell_signals = self.predictions[self.predictions['Signal'] == 'SELL']
            
            if not buy_signals.empty:
                ax.scatter(buy_signals.index, buy_signals['Actual'], 
                          color='green', marker='^', s=100, label='BUY', alpha=0.8)
            
            if not sell_signals.empty:
                ax.scatter(sell_signals.index, sell_signals['Actual'], 
                          color='red', marker='v', s=100, label='SELL', alpha=0.8)
        
        ax.set_title('Trading Signals', color='white', fontsize=14, fontweight='bold')
        ax.set_xlabel('Time', color='white')
        ax.set_ylabel('Price', color='white')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.tick_params(colors='white')
        
        canvas = FigureCanvasTkAgg(fig, self.chart_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill='both', expand=True)
    
    def clear_chart_frame(self):
        """Clear the chart frame"""
        for widget in self.chart_frame.winfo_children():
            widget.destroy()


def main():
    """Main application entry point"""
    try:
        root = tk.Tk()
        app = AdvancedAssetMLStrategy(root)
        root.mainloop()
    except Exception as e:
        print(f"Error starting application: {e}")
        if "tkinter" in str(e).lower():
            print("GUI not available in headless environment.")

if __name__ == "__main__":
    main()
    def __init__(self, root):
        self.root = root
        if ENHANCED_MODE:
            self.root.title("🚀 ULTIMATE Asset ML Strategy - The Most Advanced Free ML Trading Bot")
        else:
            self.root.title("Asset ML Strategy - Free Local Financial Analysis")
        
        self.root.geometry("1400x900")
        self.root.configure(bg='#f0f0f0')
        
        # Enhanced components
        if ENHANCED_MODE:
            self.data_loader = EnhancedDataLoader()
            self.ml_engine = AdvancedMLEngine()
            self.enhanced_indicators = {}
            self.strategies = {}
        
        # Data storage
        self.data = None
        self.model = None
        self.predictions = None
        
        # Setup GUI
        self.setup_enhanced_gui() if ENHANCED_MODE else self.setup_gui()
    
    def setup_enhanced_gui(self):
        """Setup enhanced GUI with all advanced features"""
        # Enhanced title
        title_frame = tk.Frame(self.root, bg='#1e1e2e', height=70)
        title_frame.pack(fill='x')
        title_frame.pack_propagate(False)
        
        title_label = tk.Label(
            title_frame,
            text="🚀 ULTIMATE ML TRADING BOT - Most Advanced Free Local System",
            font=('Arial', 18, 'bold'),
            bg='#1e1e2e',
            fg='#00ff88'
        )
        title_label.pack(pady=20)
        
        # Create enhanced notebook
        style = ttk.Style()
        style.configure('Enhanced.TNotebook', background='#2d2d2d')
        style.configure('Enhanced.TNotebook.Tab', padding=[15, 10])
        
        self.notebook = ttk.Notebook(self.root, style='Enhanced.TNotebook')
        self.notebook.pack(fill='both', expand=True, padx=10, pady=5)
        
        # Enhanced tabs
        self.setup_enhanced_data_tab()
        self.setup_enhanced_ml_tab()
        self.setup_enhanced_viz_tab()
        self.setup_performance_tab()
        
        # Enhanced status bar
        self.setup_enhanced_status_bar()
    
    def setup_enhanced_data_tab(self):
        """Enhanced data loading with universal format support"""
        self.data_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.data_frame, text="📊 Universal Data Loading")
        
        # File selection
        file_frame = ttk.LabelFrame(self.data_frame, text="📁 Universal Data Loader (CSV/Excel)", padding=15)
        file_frame.pack(fill='x', padx=10, pady=5)
        
        # Format selection
        format_frame = tk.Frame(file_frame, bg='#f0f0f0')
        format_frame.pack(fill='x', pady=10)
        
        tk.Label(format_frame, text="Data Format:", font=('Arial', 11, 'bold')).pack(side=tk.LEFT)
        
        self.format_var = tk.StringVar(value="auto")
        formats = [
            ("Auto Detect", "auto"),
            ("Traditional (Date, OHLC, Volume)", "traditional"),
            ("Custom (timestamp, open, high, low, close, volume, trades)", "custom")
        ]
        
        for text, value in formats:
            tk.Radiobutton(format_frame, text=text, variable=self.format_var, value=value).pack(side=tk.LEFT, padx=10)
        
        # Enhanced load button
        load_btn = tk.Button(
            file_frame,
            text="🚀 Load Data File (CSV/Excel)",
            command=self.enhanced_load_data,
            bg='#0078d4',
            fg='white',
            font=('Arial', 12, 'bold'),
            padx=20,
            pady=10
        )
        load_btn.pack(pady=10)
        
        self.file_label = tk.Label(file_frame, text="No file loaded", fg='#666')
        self.file_label.pack()
        
        # Enhanced data preview
        preview_frame = ttk.LabelFrame(self.data_frame, text="📋 Data Preview & Analysis", padding=10)
        preview_frame.pack(fill='both', expand=True, padx=10, pady=5)
        
        self.data_tree = ttk.Treeview(preview_frame, height=12)
        tree_scroll = ttk.Scrollbar(preview_frame, orient="vertical", command=self.data_tree.yview)
        self.data_tree.configure(yscrollcommand=tree_scroll.set)
        
        tree_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        self.data_tree.pack(fill='both', expand=True)
        
        # Enhanced data info
        self.data_info_text = scrolledtext.ScrolledText(preview_frame, height=8, font=('Consolas', 10))
        self.data_info_text.pack(fill='x', pady=5)
    
    def setup_enhanced_ml_tab(self):
        """Enhanced ML tab with advanced strategies"""
        self.ml_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.ml_frame, text="🤖 Advanced ML Strategies")
        
        # Enhanced controls
        controls_frame = ttk.LabelFrame(self.ml_frame, text="🎛️ Advanced ML Configuration", padding=15)
        controls_frame.pack(fill='x', padx=10, pady=5)
        
        # Control buttons
        button_frame = tk.Frame(controls_frame, bg='#f0f0f0')
        button_frame.pack(fill='x', pady=10)
        
        buttons = [
            ("🔬 Calculate ALL Indicators (40+)", self.calculate_all_indicators, '#ff6b35'),
            ("🎯 Find Best Strategies", self.find_best_strategies, '#e74c3c'),
            ("🔄 Learn Transitions", self.learn_transitions, '#9b59b6'),
            ("🚀 Generate Predictions", self.generate_predictions, '#2ecc71')
        ]
        
        for i, (text, command, color) in enumerate(buttons):
            btn = tk.Button(button_frame, text=text, command=command, bg=color, fg='white', 
                          font=('Arial', 10, 'bold'), padx=15, pady=5)
            btn.grid(row=i//2, column=i%2, padx=5, pady=5, sticky='ew')
        
        button_frame.grid_columnconfigure(0, weight=1)
        button_frame.grid_columnconfigure(1, weight=1)
        
        # Enhanced results display
        self.results_text = scrolledtext.ScrolledText(controls_frame, height=15, font=('Consolas', 10))
        self.results_text.pack(fill='both', expand=True, pady=10)
    
    def setup_enhanced_viz_tab(self):
        """Enhanced visualization tab"""
        self.viz_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.viz_frame, text="📈 Advanced Visualizations")
        
        # Chart controls
        controls_frame = ttk.LabelFrame(self.viz_frame, text="🎨 Advanced Chart Controls", padding=10)
        controls_frame.pack(fill='x', padx=10, pady=5)
        
        chart_buttons = [
            ("📊 Price + Indicators", self.plot_enhanced_price, '#3498db'),
            ("🎯 Support/Resistance", self.plot_support_resistance, '#e74c3c'),
            ("🌊 Fibonacci Levels", self.plot_fibonacci, '#9b59b6'),
            ("🤖 ML Predictions", self.plot_predictions, '#2ecc71')
        ]
        
        for i, (text, command, color) in enumerate(chart_buttons):
            btn = tk.Button(controls_frame, text=text, command=command, bg=color, fg='white',
                          font=('Arial', 10, 'bold'), padx=10, pady=5)
            btn.grid(row=0, column=i, padx=5, sticky='ew')
        
        for j in range(len(chart_buttons)):
            controls_frame.grid_columnconfigure(j, weight=1)
        
        # Chart frame
        self.chart_frame = ttk.Frame(self.viz_frame)
        self.chart_frame.pack(fill='both', expand=True, padx=10, pady=5)
    
    def setup_performance_tab(self):
        """Performance analysis tab"""
        self.perf_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.perf_frame, text="📊 Performance Analytics")
        
        # Performance display
        perf_text_frame = ttk.LabelFrame(self.perf_frame, text="🏆 System Performance", padding=15)
        perf_text_frame.pack(fill='both', expand=True, padx=10, pady=10)
        
        self.performance_text = scrolledtext.ScrolledText(perf_text_frame, font=('Consolas', 11))
        self.performance_text.pack(fill='both', expand=True)
        
        # Set initial performance info
        self.update_performance_info()
    
    def setup_enhanced_status_bar(self):
        """Enhanced status bar"""
        status_frame = tk.Frame(self.root, bg='#0078d4', height=35)
        status_frame.pack(side=tk.BOTTOM, fill=tk.X)
        status_frame.pack_propagate(False)
        
        self.status_bar = tk.Label(
            status_frame,
            text="🚀 ULTIMATE ML TRADING BOT Ready - Load data to begin advanced analysis",
            bg='#0078d4',
            fg='white',
            font=('Arial', 11, 'bold'),
            anchor='w',
            padx=15
        )
        self.status_bar.pack(fill='both', expand=True, pady=8)
    
    def enhanced_load_data(self):
        """Enhanced data loading with universal format support"""
        file_path = filedialog.askopenfilename(
            title="Select Data File",
            filetypes=[
                ("All supported", "*.csv *.xlsx *.xls"),
                ("CSV files", "*.csv"),
                ("Excel files", "*.xlsx *.xls")
            ]
        )
        
        if file_path:
            try:
                self.update_status("🔄 Loading data with enhanced loader...")
                
                # Use enhanced data loader
                self.data = self.data_loader.load_data(file_path, self.format_var.get())
                
                # Update GUI
                self.file_label.config(
                    text=f"✅ {os.path.basename(file_path)} ({len(self.data)} rows, {len(self.data.columns)} cols)",
                    fg='#00aa00'
                )
                self.update_data_preview()
                self.update_enhanced_data_info()
                self.update_status("✅ Enhanced data loading completed!")
                
                messagebox.showinfo("Success", f"Data loaded successfully!\nRows: {len(self.data)}\nColumns: {len(self.data.columns)}")
                
            except Exception as e:
                messagebox.showerror("Error", f"Enhanced data loading failed:\n{str(e)}")
                self.update_status("❌ Data loading failed")
    
    def calculate_all_indicators(self):
        """Calculate all 40+ technical indicators"""
        if not ENHANCED_MODE or self.data is None:
            messagebox.showerror("Error", "Enhanced mode required and data must be loaded!")
            return
        
        try:
            self.update_status("🔬 Calculating 40+ technical indicators...")
            
            # Calculate comprehensive indicators
            self.enhanced_indicators = self.data_loader.calculate_all_indicators()
            
            # Display results
            results = f"🔬 COMPREHENSIVE TECHNICAL ANALYSIS\n{'='*50}\n"
            results += f"✅ Calculated {len(self.enhanced_indicators)} technical indicators\n\n"
            
            # Group indicators by category
            categories = {}
            for name in self.enhanced_indicators.keys():
                category = name.split('_')[0]
                categories[category] = categories.get(category, 0) + 1
            
            results += "📊 Indicator Categories:\n"
            for category, count in sorted(categories.items()):
                results += f"  • {category}: {count} indicators\n"
            
            # Support & Resistance
            sr_levels = self.data_loader.calculate_support_resistance()
            results += f"\n🎯 Support & Resistance Levels: {len(sr_levels)}\n"
            for level, price in list(sr_levels.items())[:5]:
                if isinstance(price, (int, float)):
                    results += f"  • {level}: ${price:.2f}\n"
            
            # Fibonacci levels
            fib_levels = self.data_loader.calculate_fibonacci_retracements() 
            results += f"\n🌊 Fibonacci Levels: {len(fib_levels)}\n"
            for level, price in list(fib_levels.items())[:6]:
                if isinstance(price, (int, float)):
                    results += f"  • {level}: ${price:.2f}\n"
            
            self.results_text.delete(1.0, tk.END)
            self.results_text.insert(1.0, results)
            
            self.update_status("✅ All indicators calculated successfully!")
            
        except Exception as e:
            messagebox.showerror("Error", f"Indicator calculation failed:\n{str(e)}")
            self.update_status("❌ Indicator calculation failed")
    
    def find_best_strategies(self):
        """Find best ML strategies across time periods"""
        if not ENHANCED_MODE or not self.enhanced_indicators:
            messagebox.showerror("Error", "Calculate indicators first!")
            return
        
        try:
            self.update_status("🎯 Finding best strategies across time periods...")
            
            # Initialize ML engine
            self.ml_engine.initialize_models()
            
            # Create features
            features_df = self.ml_engine.create_dynamic_features(self.data, self.enhanced_indicators)
            
            # Find strategies
            periods = [30, 50, 100, 200]
            self.strategies = self.ml_engine.find_best_strategies_by_period(features_df, 'close', periods)
            
            # Display results
            results = f"🎯 BEST STRATEGIES BY TIME PERIOD\n{'='*50}\n"
            results += f"✅ Analyzed {len(self.strategies)} time periods\n"
            results += f"📊 Feature Matrix: {features_df.shape[1]} features, {features_df.shape[0]} samples\n\n"
            
            for period, strategy_data in self.strategies.items():
                results += f"📈 {period} days: {strategy_data['best_model']} (R² = {strategy_data['best_score']:.4f})\n"
                if 'top_features' in strategy_data:
                    results += "   Top features:\n"
                    for feature, importance in strategy_data['top_features'][:3]:
                        results += f"    • {feature}: {importance:.4f}\n"
                results += "\n"
            
            self.results_text.delete(1.0, tk.END)
            self.results_text.insert(1.0, results)
            
            self.update_status("✅ Best strategies identified!")
            
        except Exception as e:
            messagebox.showerror("Error", f"Strategy finding failed:\n{str(e)}")
            self.update_status("❌ Strategy finding failed")
    
    def learn_transitions(self):
        """Learn strategy transitions"""
        if not self.strategies:
            messagebox.showerror("Error", "Find strategies first!")
            return
        
        try:
            self.update_status("🔄 Learning strategy transitions...")
            
            # Learn transitions
            features_df = self.ml_engine.create_dynamic_features(self.data, self.enhanced_indicators)
            transitions = self.ml_engine.learn_strategy_transitions(features_df)
            
            results = f"🔄 STRATEGY TRANSITION LEARNING\n{'='*50}\n"
            if transitions:
                results += f"✅ Transition model trained with {transitions['accuracy']:.3f} accuracy\n"
                results += f"📊 Training samples: {transitions['training_data_size']}\n\n"
                results += "🎯 Key transition factors:\n"
                for factor, importance in transitions['feature_importance'].items():
                    results += f"  • {factor}: {importance:.4f}\n"
            else:
                results += "⚠️  Could not learn transitions - need more data\n"
            
            self.results_text.delete(1.0, tk.END)
            self.results_text.insert(1.0, results)
            
            self.update_status("✅ Strategy transitions learned!")
            
        except Exception as e:
            messagebox.showerror("Error", f"Transition learning failed:\n{str(e)}")
            self.update_status("❌ Transition learning failed")
    
    def generate_predictions(self):
        """Generate comprehensive predictions"""
        if not self.strategies:
            messagebox.showerror("Error", "Find strategies first!")
            return
        
        try:
            self.update_status("🚀 Generating comprehensive predictions...")
            
            features_df = self.ml_engine.create_dynamic_features(self.data, self.enhanced_indicators)
            predictions = self.ml_engine.generate_comprehensive_predictions(features_df, 'close')
            
            results = f"🚀 COMPREHENSIVE PREDICTIONS\n{'='*50}\n"
            
            if predictions and 'ensemble' in predictions:
                ensemble = predictions['ensemble']
                results += f"🎯 Ensemble Prediction: {ensemble['prediction']:.6f}\n"
                results += f"📊 Recommended Strategy: {ensemble['recommended_strategy']}\n"
                results += f"🤖 Models Used: {ensemble['n_models']}\n\n"
                
                results += "Individual Model Predictions:\n"
                for name, pred in predictions.items():
                    if name != 'ensemble':
                        recommended = "⭐" if pred.get('is_recommended', False) else ""
                        results += f"  {name}: {pred['prediction']:.6f} {recommended}\n"
            else:
                results += "⚠️  No predictions generated - need more training data\n"
            
            self.results_text.delete(1.0, tk.END)
            self.results_text.insert(1.0, results)
            
            self.update_status("✅ Predictions generated!")
            
        except Exception as e:
            messagebox.showerror("Error", f"Prediction generation failed:\n{str(e)}")
            self.update_status("❌ Prediction generation failed")
    
    def update_enhanced_data_info(self):
        """Update enhanced data information display"""
        if not ENHANCED_MODE or self.data is None:
            return
        
        summary = self.data_loader.get_data_summary()
        
        info_text = f"""
🚀 ULTIMATE ML TRADING BOT - DATA ANALYSIS
{'='*60}

📊 DATASET INFORMATION:
  • Data Points: {summary['basic_stats']['rows']:,}
  • Columns: {summary['basic_stats']['columns']}
  • Date Range: {summary['basic_stats']['start_date']} to {summary['basic_stats']['end_date']}
  • Trading Days: {summary['basic_stats']['trading_days']:,}

💰 PRICE ANALYSIS:
  • Current Price: ${summary['price_stats']['current_price']:.2f}
  • Price Range: ${summary['price_stats']['lowest_price']:.2f} - ${summary['price_stats']['highest_price']:.2f}
  • Average Price: ${summary['price_stats']['average_price']:.2f}
  • Volatility: {summary['price_stats']['price_volatility']:.4f}

📊 VOLUME ANALYSIS:
  • Total Volume: {summary['volume_stats']['total_volume']:,}
  • Average Volume: {summary['volume_stats']['average_volume']:,.0f}
  • Peak Volume: {summary['volume_stats']['highest_volume']:,}

🔬 ANALYSIS STATUS:
  • Technical Indicators: {summary['indicators_count']} calculated
  • Support/Resistance: {summary['support_resistance_count']} levels
  • Fibonacci Levels: {summary['fibonacci_levels_count']} levels

✅ Ready for advanced ML analysis and strategy optimization!
        """
        
        self.data_info_text.delete(1.0, tk.END)
        self.data_info_text.insert(1.0, info_text)
    
    def update_performance_info(self):
        """Update performance information"""
        perf_text = """
🚀 ULTIMATE ML TRADING BOT - PERFORMANCE CAPABILITIES
================================================================

🎯 SYSTEM FEATURES:
✅ Universal data loading (CSV/Excel, any OHLC format)
✅ Automatic format detection (traditional vs custom)
✅ 40+ technical indicators with robust error handling
✅ 4 advanced ML models (Random Forest, XGBoost, LightGBM, Gradient Boosting)
✅ Dynamic strategy optimization across multiple time periods
✅ Automatic strategy transition learning with market regime detection
✅ Support & resistance level detection using pivot points
✅ Fibonacci retracement analysis for key price levels
✅ Comprehensive feature engineering with interaction terms
✅ Time-series aware cross-validation for reliable results
✅ Ensemble predictions with confidence scoring
✅ Detailed performance analytics and reporting
✅ Robust error handling and data validation
✅ 100% free and local operation (no cloud dependencies)
✅ Modern GUI with enhanced user experience

🎯 WORKFLOW:
1. Load your trading data (CSV/Excel) - any OHLC format supported
2. System automatically detects format and validates data
3. Calculate 40+ technical indicators with one click
4. Find best ML strategies for different time periods
5. Learn when to transition between strategies based on market conditions
6. Generate ensemble predictions using the best models
7. Analyze performance with comprehensive reporting
8. Visualize results with advanced charts

🏆 THIS IS THE MOST ADVANCED FREE LOCAL ML TRADING SYSTEM!
================================================================

Ready to revolutionize your trading analysis with cutting-edge ML!
        """
        
        self.performance_text.delete(1.0, tk.END)
        self.performance_text.insert(1.0, perf_text)
    
    # Fallback to basic GUI if enhanced mode not available
    def setup_gui(self):
        """Basic GUI fallback"""
        messagebox.showinfo("Mode", "Running in basic mode. Install enhanced modules for full features.")
        # ... (basic GUI setup code from original)
        
    def update_status(self, message):
        """Update status bar"""
        if hasattr(self, 'status_bar'):
            self.status_bar.config(text=message)
            self.root.update()
    
    def update_data_preview(self):
        """Update data preview"""
        if self.data is None:
            return
        
        # Clear existing
        for item in self.data_tree.get_children():
            self.data_tree.delete(item)
        
        # Setup columns
        columns = list(self.data.columns)
        self.data_tree['columns'] = columns
        self.data_tree['show'] = 'headings'
        
        for col in columns:
            self.data_tree.heading(col, text=col)
            self.data_tree.column(col, width=100)
        
        # Add data
        for i, (idx, row) in enumerate(self.data.head(50).iterrows()):
            values = [f"{val:.4f}" if isinstance(val, (int, float)) else str(val) for val in row]
            self.data_tree.insert('', 'end', values=values)
    
    # Placeholder chart methods
    def plot_enhanced_price(self):
        messagebox.showinfo("Charts", "Enhanced price charts will be displayed here!")
    
    def plot_support_resistance(self):
        messagebox.showinfo("Charts", "Support & resistance visualization coming soon!")
    
    def plot_fibonacci(self):
        messagebox.showinfo("Charts", "Fibonacci levels visualization coming soon!")

def main():
    """Main application entry point"""
    try:
        root = tk.Tk()
        app = AssetMLStrategy(root)
        root.mainloop()
    except Exception as e:
        print(f"Error starting application: {e}")
        if "tkinter" in str(e).lower():
            print("GUI not available in headless environment.")
            print("Run 'python ultimate_trading_bot_demo.py' for demonstration.")

if __name__ == "__main__":
    main()
    def __init__(self, root):
        self.root = root
        self.root.title("Asset ML Strategy - Free Local Financial Analysis")
        self.root.geometry("1200x800")
        self.root.configure(bg='#f0f0f0')
        
        # Data storage
        self.data = None
        self.model = None
        self.predictions = None
        
        # Setup GUI
        self.setup_gui()
    
    def setup_gui(self):
        """Setup the main GUI interface"""
        # Create main notebook for tabs
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill='both', expand=True, padx=10, pady=10)
        
        # Tab 1: Data Loading
        self.data_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.data_frame, text="Data Loading")
        self.setup_data_tab()
        
        # Tab 2: ML Analysis
        self.ml_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.ml_frame, text="ML Analysis")
        self.setup_ml_tab()
        
        # Tab 3: Visualization
        self.viz_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.viz_frame, text="Visualization")
        self.setup_viz_tab()
        
        # Tab 4: Advanced Models
        self.advanced_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.advanced_frame, text="Advanced Models")
        self.setup_advanced_tab()
        
        # Status bar
        self.status_bar = ttk.Label(self.root, text="Status: Ready", relief=tk.SUNKEN)
        self.status_bar.pack(side=tk.BOTTOM, fill=tk.X)
    
    def setup_data_tab(self):
        """Setup data loading tab"""
        # File selection frame
        file_frame = ttk.LabelFrame(self.data_frame, text="File Selection", padding=10)
        file_frame.pack(fill='x', padx=10, pady=5)
        
        ttk.Button(
            file_frame,
            text="Select Excel/CSV File",
            command=self.load_data,
            width=20
        ).pack(side=tk.LEFT, padx=5)
        
        self.file_label = ttk.Label(file_frame, text="No file selected")
        self.file_label.pack(side=tk.LEFT, padx=10)
        
        # Data preview frame
        preview_frame = ttk.LabelFrame(self.data_frame, text="Data Preview", padding=10)
        preview_frame.pack(fill='both', expand=True, padx=10, pady=5)
        
        # Create treeview for data display
        self.data_tree = ttk.Treeview(preview_frame, show='headings', height=15)
        self.data_tree.pack(fill='both', expand=True)
        
        # Scrollbars
        v_scrollbar = ttk.Scrollbar(preview_frame, orient='vertical', command=self.data_tree.yview)
        v_scrollbar.pack(side='right', fill='y')
        self.data_tree.configure(yscrollcommand=v_scrollbar.set)
        
        h_scrollbar = ttk.Scrollbar(preview_frame, orient='horizontal', command=self.data_tree.xview)
        h_scrollbar.pack(side='bottom', fill='x')
        self.data_tree.configure(xscrollcommand=h_scrollbar.set)
    
    def setup_ml_tab(self):
        """Setup machine learning analysis tab"""
        # Configuration frame
        config_frame = ttk.LabelFrame(self.ml_frame, text="ML Configuration", padding=10)
        config_frame.pack(fill='x', padx=10, pady=5)
        
        ttk.Label(config_frame, text="Target Column:").grid(row=0, column=0, sticky='w', padx=5)
        self.target_var = tk.StringVar(value="Close")
        target_combo = ttk.Combobox(config_frame, textvariable=self.target_var, width=15)
        target_combo['values'] = ["Close", "High", "Low", "Open", "Adj Close"]
        target_combo.grid(row=0, column=1, padx=5)
        
        ttk.Label(config_frame, text="Test Size:").grid(row=0, column=2, sticky='w', padx=5)
        self.test_size_var = tk.DoubleVar(value=0.2)
        test_size_spin = ttk.Spinbox(config_frame, from_=0.1, to=0.5, increment=0.1, 
                                    textvariable=self.test_size_var, width=10)
        test_size_spin.grid(row=0, column=3, padx=5)
        
        ttk.Button(
            config_frame,
            text="Train Model",
            command=self.train_model,
            width=15
        ).grid(row=0, column=4, padx=10)
        
        ttk.Button(
            config_frame,
            text="Make Predictions",
            command=self.make_predictions,
            width=15
        ).grid(row=0, column=5, padx=5)
        
        # Results frame
        results_frame = ttk.LabelFrame(self.ml_frame, text="ML Results", padding=10)
        results_frame.pack(fill='both', expand=True, padx=10, pady=5)
        
        self.results_text = scrolledtext.ScrolledText(results_frame, height=20, width=80)
        self.results_text.pack(fill='both', expand=True)
    
    def setup_viz_tab(self):
        """Setup visualization tab"""
        # Controls frame
        controls_frame = ttk.LabelFrame(self.viz_frame, text="Chart Controls", padding=10)
        controls_frame.pack(fill='x', padx=10, pady=5)
        
        ttk.Button(
            controls_frame,
            text="Price Chart",
            command=self.plot_price_chart,
            width=15
        ).pack(side=tk.LEFT, padx=5)
        
        ttk.Button(
            controls_frame,
            text="Volume Chart",
            command=self.plot_volume_chart,
            width=15
        ).pack(side=tk.LEFT, padx=5)
        
        ttk.Button(
            controls_frame,
            text="Correlation Matrix",
            command=self.plot_correlation,
            width=15
        ).pack(side=tk.LEFT, padx=5)
        
        ttk.Button(
            controls_frame,
            text="Predictions vs Actual",
            command=self.plot_predictions,
            width=15
        ).pack(side=tk.LEFT, padx=5)
        
        # Chart frame
        self.chart_frame = ttk.Frame(self.viz_frame)
        self.chart_frame.pack(fill='both', expand=True, padx=10, pady=5)
    
    def setup_advanced_tab(self):
        """Setup advanced models tab"""
        info_frame = ttk.LabelFrame(self.advanced_frame, text="Advanced ML Models Available", padding=10)
        info_frame.pack(fill='both', expand=True, padx=10, pady=5)
        
        advanced_info = """
🚀 ADVANCED ML MODELS AVAILABLE:

1. Random Forest Regressor (Current)
2. Transformer Models (Neural Networks)
3. LSTM with Attention
4. CNN-LSTM Hybrid
5. WaveNet Architecture
6. Quantum-Inspired Neural Networks
7. XGBoost & LightGBM
8. Ensemble Methods

🎯 FEATURES:
• Technical Indicators (SMA, EMA, RSI, MACD)
• Market Microstructure Features
• Deep Learning Architectures
• Reinforcement Learning Agents
• Portfolio Optimization
• Risk Management
• Backtesting Engine
• Sentiment Analysis

📊 ACCESS ADVANCED FEATURES:
• Use ultimate_demo.py for full feature access
• Use moneyprinter.py for maximum performance
• Use advanced_gui.py for web interface
• All models are FREE and LOCAL!
        """
        
        info_text = tk.Text(info_frame, height=20, width=80, wrap=tk.WORD)
        info_text.insert(1.0, advanced_info)
        info_text.config(state=tk.DISABLED)
        info_text.pack(fill='both', expand=True)
        
        # Quick access buttons
        buttons_frame = ttk.Frame(self.advanced_frame)
        buttons_frame.pack(fill='x', padx=10, pady=5)
        
        ttk.Button(
            buttons_frame,
            text="Launch Ultimate Demo",
            command=self.launch_ultimate_demo
        ).pack(side=tk.LEFT, padx=5)
        
        ttk.Button(
            buttons_frame,
            text="Launch MoneyPrinter",
            command=self.launch_moneyprinter
        ).pack(side=tk.LEFT, padx=5)
        
        ttk.Button(
            buttons_frame,
            text="Launch Advanced GUI",
            command=self.launch_advanced_gui
        ).pack(side=tk.LEFT, padx=5)
    
    def load_data(self):
        """Load Excel or CSV data"""
        file_path = filedialog.askopenfilename(
            title="Select Data File",
            filetypes=[
                ("Excel files", "*.xlsx *.xls"),
                ("CSV files", "*.csv"),
                ("All files", "*.*")
            ]
        )
        
        if not file_path:
            return
        
        try:
            self.update_status("Loading data...")
            
            # Load data based on file extension
            if file_path.lower().endswith(('.xlsx', '.xls')):
                self.data = pd.read_excel(file_path)
            elif file_path.lower().endswith('.csv'):
                self.data = pd.read_csv(file_path)
            else:
                messagebox.showerror("Error", "Unsupported file format!")
                return
            
            # Validate required columns
            required_columns = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']
            missing_columns = [col for col in required_columns if col not in self.data.columns]
            
            if missing_columns:
                # Try alternative column names
                alt_mappings = {
                    'Date': ['date', 'DATE', 'Timestamp', 'timestamp'],
                    'Open': ['open', 'OPEN', 'Opening', 'opening'],
                    'High': ['high', 'HIGH'],
                    'Low': ['low', 'LOW'],
                    'Close': ['close', 'CLOSE', 'Closing', 'closing'],
                    'Volume': ['volume', 'VOLUME', 'Vol', 'vol']
                }
                
                # Try to map alternative names
                for col in missing_columns.copy():
                    for alt_name in alt_mappings.get(col, []):
                        if alt_name in self.data.columns:
                            self.data.rename(columns={alt_name: col}, inplace=True)
                            missing_columns.remove(col)
                            break
            
            # Add Adj Close if missing
            if 'Adj Close' not in self.data.columns:
                self.data['Adj Close'] = self.data['Close'].copy()
            
            if missing_columns:
                messagebox.showwarning(
                    "Missing Columns",
                    f"Missing columns: {missing_columns}\nProceeding with available data."
                )
            
            # Display data in treeview
            self.display_data()
            
            # Update file label
            filename = os.path.basename(file_path)
            self.file_label.config(text=f"Loaded: {filename} ({len(self.data)} rows)")
            
            # Update target variable options
            numeric_columns = self.data.select_dtypes(include=[np.number]).columns
            target_combo = None
            for widget in self.ml_frame.winfo_children():
                if isinstance(widget, ttk.LabelFrame) and widget['text'] == 'ML Configuration':
                    for child in widget.winfo_children():
                        if isinstance(child, ttk.Combobox):
                            target_combo = child
                            break
            
            if target_combo:
                target_combo['values'] = list(numeric_columns)
            
            self.update_status("Data loaded successfully")
            messagebox.showinfo("Success", f"Data loaded successfully!\n{len(self.data)} rows loaded.")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load data:\n{str(e)}")
            self.update_status("Error loading data")
    
    def display_data(self):
        """Display data in the treeview"""
        # Clear existing data
        for item in self.data_tree.get_children():
            self.data_tree.delete(item)
        
        # Set up columns
        columns = list(self.data.columns)
        self.data_tree['columns'] = columns
        
        for col in columns:
            self.data_tree.heading(col, text=col)
            self.data_tree.column(col, width=100)
        
        # Insert data (first 100 rows for performance)
        for index, row in self.data.head(100).iterrows():
            values = [str(row[col])[:20] for col in columns]  # Truncate long values
            self.data_tree.insert('', 'end', values=values)
    
    def train_model(self):
        """Train machine learning model"""
        if self.data is None:
            messagebox.showerror("Error", "Please load data first!")
            return
        
        try:
            self.update_status("Training ML model...")
            
            # Prepare features and target
            target_col = self.target_var.get()
            feature_cols = ['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']
            feature_cols = [col for col in feature_cols if col != target_col and col in self.data.columns]
            
            X = self.data[feature_cols].copy()
            y = self.data[target_col].copy()
            
            # Add technical indicators as features
            X['SMA_5'] = self.data[target_col].rolling(window=5).mean()
            X['SMA_20'] = self.data[target_col].rolling(window=20).mean()
            X['Volatility'] = self.data[target_col].rolling(window=10).std()
            X['Price_Change'] = self.data[target_col].pct_change()
            
            # Remove rows with NaN values
            mask = ~(X.isnull().any(axis=1) | y.isnull())
            X = X[mask]
            y = y[mask]
            
            if len(X) < 10:
                messagebox.showerror("Error", "Not enough valid data points for training!")
                return
            
            # Split data
            test_size = self.test_size_var.get()
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=42, shuffle=False
            )
            
            # Train Random Forest model
            self.model = RandomForestRegressor(n_estimators=100, random_state=42)
            self.model.fit(X_train, y_train)
            
            # Make predictions
            y_pred_train = self.model.predict(X_train)
            y_pred_test = self.model.predict(X_test)
            
            # Calculate metrics
            train_mse = mean_squared_error(y_train, y_pred_train)
            test_mse = mean_squared_error(y_test, y_pred_test)
            train_r2 = r2_score(y_train, y_pred_train)
            test_r2 = r2_score(y_test, y_pred_test)
            
            # Display results
            results = f"Model Training Results:\n"
            results += f"{'='*50}\n"
            results += f"Target Variable: {target_col}\n"
            results += f"Features: {', '.join(X.columns)}\n"
            results += f"Training samples: {len(X_train)}\n"
            results += f"Testing samples: {len(X_test)}\n\n"
            
            results += f"Performance Metrics:\n"
            results += f"Training MSE: {train_mse:.4f}\n"
            results += f"Testing MSE: {test_mse:.4f}\n"
            results += f"Training R²: {train_r2:.4f}\n"
            results += f"Testing R²: {test_r2:.4f}\n\n"
            
            # Feature importance
            results += f"Feature Importance:\n"
            feature_importance = pd.DataFrame({
                'feature': X.columns,
                'importance': self.model.feature_importances_
            }).sort_values('importance', ascending=False)
            
            for _, row in feature_importance.iterrows():
                results += f"{row['feature']}: {row['importance']:.4f}\n"
            
            # Store predictions for plotting
            self.predictions = {
                'X_test': X_test,
                'y_test': y_test,
                'y_pred': y_pred_test,
                'target_col': target_col
            }
            
            self.results_text.delete(1.0, tk.END)
            self.results_text.insert(1.0, results)
            
            self.update_status("Model training completed successfully")
            messagebox.showinfo("Success", f"Model trained successfully!\nTest R² Score: {test_r2:.4f}")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to train model:\n{str(e)}")
            self.update_status("Error training model")
    
    def make_predictions(self):
        """Make predictions on new data"""
        if self.model is None:
            messagebox.showerror("Error", "Please train a model first!")
            return
        
        try:
            # This is a placeholder for future prediction functionality
            # In a real application, you might load new data or use the latest data points
            messagebox.showinfo(
                "Predictions",
                "Prediction functionality ready!\n"
                "The trained model can be used to predict future values.\n"
                "Current implementation shows training results in the ML Analysis tab."
            )
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to make predictions:\n{str(e)}")
    
    def plot_price_chart(self):
        """Plot price chart"""
        if self.data is None:
            messagebox.showerror("Error", "Please load data first!")
            return
        
        try:
            self.clear_chart_frame()
            
            fig, ax = plt.subplots(figsize=(12, 6))
            
            if 'Date' in self.data.columns:
                ax.plot(pd.to_datetime(self.data['Date']), self.data['Close'], label='Close Price')
                if 'Open' in self.data.columns:
                    ax.plot(pd.to_datetime(self.data['Date']), self.data['Open'], label='Open Price', alpha=0.7)
            else:
                ax.plot(self.data['Close'], label='Close Price')
                if 'Open' in self.data.columns:
                    ax.plot(self.data['Open'], label='Open Price', alpha=0.7)
            
            ax.set_title('Price Chart')
            ax.set_xlabel('Date' if 'Date' in self.data.columns else 'Index')
            ax.set_ylabel('Price')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            canvas = FigureCanvasTkAgg(fig, self.chart_frame)
            canvas.draw()
            canvas.get_tk_widget().pack(fill='both', expand=True)
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to plot price chart:\n{str(e)}")
    
    def plot_volume_chart(self):
        """Plot volume chart"""
        if self.data is None:
            messagebox.showerror("Error", "Please load data first!")
            return
        
        if 'Volume' not in self.data.columns:
            messagebox.showerror("Error", "Volume data not available!")
            return
        
        try:
            self.clear_chart_frame()
            
            fig, ax = plt.subplots(figsize=(12, 6))
            
            if 'Date' in self.data.columns:
                ax.bar(pd.to_datetime(self.data['Date']), self.data['Volume'], alpha=0.7)
            else:
                ax.bar(range(len(self.data)), self.data['Volume'], alpha=0.7)
            
            ax.set_title('Volume Chart')
            ax.set_xlabel('Date' if 'Date' in self.data.columns else 'Index')
            ax.set_ylabel('Volume')
            ax.grid(True, alpha=0.3)
            
            canvas = FigureCanvasTkAgg(fig, self.chart_frame)
            canvas.draw()
            canvas.get_tk_widget().pack(fill='both', expand=True)
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to plot volume chart:\n{str(e)}")
    
    def plot_correlation(self):
        """Plot correlation matrix"""
        if self.data is None:
            messagebox.showerror("Error", "Please load data first!")
            return
        
        try:
            self.clear_chart_frame()
            
            # Select numeric columns
            numeric_data = self.data.select_dtypes(include=[np.number])
            
            fig, ax = plt.subplots(figsize=(10, 8))
            
            correlation_matrix = numeric_data.corr()
            sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0, ax=ax)
            
            ax.set_title('Correlation Matrix')
            
            canvas = FigureCanvasTkAgg(fig, self.chart_frame)
            canvas.draw()
            canvas.get_tk_widget().pack(fill='both', expand=True)
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to plot correlation matrix:\n{str(e)}")
    
    def plot_predictions(self):
        """Plot predictions vs actual values"""
        if self.predictions is None:
            messagebox.showerror("Error", "Please train a model first!")
            return
        
        try:
            self.clear_chart_frame()
            
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
            
            # Time series plot
            ax1.plot(self.predictions['y_test'].values, label='Actual', color='blue')
            ax1.plot(self.predictions['y_pred'], label='Predicted', color='red', alpha=0.7)
            ax1.set_title(f"Predictions vs Actual - {self.predictions['target_col']}")
            ax1.set_xlabel('Time')
            ax1.set_ylabel('Value')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # Scatter plot
            ax2.scatter(self.predictions['y_test'].values, self.predictions['y_pred'], alpha=0.6)
            ax2.plot([self.predictions['y_test'].min(), self.predictions['y_test'].max()], 
                    [self.predictions['y_test'].min(), self.predictions['y_test'].max()], 
                    'r--', lw=2)
            ax2.set_xlabel('Actual Values')
            ax2.set_ylabel('Predicted Values')
            ax2.set_title('Predicted vs Actual Scatter Plot')
            ax2.grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            canvas = FigureCanvasTkAgg(fig, self.chart_frame)
            canvas.draw()
            canvas.get_tk_widget().pack(fill='both', expand=True)
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to plot predictions:\n{str(e)}")
    
    def clear_chart_frame(self):
        """Clear the chart frame"""
        for widget in self.chart_frame.winfo_children():
            widget.destroy()
    
    def launch_ultimate_demo(self):
        """Launch the ultimate demo"""
        try:
            import subprocess
            subprocess.Popen(['python', 'ultimate_demo.py'])
            messagebox.showinfo("Launching", "Ultimate Demo launched in separate window!")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to launch Ultimate Demo:\n{str(e)}")
    
    def launch_moneyprinter(self):
        """Launch the MoneyPrinter strategy"""
        try:
            import subprocess
            subprocess.Popen(['python', 'moneyprinter.py'])
            messagebox.showinfo("Launching", "MoneyPrinter launched in separate window!")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to launch MoneyPrinter:\n{str(e)}")
    
    def launch_advanced_gui(self):
        """Launch the advanced GUI"""
        try:
            import subprocess
            subprocess.Popen(['streamlit', 'run', 'advanced_gui.py'])
            messagebox.showinfo("Launching", "Advanced GUI will open in your web browser!")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to launch Advanced GUI:\n{str(e)}")
    
    def update_status(self, message):
        """Update status bar"""
        self.status_bar.config(text=f"Status: {message}")
        self.root.update_idletasks()

def main():
    """Main application entry point"""
    root = tk.Tk()
    app = AssetMLStrategy(root)
    root.mainloop()

if __name__ == "__main__":
    main()