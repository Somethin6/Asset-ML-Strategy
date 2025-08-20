#!/usr/bin/env python3
"""
🚀 ULTIMATE ADVANCED GUI 🚀
The most advanced trading GUI ever created with 143+ features and 16+ ML models!
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

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO)

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

class UltimateAdvancedGUI:
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
        
        # Setup GUI
        self.setup_gui()
        self.show_welcome()
    
    def show_welcome(self):
        """Show welcome message"""
        mode = "ULTIMATE ADVANCED" if ADVANCED_MODE else "BASIC"
        features = "143+ Advanced Features & 16+ ML Models" if ADVANCED_MODE else "Basic Features & ML Models"
        
        welcome_msg = f"""
🚀 WELCOME TO THE {mode} ASSET ML STRATEGY! 🚀

{features}

REVOLUTIONARY CAPABILITIES:
{'✅ 143+ Technical & Statistical Features' if ADVANCED_MODE else '✅ Basic Technical Features'}
{'✅ 16+ ML Models with Ensemble Learning' if ADVANCED_MODE else '✅ Random Forest ML Model'}
{'✅ Advanced Signal Generation & Risk Management' if ADVANCED_MODE else '✅ Basic Signal Generation'}
{'✅ Comprehensive Backtesting & Portfolio Optimization' if ADVANCED_MODE else '✅ Basic Performance Metrics'}
✅ Professional GUI Interface
✅ 100% Free & Local Operation

Ready to transform your trading with {'cutting-edge AI technology' if ADVANCED_MODE else 'reliable ML analysis'}!
        """
        messagebox.showinfo("Ultimate Advanced Trading System", welcome_msg)
    
    def setup_gui(self):
        """Setup the GUI"""
        # Header
        header_frame = tk.Frame(self.root, bg='#0d1421', height=80)
        header_frame.pack(fill='x')
        header_frame.pack_propagate(False)
        
        title_text = "🚀 ULTIMATE ADVANCED ASSET ML STRATEGY 🚀" if ADVANCED_MODE else "🔧 ASSET ML STRATEGY"
        subtitle = "Most Advanced Free Local Trading System" if ADVANCED_MODE else "Free Local Trading Analysis"
        
        tk.Label(header_frame, text=title_text, font=('Arial', 20, 'bold'),
                bg='#0d1421', fg='#00ff88').pack(pady=(10, 0))
        tk.Label(header_frame, text=subtitle, font=('Arial', 12),
                bg='#0d1421', fg='#888888').pack()
        
        # Notebook
        style = ttk.Style()
        style.theme_use('clam')
        style.configure('TNotebook', background='#2d2d2d')
        style.configure('TNotebook.Tab', padding=[20, 10])
        
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill='both', expand=True, padx=10, pady=10)
        
        # Setup tabs
        self.setup_data_tab()
        if ADVANCED_MODE:
            self.setup_features_tab()
        self.setup_ml_tab()
        self.setup_signals_tab()
        self.setup_backtest_tab()
        self.setup_viz_tab()
        
        # Status bar
        self.status_frame = tk.Frame(self.root, bg='#0d1421', height=30)
        self.status_frame.pack(side=tk.BOTTOM, fill=tk.X)
        self.status_frame.pack_propagate(False)
        
        self.status_label = tk.Label(self.status_frame, text="🚀 Ultimate Advanced Trading System Ready - Load data to begin!",
                                   bg='#0d1421', fg='#00ff88', font=('Arial', 10), anchor='w', padx=10)
        self.status_label.pack(fill='both', expand=True, pady=5)
    
    def update_status(self, message):
        """Update status bar"""
        self.status_label.config(text=message)
        self.root.update()
    
    def setup_data_tab(self):
        """Data loading tab"""
        self.data_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.data_frame, text="📊 Data Loading")
        
        # File controls
        file_frame = ttk.LabelFrame(self.data_frame, text="📁 Load Trading Data", padding=15)
        file_frame.pack(fill='x', padx=10, pady=5)
        
        ttk.Button(file_frame, text="🚀 Load Data File", command=self.load_data, width=25).pack(side=tk.LEFT, padx=5)
        self.file_label = ttk.Label(file_frame, text="No file loaded")
        self.file_label.pack(side=tk.LEFT, padx=10)
        
        # Data preview
        preview_frame = ttk.LabelFrame(self.data_frame, text="📋 Data Preview", padding=10)
        preview_frame.pack(fill='both', expand=True, padx=10, pady=5)
        
        self.data_tree = ttk.Treeview(preview_frame, height=12)
        ttk.Scrollbar(preview_frame, orient="vertical", command=self.data_tree.yview).pack(side=tk.RIGHT, fill=tk.Y)
        self.data_tree.pack(fill='both', expand=True)
        
        # Data info
        info_frame = ttk.LabelFrame(self.data_frame, text="📊 Data Information", padding=10)
        info_frame.pack(fill='x', padx=10, pady=5)
        
        self.data_info_text = scrolledtext.ScrolledText(info_frame, height=6)
        self.data_info_text.pack(fill='both', expand=True)
    
    def setup_features_tab(self):
        """Advanced features tab"""
        self.features_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.features_frame, text="🔬 Advanced Features")
        
        controls_frame = ttk.LabelFrame(self.features_frame, text="🔬 Feature Engineering", padding=15)
        controls_frame.pack(fill='x', padx=10, pady=5)
        
        ttk.Button(controls_frame, text="🚀 Generate 143+ Features", 
                  command=self.generate_features, width=25).pack(side=tk.LEFT, padx=5)
        self.features_label = ttk.Label(controls_frame, text="No features generated")
        self.features_label.pack(side=tk.LEFT, padx=10)
        
        summary_frame = ttk.LabelFrame(self.features_frame, text="📊 Feature Summary", padding=10)
        summary_frame.pack(fill='both', expand=True, padx=10, pady=5)
        
        self.features_text = scrolledtext.ScrolledText(summary_frame, height=20)
        self.features_text.pack(fill='both', expand=True)
    
    def setup_ml_tab(self):
        """ML training tab"""
        self.ml_frame = ttk.Frame(self.notebook)
        tab_text = "🤖 Advanced ML Ensemble" if ADVANCED_MODE else "🤖 ML Analysis"
        self.notebook.add(self.ml_frame, text=tab_text)
        
        controls_frame = ttk.LabelFrame(self.ml_frame, text="🤖 ML Training", padding=15)
        controls_frame.pack(fill='x', padx=10, pady=5)
        
        button_text = "🚀 Train 16+ ML Models" if ADVANCED_MODE else "🔧 Train ML Model"
        ttk.Button(controls_frame, text=button_text, command=self.train_models, width=25).pack(side=tk.LEFT, padx=5)
        self.ml_label = ttk.Label(controls_frame, text="Models not trained")
        self.ml_label.pack(side=tk.LEFT, padx=10)
        
        results_frame = ttk.LabelFrame(self.ml_frame, text="🏆 ML Results", padding=10)
        results_frame.pack(fill='both', expand=True, padx=10, pady=5)
        
        self.ml_results_text = scrolledtext.ScrolledText(results_frame, height=20)
        self.ml_results_text.pack(fill='both', expand=True)
    
    def setup_signals_tab(self):
        """Signals tab"""
        self.signals_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.signals_frame, text="📈 Trading Signals")
        
        controls_frame = ttk.LabelFrame(self.signals_frame, text="📈 Signal Generation", padding=15)
        controls_frame.pack(fill='x', padx=10, pady=5)
        
        ttk.Button(controls_frame, text="🎯 Generate Trading Signals", 
                  command=self.generate_signals, width=25).pack(side=tk.LEFT, padx=5)
        self.signals_label = ttk.Label(controls_frame, text="No signals generated")
        self.signals_label.pack(side=tk.LEFT, padx=10)
        
        signals_frame = ttk.LabelFrame(self.signals_frame, text="📊 Trading Signals", padding=10)
        signals_frame.pack(fill='both', expand=True, padx=10, pady=5)
        
        self.signals_text = scrolledtext.ScrolledText(signals_frame, height=20)
        self.signals_text.pack(fill='both', expand=True)
    
    def setup_backtest_tab(self):
        """Backtest tab"""
        self.backtest_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.backtest_frame, text="📊 Backtesting")
        
        controls_frame = ttk.LabelFrame(self.backtest_frame, text="📊 Backtesting", padding=15)
        controls_frame.pack(fill='x', padx=10, pady=5)
        
        ttk.Button(controls_frame, text="🚀 Run Advanced Backtest", 
                  command=self.run_backtest, width=25).pack(side=tk.LEFT, padx=5)
        self.backtest_label = ttk.Label(controls_frame, text="Backtest not run")
        self.backtest_label.pack(side=tk.LEFT, padx=10)
        
        results_frame = ttk.LabelFrame(self.backtest_frame, text="📈 Results", padding=10)
        results_frame.pack(fill='both', expand=True, padx=10, pady=5)
        
        self.backtest_text = scrolledtext.ScrolledText(results_frame, height=20)
        self.backtest_text.pack(fill='both', expand=True)
    
    def setup_viz_tab(self):
        """Visualization tab"""
        self.viz_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.viz_frame, text="📊 Visualizations")
        
        controls_frame = ttk.LabelFrame(self.viz_frame, text="📊 Charts", padding=10)
        controls_frame.pack(fill='x', padx=10, pady=5)
        
        buttons = [("📈 Price Chart", self.plot_price), ("📊 Volume", self.plot_volume),
                  ("🎯 Predictions", self.plot_predictions), ("📈 Signals", self.plot_signals)]
        
        for i, (text, command) in enumerate(buttons):
            ttk.Button(controls_frame, text=text, command=command, width=20).grid(
                row=i//2, column=i%2, padx=5, pady=5, sticky='ew')
        
        controls_frame.grid_columnconfigure(0, weight=1)
        controls_frame.grid_columnconfigure(1, weight=1)
        
        self.chart_frame = ttk.Frame(self.viz_frame)
        self.chart_frame.pack(fill='both', expand=True, padx=10, pady=5)
    
    def load_data(self):
        """Load data file"""
        file_path = filedialog.askopenfilename(
            title="Select Trading Data File",
            filetypes=[("All supported", "*.csv *.xlsx *.xls"), ("CSV files", "*.csv"), ("Excel files", "*.xlsx *.xls")]
        )
        
        if not file_path:
            return
        
        try:
            self.update_status("📊 Loading trading data...")
            
            if file_path.lower().endswith(('.xlsx', '.xls')):
                self.data = pd.read_excel(file_path)
            else:
                self.data = pd.read_csv(file_path)
            
            self.validate_data()
            self.display_data()
            self.update_data_info()
            
            filename = os.path.basename(file_path)
            self.file_label.config(text=f"✅ {filename} ({len(self.data)} rows)")
            
            if ADVANCED_MODE:
                self.advanced_system.load_data(self.data)
            
            self.update_status("✅ Data loaded successfully!")
            messagebox.showinfo("Success", f"Data loaded successfully!\n{len(self.data)} rows loaded.")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load data:\n{str(e)}")
            self.update_status("❌ Data loading failed")
    
    def validate_data(self):
        """Validate and clean data"""
        # Column mappings
        mappings = {
            'Date': ['date', 'DATE', 'Timestamp', 'timestamp'],
            'Open': ['open', 'OPEN'],
            'High': ['high', 'HIGH'],
            'Low': ['low', 'LOW'], 
            'Close': ['close', 'CLOSE'],
            'Volume': ['volume', 'VOLUME', 'Vol', 'vol']
        }
        
        for standard, alternatives in mappings.items():
            if standard not in self.data.columns:
                for alt in alternatives:
                    if alt in self.data.columns:
                        self.data.rename(columns={alt: standard}, inplace=True)
                        break
        
        if 'Adj Close' not in self.data.columns:
            self.data['Adj Close'] = self.data['Close'].copy()
        
        if 'Date' in self.data.columns:
            self.data['Date'] = pd.to_datetime(self.data['Date'])
            self.data = self.data.sort_values('Date').reset_index(drop=True)
    
    def display_data(self):
        """Display data in treeview"""
        for item in self.data_tree.get_children():
            self.data_tree.delete(item)
        
        columns = list(self.data.columns)
        self.data_tree['columns'] = columns
        self.data_tree['show'] = 'headings'
        
        for col in columns:
            self.data_tree.heading(col, text=col)
            self.data_tree.column(col, width=100)
        
        for _, row in self.data.head(50).iterrows():
            values = [str(val)[:15] for val in row]
            self.data_tree.insert('', 'end', values=values)
    
    def update_data_info(self):
        """Update data information"""
        info_text = f"📊 DATASET INFORMATION:\n"
        info_text += f"Shape: {self.data.shape[0]:,} rows × {self.data.shape[1]} columns\n"
        
        if 'Date' in self.data.columns:
            info_text += f"Date range: {self.data['Date'].min()} to {self.data['Date'].max()}\n"
        
        if 'Close' in self.data.columns:
            info_text += f"\n💰 PRICE ANALYSIS:\n"
            info_text += f"Current price: ${self.data['Close'].iloc[-1]:.2f}\n"
            info_text += f"Price range: ${self.data['Close'].min():.2f} - ${self.data['Close'].max():.2f}\n"
        
        if 'Volume' in self.data.columns:
            info_text += f"\n📊 VOLUME ANALYSIS:\n"
            info_text += f"Average volume: {self.data['Volume'].mean():,.0f}\n"
        
        info_text += f"\n✅ Ready for {'advanced' if ADVANCED_MODE else 'basic'} analysis!\n"
        
        self.data_info_text.delete(1.0, tk.END)
        self.data_info_text.insert(1.0, info_text)
    
    def generate_features(self):
        """Generate advanced features"""
        if not ADVANCED_MODE:
            messagebox.showwarning("Not Available", "Advanced features require advanced mode")
            return
        
        if self.data is None:
            messagebox.showerror("Error", "Please load data first!")
            return
        
        try:
            self.update_status("🔬 Generating 143+ advanced features...")
            
            success = self.advanced_system.engineer_features()
            
            if success:
                self.features = self.advanced_system.features
                summary = self.feature_engine.create_feature_summary(self.features)
                
                self.features_label.config(text=f"✅ Generated {summary['total_features']} features")
                
                feature_text = f"🔬 ADVANCED FEATURE ENGINEERING RESULTS\n{'='*50}\n\n"
                feature_text += f"📊 TOTAL FEATURES: {summary['total_features']}\n\n"
                feature_text += f"📋 FEATURE CATEGORIES:\n"
                for category, count in summary['categories'].items():
                    feature_text += f"  • {category}: {count} features\n"
                
                feature_text += f"\n✅ Ready for advanced ML training! 🚀\n"
                
                self.features_text.delete(1.0, tk.END)
                self.features_text.insert(1.0, feature_text)
                
                self.update_status("✅ Advanced features generated!")
                messagebox.showinfo("Success", f"Generated {summary['total_features']} features!")
            else:
                raise Exception("Feature engineering failed")
                
        except Exception as e:
            messagebox.showerror("Error", f"Feature generation failed:\n{str(e)}")
            self.update_status("❌ Feature generation failed")
    
    def train_models(self):
        """Train ML models"""
        if self.data is None:
            messagebox.showerror("Error", "Please load data first!")
            return
        
        try:
            if ADVANCED_MODE:
                self.update_status("🤖 Training 16+ advanced ML models...")
                
                success = self.advanced_system.train_models()
                
                if success:
                    self.model_trained = True
                    report = self.advanced_system.ml_ensemble.generate_model_report()
                    
                    self.ml_results_text.delete(1.0, tk.END)
                    self.ml_results_text.insert(1.0, report)
                    
                    self.ml_label.config(text="✅ 16+ models trained!")
                    self.update_status("✅ Advanced ML ensemble trained!")
                    
                    messagebox.showinfo("Success", "Advanced ML ensemble trained!")
                else:
                    raise Exception("Advanced ML training failed")
            else:
                self.train_basic_ml()
                
        except Exception as e:
            messagebox.showerror("Error", f"ML training failed:\n{str(e)}")
            self.update_status("❌ ML training failed")
    
    def train_basic_ml(self):
        """Basic ML training fallback"""
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import mean_squared_error, r2_score
        
        self.update_status("🔧 Training basic ML model...")
        
        # Basic features
        feature_cols = ['Open', 'High', 'Low', 'Adj Close', 'Volume']
        feature_cols = [col for col in feature_cols if col in self.data.columns]
        
        X = self.data[feature_cols].copy()
        y = self.data['Close'].copy()
        
        # Simple features
        X['SMA_10'] = self.data['Close'].rolling(10).mean()
        X['Volatility'] = self.data['Close'].rolling(10).std()
        
        # Clean data
        mask = ~(X.isnull().any(axis=1) | y.isnull())
        X, y = X[mask], y[mask]
        
        # Train
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        
        # Evaluate
        y_pred = model.predict(X_test)
        r2 = r2_score(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        
        self.model_trained = True
        
        results_text = f"🔧 BASIC ML MODEL RESULTS\n{'='*40}\n\n"
        results_text += f"Model: Random Forest\nFeatures: {len(X.columns)}\n"
        results_text += f"Performance:\nR² Score: {r2:.4f}\nMSE: {mse:.4f}\n\n✅ Basic model trained!\n"
        
        self.ml_results_text.delete(1.0, tk.END)
        self.ml_results_text.insert(1.0, results_text)
        
        self.ml_label.config(text="✅ Basic model trained!")
        self.update_status("✅ Basic ML model trained!")
    
    def generate_signals(self):
        """Generate trading signals"""
        if not self.model_trained:
            messagebox.showerror("Error", "Please train models first!")
            return
        
        try:
            if ADVANCED_MODE:
                self.update_status("🎯 Generating advanced trading signals...")
                
                pred_success = self.advanced_system.generate_predictions()
                signal_success = self.advanced_system.generate_trading_signals()
                
                if pred_success and signal_success:
                    self.predictions = self.advanced_system.predictions
                    
                    signals_text = f"🎯 ADVANCED TRADING SIGNALS\n{'='*40}\n\n"
                    
                    signal_counts = self.predictions['Signal'].value_counts()
                    signals_text += f"📊 SIGNAL SUMMARY:\n"
                    for signal, count in signal_counts.items():
                        signals_text += f"  • {signal}: {count} signals\n"
                    
                    latest = self.predictions.iloc[-1]
                    signals_text += f"\n🎯 LATEST SIGNAL: {latest['Signal']} (Confidence: {latest['Confidence']:.2f})\n"
                    signals_text += f"\n✅ Advanced signals ready! 🚀\n"
                    
                    self.signals_text.delete(1.0, tk.END)
                    self.signals_text.insert(1.0, signals_text)
                    
                    self.signals_label.config(text=f"✅ Generated {len(self.predictions)} signals")
                    self.update_status("✅ Trading signals generated!")
                    
                    messagebox.showinfo("Success", "Advanced trading signals generated!")
                else:
                    raise Exception("Signal generation failed")
            else:
                self.generate_basic_signals()
                
        except Exception as e:
            messagebox.showerror("Error", f"Signal generation failed:\n{str(e)}")
            self.update_status("❌ Signal generation failed")
    
    def generate_basic_signals(self):
        """Basic signal generation"""
        signals_text = f"🔧 BASIC SIGNALS\n{'='*30}\n\n"
        
        self.data['SMA_10'] = self.data['Close'].rolling(10).mean()
        self.data['SMA_20'] = self.data['Close'].rolling(20).mean()
        
        conditions = [self.data['SMA_10'] > self.data['SMA_20'], self.data['SMA_10'] < self.data['SMA_20']]
        self.data['Signal'] = np.select(conditions, ['BUY', 'SELL'], default='HOLD')
        
        signal_counts = self.data['Signal'].value_counts()
        for signal, count in signal_counts.items():
            signals_text += f"  • {signal}: {count} signals\n"
        
        signals_text += f"\nLatest: {self.data['Signal'].iloc[-1]}\n✅ Basic signals generated!\n"
        
        self.signals_text.delete(1.0, tk.END)
        self.signals_text.insert(1.0, signals_text)
        
        self.signals_label.config(text="✅ Basic signals generated")
        self.update_status("✅ Basic signals generated!")
    
    def run_backtest(self):
        """Run backtest"""
        if self.predictions is None and not hasattr(self.data, 'Signal'):
            messagebox.showerror("Error", "Please generate signals first!")
            return
        
        try:
            if ADVANCED_MODE and self.predictions is not None:
                self.update_status("📊 Running advanced backtest...")
                
                self.backtest_results = self.advanced_system.run_backtest()
                
                if self.backtest_results:
                    backtest_text = f"📊 ADVANCED BACKTEST RESULTS\n{'='*40}\n\n"
                    backtest_text += f"💰 PERFORMANCE:\n"
                    backtest_text += f"Total Return: {self.backtest_results['total_return']:.2%}\n"
                    backtest_text += f"Win Rate: {self.backtest_results['win_rate']:.1%}\n"
                    backtest_text += f"Total Trades: {self.backtest_results['total_trades']}\n"
                    backtest_text += f"Sharpe Ratio: {self.backtest_results['sharpe_ratio']:.2f}\n"
                    backtest_text += f"Max Drawdown: {self.backtest_results['max_drawdown']:.2%}\n"
                    backtest_text += f"\n✅ Advanced backtest completed! 🚀\n"
                    
                    self.backtest_text.delete(1.0, tk.END)
                    self.backtest_text.insert(1.0, backtest_text)
                    
                    self.backtest_label.config(text=f"✅ {self.backtest_results['total_return']:.1%} return")
                    self.update_status("✅ Backtest completed!")
                    
                    messagebox.showinfo("Complete", f"Backtest: {self.backtest_results['total_return']:.2%} return")
                else:
                    raise Exception("Backtest failed")
            else:
                self.run_basic_backtest()
                
        except Exception as e:
            messagebox.showerror("Error", f"Backtest failed:\n{str(e)}")
            self.update_status("❌ Backtest failed")
    
    def run_basic_backtest(self):
        """Basic backtest"""
        backtest_text = f"🔧 BASIC BACKTEST\n{'='*30}\n\n"
        
        if 'Signal' in self.data.columns:
            buy_signals = (self.data['Signal'] == 'BUY').sum()
            sell_signals = (self.data['Signal'] == 'SELL').sum()
            
            backtest_text += f"Buy Signals: {buy_signals}\nSell Signals: {sell_signals}\n"
            
            returns = self.data['Close'].pct_change()
            total_return = (1 + returns).prod() - 1
            backtest_text += f"Total Return: {total_return:.2%}\n"
        
        backtest_text += f"\n✅ Basic backtest completed!\n"
        
        self.backtest_text.delete(1.0, tk.END)
        self.backtest_text.insert(1.0, backtest_text)
        
        self.backtest_label.config(text="✅ Basic backtest completed")
        self.update_status("✅ Basic backtest completed!")
    
    def clear_chart(self):
        """Clear chart frame"""
        for widget in self.chart_frame.winfo_children():
            widget.destroy()
    
    def plot_price(self):
        """Plot price chart"""
        if self.data is None:
            messagebox.showerror("Error", "Please load data first!")
            return
        
        self.clear_chart()
        
        fig, ax = plt.subplots(figsize=(12, 6))
        fig.patch.set_facecolor('#2d2d2d')
        ax.set_facecolor('#2d2d2d')
        
        ax.plot(self.data['Close'], label='Close Price', color='#00ff88', linewidth=2)
        ax.set_title('Price Chart', color='white', fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.tick_params(colors='white')
        
        canvas = FigureCanvasTkAgg(fig, self.chart_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill='both', expand=True)
    
    def plot_volume(self):
        """Plot volume chart"""
        if self.data is None or 'Volume' not in self.data.columns:
            messagebox.showerror("Error", "Volume data not available!")
            return
        
        self.clear_chart()
        
        fig, ax = plt.subplots(figsize=(12, 6))
        fig.patch.set_facecolor('#2d2d2d')
        ax.set_facecolor('#2d2d2d')
        
        ax.bar(range(len(self.data)), self.data['Volume'], alpha=0.7, color='#ff6b35')
        ax.set_title('Volume Chart', color='white', fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.tick_params(colors='white')
        
        canvas = FigureCanvasTkAgg(fig, self.chart_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill='both', expand=True)
    
    def plot_predictions(self):
        """Plot predictions"""
        if self.predictions is None:
            messagebox.showerror("Error", "No predictions available!")
            return
        
        self.clear_chart()
        
        fig, ax = plt.subplots(figsize=(12, 6))
        fig.patch.set_facecolor('#2d2d2d')
        ax.set_facecolor('#2d2d2d')
        
        ax.plot(self.predictions['Actual'], label='Actual', color='#00ff88', linewidth=2)
        ax.plot(self.predictions['Predicted'], label='Predicted', color='#ff6b35', linewidth=2, alpha=0.8)
        ax.set_title('Predictions vs Actual', color='white', fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.tick_params(colors='white')
        
        canvas = FigureCanvasTkAgg(fig, self.chart_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill='both', expand=True)
    
    def plot_signals(self):
        """Plot signals"""
        if self.predictions is None and 'Signal' not in self.data.columns:
            messagebox.showerror("Error", "No signals available!")
            return
        
        self.clear_chart()
        
        fig, ax = plt.subplots(figsize=(12, 6))
        fig.patch.set_facecolor('#2d2d2d')
        ax.set_facecolor('#2d2d2d')
        
        if self.predictions is not None:
            ax.plot(self.predictions['Actual'], label='Price', color='white', linewidth=2)
            
            buy_signals = self.predictions[self.predictions['Signal'] == 'BUY']
            sell_signals = self.predictions[self.predictions['Signal'] == 'SELL']
            
            if not buy_signals.empty:
                ax.scatter(buy_signals.index, buy_signals['Actual'], 
                          color='green', marker='^', s=100, label='BUY', alpha=0.8)
            
            if not sell_signals.empty:
                ax.scatter(sell_signals.index, sell_signals['Actual'], 
                          color='red', marker='v', s=100, label='SELL', alpha=0.8)
        
        ax.set_title('Trading Signals', color='white', fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.tick_params(colors='white')
        
        canvas = FigureCanvasTkAgg(fig, self.chart_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill='both', expand=True)


def main():
    """Main entry point"""
    try:
        root = tk.Tk()
        app = UltimateAdvancedGUI(root)
        root.mainloop()
    except Exception as e:
        print(f"Error starting application: {e}")

if __name__ == "__main__":
    main()