#!/usr/bin/env python3
"""
Asset ML Strategy - Main Application
A free, local machine learning tool for financial Excel data analysis.
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import seaborn as sns
import os
from datetime import datetime

class AssetMLStrategy:
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