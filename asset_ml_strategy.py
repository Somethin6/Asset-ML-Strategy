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
        """Create the main GUI interface"""
        # Main title
        title_label = tk.Label(
            self.root,
            text="Asset ML Strategy - Financial Data Analyzer",
            font=("Arial", 16, "bold"),
            bg='#f0f0f0',
            fg='#333333'
        )
        title_label.pack(pady=10)
        
        # Create notebook for tabs
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill='both', expand=True, padx=10, pady=5)
        
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
        
        # Status bar
        self.status_bar = tk.Label(
            self.root,
            text="Ready - Load an Excel file to begin",
            relief=tk.SUNKEN,
            anchor=tk.W,
            bg='#e0e0e0'
        )
        self.status_bar.pack(side=tk.BOTTOM, fill=tk.X)
        
    def setup_data_tab(self):
        """Setup data loading tab"""
        # File selection frame
        file_frame = ttk.LabelFrame(self.data_frame, text="Excel File Selection", padding=10)
        file_frame.pack(fill='x', padx=10, pady=5)
        
        ttk.Button(
            file_frame,
            text="Select Excel File",
            command=self.load_excel_file,
            width=20
        ).pack(side=tk.LEFT, padx=5)
        
        self.file_label = ttk.Label(file_frame, text="No file selected")
        self.file_label.pack(side=tk.LEFT, padx=10)
        
        # Data preview frame
        preview_frame = ttk.LabelFrame(self.data_frame, text="Data Preview", padding=10)
        preview_frame.pack(fill='both', expand=True, padx=10, pady=5)
        
        # Create treeview for data display
        self.data_tree = ttk.Treeview(preview_frame)
        data_scrollbar_y = ttk.Scrollbar(preview_frame, orient='vertical', command=self.data_tree.yview)
        data_scrollbar_x = ttk.Scrollbar(preview_frame, orient='horizontal', command=self.data_tree.xview)
        self.data_tree.configure(yscrollcommand=data_scrollbar_y.set, xscrollcommand=data_scrollbar_x.set)
        
        self.data_tree.pack(side=tk.LEFT, fill='both', expand=True)
        data_scrollbar_y.pack(side=tk.RIGHT, fill='y')
        data_scrollbar_x.pack(side=tk.BOTTOM, fill='x')
        
        # Data info frame
        info_frame = ttk.LabelFrame(self.data_frame, text="Data Information", padding=10)
        info_frame.pack(fill='x', padx=10, pady=5)
        
        self.info_text = scrolledtext.ScrolledText(info_frame, height=6, width=80)
        self.info_text.pack(fill='both', expand=True)
        
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
        
        self.results_text = scrolledtext.ScrolledText(results_frame, height=20, width=100)
        self.results_text.pack(fill='both', expand=True)
        
    def setup_viz_tab(self):
        """Setup visualization tab"""
        # Chart controls
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
        
        # Chart display frame
        chart_frame = ttk.LabelFrame(self.viz_frame, text="Charts", padding=5)
        chart_frame.pack(fill='both', expand=True, padx=10, pady=5)
        
        # Matplotlib figure
        self.fig, self.ax = plt.subplots(figsize=(10, 6))
        self.canvas = FigureCanvasTkAgg(self.fig, chart_frame)
        self.canvas.get_tk_widget().pack(fill='both', expand=True)
        
    def load_excel_file(self):
        """Load Excel file and validate columns"""
        file_path = filedialog.askopenfilename(
            title="Select Excel File",
            filetypes=[("Excel files", "*.xlsx *.xls"), ("All files", "*.*")]
        )
        
        if file_path:
            try:
                self.update_status("Loading Excel file...")
                
                # Load the Excel file
                self.data = pd.read_excel(file_path)
                
                # Check for required columns
                required_columns = ['Date', 'Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']
                missing_columns = [col for col in required_columns if col not in self.data.columns]
                
                if missing_columns:
                    messagebox.showerror(
                        "Missing Columns",
                        f"The following required columns are missing:\n{', '.join(missing_columns)}\n\n"
                        f"Required columns: {', '.join(required_columns)}"
                    )
                    return
                
                # Convert Date column to datetime
                self.data['Date'] = pd.to_datetime(self.data['Date'])
                
                # Update GUI
                self.file_label.config(text=f"Loaded: {os.path.basename(file_path)}")
                self.update_data_preview()
                self.update_data_info()
                self.update_status(f"Successfully loaded {len(self.data)} rows of data")
                
                messagebox.showinfo(
                    "Success",
                    f"Excel file loaded successfully!\n"
                    f"Rows: {len(self.data)}\n"
                    f"Columns: {len(self.data.columns)}"
                )
                
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load Excel file:\n{str(e)}")
                self.update_status("Error loading file")
    
    def update_data_preview(self):
        """Update the data preview treeview"""
        if self.data is None:
            return
        
        # Clear existing data
        for item in self.data_tree.get_children():
            self.data_tree.delete(item)
        
        # Setup columns
        columns = list(self.data.columns)
        self.data_tree['columns'] = columns
        self.data_tree['show'] = 'headings'
        
        for col in columns:
            self.data_tree.heading(col, text=col)
            self.data_tree.column(col, width=100)
        
        # Add data (first 100 rows for performance)
        for index, row in self.data.head(100).iterrows():
            values = [str(row[col]) for col in columns]
            self.data_tree.insert('', 'end', values=values)
    
    def update_data_info(self):
        """Update data information display"""
        if self.data is None:
            return
        
        info_text = f"Dataset Information:\n"
        info_text += f"Shape: {self.data.shape[0]} rows × {self.data.shape[1]} columns\n"
        info_text += f"Date range: {self.data['Date'].min()} to {self.data['Date'].max()}\n\n"
        
        info_text += "Column Statistics:\n"
        numeric_columns = ['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']
        for col in numeric_columns:
            if col in self.data.columns:
                info_text += f"{col}: Min={self.data[col].min():.2f}, Max={self.data[col].max():.2f}, Mean={self.data[col].mean():.2f}\n"
        
        info_text += f"\nMissing values:\n"
        missing = self.data.isnull().sum()
        for col, count in missing.items():
            if count > 0:
                info_text += f"{col}: {count}\n"
        
        if missing.sum() == 0:
            info_text += "No missing values found.\n"
        
        self.info_text.delete(1.0, tk.END)
        self.info_text.insert(1.0, info_text)
    
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
        
        self.ax.clear()
        
        # Plot OHLC prices
        self.ax.plot(self.data['Date'], self.data['Open'], label='Open', alpha=0.7)
        self.ax.plot(self.data['Date'], self.data['High'], label='High', alpha=0.7)
        self.ax.plot(self.data['Date'], self.data['Low'], label='Low', alpha=0.7)
        self.ax.plot(self.data['Date'], self.data['Close'], label='Close', linewidth=2)
        
        self.ax.set_title('Price Chart (OHLC)')
        self.ax.set_xlabel('Date')
        self.ax.set_ylabel('Price')
        self.ax.legend()
        self.ax.grid(True, alpha=0.3)
        
        # Rotate x-axis labels for better readability
        plt.setp(self.ax.xaxis.get_majorticklabels(), rotation=45)
        
        self.canvas.draw()
        self.update_status("Price chart displayed")
    
    def plot_volume_chart(self):
        """Plot volume chart"""
        if self.data is None:
            messagebox.showerror("Error", "Please load data first!")
            return
        
        self.ax.clear()
        
        # Plot volume
        self.ax.bar(self.data['Date'], self.data['Volume'], alpha=0.7, color='orange')
        
        self.ax.set_title('Trading Volume')
        self.ax.set_xlabel('Date')
        self.ax.set_ylabel('Volume')
        self.ax.grid(True, alpha=0.3)
        
        # Rotate x-axis labels for better readability
        plt.setp(self.ax.xaxis.get_majorticklabels(), rotation=45)
        
        self.canvas.draw()
        self.update_status("Volume chart displayed")
    
    def plot_correlation(self):
        """Plot correlation matrix"""
        if self.data is None:
            messagebox.showerror("Error", "Please load data first!")
            return
        
        self.ax.clear()
        
        # Calculate correlation matrix
        numeric_cols = ['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']
        corr_data = self.data[numeric_cols].corr()
        
        # Plot heatmap
        im = self.ax.imshow(corr_data.values, cmap='coolwarm', aspect='auto', vmin=-1, vmax=1)
        
        # Add labels
        self.ax.set_xticks(range(len(corr_data.columns)))
        self.ax.set_yticks(range(len(corr_data.columns)))
        self.ax.set_xticklabels(corr_data.columns, rotation=45, ha='right')
        self.ax.set_yticklabels(corr_data.columns)
        
        # Add correlation values as text
        for i in range(len(corr_data.columns)):
            for j in range(len(corr_data.columns)):
                text = self.ax.text(j, i, f'{corr_data.iloc[i, j]:.2f}',
                                   ha='center', va='center', color='black', fontsize=8)
        
        self.ax.set_title('Correlation Matrix')
        
        # Add colorbar
        self.fig.colorbar(im, ax=self.ax)
        
        self.canvas.draw()
        self.update_status("Correlation matrix displayed")
    
    def plot_predictions(self):
        """Plot predictions vs actual values"""
        if self.predictions is None:
            messagebox.showerror("Error", "Please train a model first!")
            return
        
        self.ax.clear()
        
        # Plot actual vs predicted
        y_test = self.predictions['y_test']
        y_pred = self.predictions['y_pred']
        
        # Scatter plot
        self.ax.scatter(y_test, y_pred, alpha=0.6, color='blue')
        
        # Perfect prediction line
        min_val = min(y_test.min(), y_pred.min())
        max_val = max(y_test.max(), y_pred.max())
        self.ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Prediction')
        
        self.ax.set_xlabel(f'Actual {self.predictions["target_col"]}')
        self.ax.set_ylabel(f'Predicted {self.predictions["target_col"]}')
        self.ax.set_title('Predictions vs Actual Values')
        self.ax.legend()
        self.ax.grid(True, alpha=0.3)
        
        self.canvas.draw()
        self.update_status("Predictions chart displayed")
    
    def update_status(self, message):
        """Update status bar"""
        self.status_bar.config(text=f"Status: {message}")
        self.root.update_idletasks()

def main():
    """Main application entry point"""
    try:
        # Create main window
        root = tk.Tk()
        
        # Create application
        app = AssetMLStrategy(root)
        
        # Start GUI event loop
        root.mainloop()
        
    except Exception as e:
        print(f"Error starting application: {e}")
        messagebox.showerror("Startup Error", f"Failed to start application:\n{str(e)}")

if __name__ == "__main__":
    main()