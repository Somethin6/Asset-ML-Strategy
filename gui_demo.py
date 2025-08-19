#!/usr/bin/env python3
"""
GUI Demo - Create screenshots of the Asset ML Strategy GUI
"""

import tkinter as tk
from tkinter import ttk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import pandas as pd
import numpy as np
import os

def create_demo_gui():
    """Create a demo version of the GUI for screenshots"""
    
    # Create main window
    root = tk.Tk()
    root.title("Asset ML Strategy - Free Local Financial Analysis")
    root.geometry("1200x800")
    root.configure(bg='#f0f0f0')
    
    # Main title
    title_label = tk.Label(
        root,
        text="Asset ML Strategy - Financial Data Analyzer",
        font=("Arial", 16, "bold"),
        bg='#f0f0f0',
        fg='#333333'
    )
    title_label.pack(pady=10)
    
    # Create notebook for tabs
    notebook = ttk.Notebook(root)
    notebook.pack(fill='both', expand=True, padx=10, pady=5)
    
    # Tab 1: Data Loading
    data_frame = ttk.Frame(notebook)
    notebook.add(data_frame, text="Data Loading")
    
    # File selection frame
    file_frame = ttk.LabelFrame(data_frame, text="Excel File Selection", padding=10)
    file_frame.pack(fill='x', padx=10, pady=5)
    
    ttk.Button(file_frame, text="Select Excel File", width=20).pack(side=tk.LEFT, padx=5)
    ttk.Label(file_frame, text="Loaded: sample_data.xlsx (20 rows)").pack(side=tk.LEFT, padx=10)
    
    # Data preview frame
    preview_frame = ttk.LabelFrame(data_frame, text="Data Preview", padding=10)
    preview_frame.pack(fill='both', expand=True, padx=10, pady=5)
    
    # Create sample data display
    columns = ['Date', 'Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']
    tree = ttk.Treeview(preview_frame, columns=columns, show='headings', height=10)
    
    for col in columns:
        tree.heading(col, text=col)
        tree.column(col, width=100)
    
    # Add sample data
    sample_rows = [
        ['2023-01-01', '150.25', '155.80', '149.50', '154.30', '154.30', '1250000'],
        ['2023-01-02', '154.50', '158.20', '153.10', '157.85', '157.85', '1180000'],
        ['2023-01-03', '158.00', '159.75', '155.20', '156.90', '156.90', '980000'],
        ['2023-01-04', '157.20', '160.10', '156.80', '159.45', '159.45', '1050000'],
        ['2023-01-05', '159.80', '162.30', '158.50', '161.25', '161.25', '1320000'],
    ]
    
    for row in sample_rows:
        tree.insert('', 'end', values=row)
    
    tree.pack(fill='both', expand=True)
    
    # Tab 2: ML Analysis
    ml_frame = ttk.Frame(notebook)
    notebook.add(ml_frame, text="ML Analysis")
    
    # Configuration frame
    config_frame = ttk.LabelFrame(ml_frame, text="ML Configuration", padding=10)
    config_frame.pack(fill='x', padx=10, pady=5)
    
    ttk.Label(config_frame, text="Target Column:").grid(row=0, column=0, sticky='w', padx=5)
    target_combo = ttk.Combobox(config_frame, width=15, value="Close")
    target_combo['values'] = ["Close", "High", "Low", "Open", "Adj Close"]
    target_combo.grid(row=0, column=1, padx=5)
    
    ttk.Label(config_frame, text="Test Size:").grid(row=0, column=2, sticky='w', padx=5)
    ttk.Spinbox(config_frame, from_=0.1, to=0.5, width=10, value=0.2).grid(row=0, column=3, padx=5)
    
    ttk.Button(config_frame, text="Train Model", width=15).grid(row=0, column=4, padx=10)
    ttk.Button(config_frame, text="Make Predictions", width=15).grid(row=0, column=5, padx=5)
    
    # Results frame
    results_frame = ttk.LabelFrame(ml_frame, text="ML Results", padding=10)
    results_frame.pack(fill='both', expand=True, padx=10, pady=5)
    
    results_text = tk.Text(results_frame, height=15, width=80, bg='white')
    results_text.insert(1.0, """Model Training Results:
==================================================
Target Variable: Close
Features: Open, High, Low, Adj Close, Volume, SMA_5, SMA_10, Volatility, Price_Change
Training samples: 7
Testing samples: 4

Performance Metrics:
Training MSE: 0.1529
Testing MSE: 20.6778
Training R²: 0.9790
Testing R²: 0.8234

Feature Importance:
Volatility: 0.2186
High: 0.1610
Low: 0.1457
SMA_5: 0.1234
Open: 0.1198
""")
    results_text.pack(fill='both', expand=True)
    
    # Tab 3: Visualization
    viz_frame = ttk.Frame(notebook)
    notebook.add(viz_frame, text="Visualization")
    
    # Chart controls
    controls_frame = ttk.LabelFrame(viz_frame, text="Chart Controls", padding=10)
    controls_frame.pack(fill='x', padx=10, pady=5)
    
    ttk.Button(controls_frame, text="Price Chart", width=15).pack(side=tk.LEFT, padx=5)
    ttk.Button(controls_frame, text="Volume Chart", width=15).pack(side=tk.LEFT, padx=5)
    ttk.Button(controls_frame, text="Correlation Matrix", width=15).pack(side=tk.LEFT, padx=5)
    ttk.Button(controls_frame, text="Predictions vs Actual", width=15).pack(side=tk.LEFT, padx=5)
    
    # Chart display
    chart_frame = ttk.LabelFrame(viz_frame, text="Charts", padding=5)
    chart_frame.pack(fill='both', expand=True, padx=10, pady=5)
    
    # Create sample chart
    fig, ax = plt.subplots(figsize=(10, 5))
    
    # Sample data for demonstration
    dates = pd.date_range('2023-01-01', periods=20, freq='D')
    prices = np.random.normal(160, 5, 20).cumsum()
    prices = 150 + (prices - prices[0]) * 0.1
    
    ax.plot(dates, prices, linewidth=2, color='blue', label='Close Price')
    ax.fill_between(dates, prices-2, prices+2, alpha=0.2, color='blue')
    ax.set_title('Sample Price Chart')
    ax.set_xlabel('Date')
    ax.set_ylabel('Price ($)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    canvas = FigureCanvasTkAgg(fig, chart_frame)
    canvas.get_tk_widget().pack(fill='both', expand=True)
    
    # Status bar
    status_bar = tk.Label(
        root,
        text="Status: Ready - Sample data loaded successfully (20 rows processed)",
        relief=tk.SUNKEN,
        anchor=tk.W,
        bg='#e0e0e0'
    )
    status_bar.pack(side=tk.BOTTOM, fill=tk.X)
    
    return root

def main():
    """Create and display demo GUI"""
    print("Creating Asset ML Strategy GUI Demo...")
    
    try:
        root = create_demo_gui()
        
        # Update the display
        root.update()
        
        print("✓ GUI Demo created successfully")
        print("GUI is ready for use with Excel files containing financial data!")
        print("Required columns: Date, Open, High, Low, Close, Adj Close, Volume")
        
        # In a normal environment, this would show the GUI
        # root.mainloop()
        
        # For headless environments, we'll just confirm it works
        root.destroy()
        print("✓ GUI components tested successfully")
        
    except Exception as e:
        print(f"Demo error: {e}")

if __name__ == "__main__":
    main()