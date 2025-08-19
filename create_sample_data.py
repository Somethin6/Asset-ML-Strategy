#!/usr/bin/env python3
"""
Create sample data files for immediate testing
"""

import pandas as pd
import numpy as np
import os

def create_sample_data():
    """Create realistic sample financial data"""
    
    # Create directory
    os.makedirs('sample_data', exist_ok=True)
    
    # Generate sample financial data
    np.random.seed(42)
    dates = pd.date_range('2020-01-01', periods=252, freq='D')
    base_price = 100
    prices = [base_price]

    for i in range(1, 252):
        change = np.random.normal(0, 0.015)
        new_price = prices[-1] * (1 + change)
        prices.append(max(new_price, 1))  # Ensure positive prices

    # Create OHLCV data
    data = pd.DataFrame({
        'Date': dates,
        'Open': prices,
        'High': [p * (1 + abs(np.random.normal(0, 0.008))) for p in prices],
        'Low': [p * (1 - abs(np.random.normal(0, 0.008))) for p in prices],
        'Close': [p * (1 + np.random.normal(0, 0.003)) for p in prices],
        'Volume': [int(np.random.normal(1000000, 200000)) for _ in prices]
    })

    # Fix OHLC relationships
    for i in range(len(data)):
        open_price = data.loc[i, 'Open']
        close_price = data.loc[i, 'Close']
        high_candidate = data.loc[i, 'High']
        low_candidate = data.loc[i, 'Low']
        
        true_high = max(open_price, close_price, high_candidate)
        true_low = min(open_price, close_price, low_candidate)
        
        data.loc[i, 'High'] = true_high
        data.loc[i, 'Low'] = true_low
        data.loc[i, 'Adj Close'] = close_price

    # Ensure positive volumes
    data['Volume'] = data['Volume'].clip(lower=100000)

    # Save sample data
    data.to_csv('sample_data/AAPL_sample.csv', index=False)
    data.to_excel('sample_data/AAPL_sample.xlsx', index=False)

    print('✅ Sample data files created:')
    print('  - sample_data/AAPL_sample.csv')  
    print('  - sample_data/AAPL_sample.xlsx')
    print(f'  - {len(data)} rows of realistic financial data')
    print(f'  - Price range: ${data["Close"].min():.2f} to ${data["Close"].max():.2f}')
    
    return data

if __name__ == "__main__":
    create_sample_data()