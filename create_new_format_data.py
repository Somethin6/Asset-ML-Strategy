#!/usr/bin/env python3
"""
Create sample data in the new format (timestamp, open, high, low, close, volume, trades)
for testing the ultimate ML trading system
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def create_new_format_sample_data():
    """Create sample data in the new format requested"""
    
    # Generate 1000 days of sample data
    start_date = datetime(2020, 1, 1)
    dates = [start_date + timedelta(days=i) for i in range(1000)]
    
    np.random.seed(42)
    
    # Generate realistic financial data
    base_price = 100.0
    prices = [base_price]
    volumes = []
    trades = []
    
    # Generate price series with realistic movement
    for i in range(1, 1000):
        # Add some trend and seasonality
        trend = 0.0001  # Small constant trend
        seasonal = 0.01 * np.sin(2 * np.pi * i / 252)  # Yearly seasonality
        
        # Random walk with controlled volatility
        volatility = 0.015 + 0.005 * abs(np.sin(i / 50))  # Varying volatility
        change = np.random.normal(trend + seasonal, volatility)
        
        # Limit extreme moves
        change = np.clip(change, -0.1, 0.1)  # Max 10% daily move
        
        new_price = prices[-1] * (1 + change)
        prices.append(max(new_price, 1.0))  # Prevent unrealistic prices
    
    # Generate OHLC data
    opens = []
    highs = []
    lows = []
    closes = prices[:]
    
    for i, close_price in enumerate(closes):
        # Generate realistic OHLC relationships
        daily_volatility = np.random.normal(0, 0.01)
        
        # Open is close of previous day plus gap
        if i == 0:
            open_price = close_price
        else:
            gap = np.random.normal(0, 0.005)
            open_price = closes[i-1] * (1 + gap)
        
        # High and low based on intraday volatility
        intraday_range = abs(np.random.normal(0, 0.02))
        high_price = max(open_price, close_price) * (1 + intraday_range/2)
        low_price = min(open_price, close_price) * (1 - intraday_range/2)
        
        opens.append(open_price)
        highs.append(high_price)
        lows.append(low_price)
        
        # Generate volume with some correlation to price movement
        base_volume = 1000000
        volume_factor = 1 + 0.5 * abs(daily_volatility) + np.random.normal(0, 0.3)
        volume = max(int(base_volume * volume_factor), 10000)
        volumes.append(volume)
        
        # Generate number of trades (roughly correlated with volume)
        base_trades = 10000
        trade_factor = 0.5 + 0.5 * (volume / base_volume) + np.random.normal(0, 0.2)
        trade_count = max(int(base_trades * trade_factor), 100)
        trades.append(trade_count)
    
    # Create DataFrame with new format
    sample_data = pd.DataFrame({
        'timestamp': dates,
        'open': opens,
        'high': highs, 
        'low': lows,
        'close': closes,
        'volume': volumes,
        'trades': trades
    })
    
    # Round numeric values
    for col in ['open', 'high', 'low', 'close']:
        sample_data[col] = sample_data[col].round(4)
    
    return sample_data

def create_traditional_format_sample():
    """Create sample data in traditional format for comparison"""
    
    # Create new format first
    new_format_data = create_new_format_sample_data()
    
    # Convert to traditional format
    traditional_data = pd.DataFrame({
        'Date': new_format_data['timestamp'],
        'Open': new_format_data['open'],
        'High': new_format_data['high'],
        'Low': new_format_data['low'],
        'Close': new_format_data['close'],
        'Volume': new_format_data['volume'],
        'Adj Close': new_format_data['close']  # Same as close for simplicity
    })
    
    return traditional_data

if __name__ == "__main__":
    print("🔄 Creating sample data files...")
    
    # Create new format sample data
    new_format_data = create_new_format_sample_data()
    
    # Save as CSV
    new_format_data.to_csv('/home/runner/work/Asset-ML-Strategy/Asset-ML-Strategy/sample_data/new_format_sample.csv', index=False)
    print(f"✅ Created new format CSV: {len(new_format_data)} rows")
    print(f"   Columns: {list(new_format_data.columns)}")
    print(f"   Date range: {new_format_data['timestamp'].min()} to {new_format_data['timestamp'].max()}")
    
    # Save as Excel
    new_format_data.to_excel('/home/runner/work/Asset-ML-Strategy/Asset-ML-Strategy/sample_data/new_format_sample.xlsx', index=False)
    print("✅ Created new format Excel file")
    
    # Create traditional format for comparison
    traditional_data = create_traditional_format_sample()
    traditional_data.to_csv('/home/runner/work/Asset-ML-Strategy/Asset-ML-Strategy/sample_data/traditional_format_sample.csv', index=False)
    traditional_data.to_excel('/home/runner/work/Asset-ML-Strategy/Asset-ML-Strategy/sample_data/traditional_format_sample.xlsx', index=False)
    print("✅ Created traditional format files for comparison")
    
    # Display sample of new format data
    print("\n📊 Sample of new format data:")
    print(new_format_data.head(10).to_string(index=False))
    
    # Display statistics
    print(f"\n📈 Data Statistics:")
    print(f"   Price range: ${new_format_data['close'].min():.2f} - ${new_format_data['close'].max():.2f}")
    print(f"   Avg volume: {new_format_data['volume'].mean():,.0f}")
    print(f"   Avg trades: {new_format_data['trades'].mean():,.0f}")
    
    print("\n🚀 Sample data files created successfully!")
    print("   You can now test the Ultimate ML Strategy system with both formats.")