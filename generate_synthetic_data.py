#!/usr/bin/env python3
"""
Generate realistic synthetic market data for testing and development.
Creates OHLCV data with realistic market behavior patterns.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os

def generate_realistic_market_data(
    start_date='2023-01-01',
    end_date='2024-01-01', 
    initial_price=100.0,
    daily_volatility=0.02,
    trend_strength=0.0001,
    mean_reversion_strength=0.001,  # Reduced to prevent explosive behavior
    volume_base=10000,
    output_file='data/market_data.csv'
):
    """
    Generate realistic synthetic market data using geometric Brownian motion
    with mean reversion and trend components.
    
    Args:
        start_date: Start date for data generation
        end_date: End date for data generation
        initial_price: Starting price
        daily_volatility: Daily volatility (standard deviation)
        trend_strength: Strength of trend component
        mean_reversion_strength: Strength of mean reversion
        volume_base: Base volume level
        output_file: Output CSV file path
    """
    
    # Create date range (hourly data for more granularity)
    date_range = pd.date_range(
        start=start_date, 
        end=end_date, 
        freq='h'  # Fixed deprecated 'H' to 'h'
    )
    
    n_periods = len(date_range)
    
    # Initialize price array
    prices = np.zeros(n_periods)
    prices[0] = initial_price
    
    # Generate random components
    np.random.seed(42)  # For reproducibility
    random_returns = np.random.normal(0, daily_volatility/24**0.5, n_periods)  # Hourly volatility
    
    # Cap extreme returns to prevent overflow
    random_returns = np.clip(random_returns, -0.1, 0.1)
    
    # Generate trend component (slight upward bias) - much smaller
    trend_component = np.cumsum(np.random.normal(trend_strength/24, 0.00001, n_periods))
    
    # Generate prices with realistic behavior
    for i in range(1, n_periods):
        # Mean reversion component - simplified and safer
        if prices[i-1] > 0:
            mean_reversion = mean_reversion_strength * (initial_price - prices[i-1]) / initial_price
        else:
            mean_reversion = 0
            
        # Combine all components with bounds checking
        total_return = random_returns[i] + trend_component[i] + mean_reversion
        
        # Cap total return to prevent overflow
        total_return = np.clip(total_return, -0.2, 0.2)
        
        # Apply return to price with minimum price floor
        new_price = prices[i-1] * (1 + total_return)
        prices[i] = max(new_price, initial_price * 0.1)  # Minimum 10% of initial price
    
    # Generate OHLCV data
    data = []
    
    for i in range(n_periods):
        # Use previous close as base (or initial price for first)
        base_price = prices[i]
        
        # Generate intraday volatility
        intraday_vol = daily_volatility * 0.3  # Lower intraday volatility
        
        # Generate OHLC with realistic relationships
        high_factor = abs(np.random.normal(1.003, intraday_vol))
        low_factor = abs(np.random.normal(0.997, intraday_vol))
        
        high = base_price * high_factor
        low = base_price * low_factor
        
        # Ensure low <= base <= high
        if low > base_price:
            low = base_price * 0.998
        if high < base_price:
            high = base_price * 1.002
        
        # Open is close to previous close with some gap
        if i == 0:
            open_price = base_price
        else:
            gap = np.random.normal(0, daily_volatility * 0.1)
            open_price = prices[i-1] * (1 + gap)
            # Ensure open is within high-low range
            open_price = max(min(open_price, high), low)
        
        close = base_price
        
        # Generate volume with some correlation to price movement
        if open_price > 0 and np.isfinite(open_price) and np.isfinite(close):
            price_change = abs(close - open_price) / open_price
        else:
            price_change = 0
            
        volume_multiplier = 1 + min(price_change * 5, 3)  # Cap volume multiplier
        volume_noise = np.random.lognormal(0, 0.3)  # Log-normal volume distribution
        
        # Ensure volume is finite and positive
        volume_calc = volume_base * volume_multiplier * volume_noise
        if np.isfinite(volume_calc) and volume_calc > 0:
            volume = int(volume_calc)
        else:
            volume = volume_base
        
        data.append({
            'time': date_range[i],
            'open': round(float(open_price), 2) if np.isfinite(open_price) else initial_price,
            'high': round(float(high), 2) if np.isfinite(high) else initial_price,
            'low': round(float(low), 2) if np.isfinite(low) else initial_price,
            'close': round(float(close), 2) if np.isfinite(close) else initial_price,
            'volume': volume
        })
    
    # Create DataFrame
    df = pd.DataFrame(data)
    
    # Ensure OHLC consistency
    for i in range(len(df)):
        high = max(df.loc[i, 'open'], df.loc[i, 'high'], df.loc[i, 'low'], df.loc[i, 'close'])
        low = min(df.loc[i, 'open'], df.loc[i, 'high'], df.loc[i, 'low'], df.loc[i, 'close'])
        df.loc[i, 'high'] = high
        df.loc[i, 'low'] = low
    
    # Save to file
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    df.to_csv(output_file, index=False)
    
    print(f"Generated {len(df)} rows of synthetic market data")
    print(f"Date range: {df['time'].min()} to {df['time'].max()}")
    print(f"Price range: ${df['close'].min():.2f} to ${df['close'].max():.2f}")
    print(f"Average volume: {df['volume'].mean():,.0f}")
    print(f"Data saved to: {output_file}")
    
    return df

def generate_multiple_assets_data():
    """Generate data for multiple assets to test portfolio strategies"""
    
    assets = {
        'BTCUSD': {'initial_price': 45000, 'volatility': 0.04, 'trend': 0.0002},
        'ETHUSD': {'initial_price': 3000, 'volatility': 0.05, 'trend': 0.0001},
        'SPXUSD': {'initial_price': 4500, 'volatility': 0.015, 'trend': 0.00015},
        'GOLD': {'initial_price': 1900, 'volatility': 0.012, 'trend': 0.00005},
    }
    
    for asset, params in assets.items():
        output_file = f'data/{asset}_data.csv'
        generate_realistic_market_data(
            initial_price=params['initial_price'],
            daily_volatility=params['volatility'],
            trend_strength=params['trend'],
            output_file=output_file
        )
        print(f"Generated data for {asset}")

if __name__ == '__main__':
    # Generate main dataset
    df = generate_realistic_market_data()
    
    # Also generate multiple asset data
    generate_multiple_assets_data()
    
    print("\nSynthetic market data generation complete!")