#!/usr/bin/env python3
"""
Real-time and historical market data integration.
Supports multiple data sources including yfinance, Alpha Vantage, and others.
"""

import pandas as pd
import numpy as np
import yfinance as yf
import requests
import os
from datetime import datetime, timedelta
from typing import Optional, List, Dict
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MarketDataFetcher:
    """
    Comprehensive market data fetcher supporting multiple sources.
    """
    
    def __init__(self, alpha_vantage_key: Optional[str] = None):
        """
        Initialize the market data fetcher.
        
        Args:
            alpha_vantage_key: Alpha Vantage API key (optional)
        """
        self.alpha_vantage_key = alpha_vantage_key or os.getenv('ALPHA_VANTAGE_KEY')
        
    def fetch_yfinance_data(
        self, 
        symbol: str, 
        period: str = '1y', 
        interval: str = '1h'
    ) -> pd.DataFrame:
        """
        Fetch data using yfinance (Yahoo Finance).
        
        Args:
            symbol: Stock/crypto symbol (e.g., 'AAPL', 'BTC-USD')
            period: Time period ('1d', '5d', '1mo', '3mo', '6mo', '1y', '2y', '5y', '10y', 'ytd', 'max')
            interval: Data interval ('1m', '2m', '5m', '15m', '30m', '60m', '90m', '1h', '1d', '5d', '1wk', '1mo', '3mo')
        
        Returns:
            DataFrame with OHLCV data
        """
        try:
            ticker = yf.Ticker(symbol)
            data = ticker.history(period=period, interval=interval)
            
            if data.empty:
                logger.warning(f"No data found for symbol {symbol}")
                return pd.DataFrame()
            
            # Standardize column names
            data.columns = [col.lower().replace(' ', '_') for col in data.columns]
            data.reset_index(inplace=True)
            
            # Rename columns to match our standard format
            column_mapping = {
                'datetime': 'time',
                'date': 'time',
                'adj_close': 'adj_close'  # Keep adj_close if available
            }
            data.rename(columns=column_mapping, inplace=True)
            
            # Ensure we have the required columns
            required_columns = ['time', 'open', 'high', 'low', 'close', 'volume']
            if not all(col in data.columns for col in required_columns):
                missing = [col for col in required_columns if col not in data.columns]
                logger.error(f"Missing required columns: {missing}")
                return pd.DataFrame()
            
            # Set time as index
            data['time'] = pd.to_datetime(data['time'])
            data.set_index('time', inplace=True)
            
            logger.info(f"Fetched {len(data)} rows of data for {symbol}")
            return data
            
        except Exception as e:
            logger.error(f"Error fetching yfinance data for {symbol}: {e}")
            return pd.DataFrame()
    
    def fetch_alpha_vantage_data(
        self, 
        symbol: str, 
        function: str = 'TIME_SERIES_INTRADAY',
        interval: str = '60min',
        outputsize: str = 'full'
    ) -> pd.DataFrame:
        """
        Fetch data using Alpha Vantage API.
        
        Args:
            symbol: Stock symbol
            function: Alpha Vantage function name
            interval: Data interval for intraday data
            outputsize: 'compact' or 'full'
        
        Returns:
            DataFrame with OHLCV data
        """
        if not self.alpha_vantage_key:
            logger.error("Alpha Vantage API key not provided")
            return pd.DataFrame()
        
        try:
            url = 'https://www.alphavantage.co/query'
            params = {
                'function': function,
                'symbol': symbol,
                'interval': interval,
                'outputsize': outputsize,
                'apikey': self.alpha_vantage_key
            }
            
            response = requests.get(url, params=params)
            data = response.json()
            
            # Handle different response formats
            time_series_key = None
            for key in data.keys():
                if 'Time Series' in key:
                    time_series_key = key
                    break
            
            if not time_series_key:
                logger.error(f"No time series data found for {symbol}")
                return pd.DataFrame()
            
            time_series = data[time_series_key]
            
            # Convert to DataFrame
            df_list = []
            for timestamp, values in time_series.items():
                row = {
                    'time': pd.to_datetime(timestamp),
                    'open': float(values['1. open']),
                    'high': float(values['2. high']),
                    'low': float(values['3. low']),
                    'close': float(values['4. close']),
                    'volume': int(values['5. volume'])
                }
                df_list.append(row)
            
            df = pd.DataFrame(df_list)
            df.set_index('time', inplace=True)
            df.sort_index(inplace=True)
            
            logger.info(f"Fetched {len(df)} rows of Alpha Vantage data for {symbol}")
            return df
            
        except Exception as e:
            logger.error(f"Error fetching Alpha Vantage data for {symbol}: {e}")
            return pd.DataFrame()
    
    def fetch_crypto_data(self, symbol: str, vs_currency: str = 'usd', days: int = 365) -> pd.DataFrame:
        """
        Fetch cryptocurrency data from CoinGecko API (free).
        
        Args:
            symbol: Crypto symbol (e.g., 'bitcoin', 'ethereum')
            vs_currency: Currency to compare against
            days: Number of days of data
        
        Returns:
            DataFrame with OHLCV data
        """
        try:
            url = f"https://api.coingecko.com/api/v3/coins/{symbol}/ohlc"
            params = {
                'vs_currency': vs_currency,
                'days': days
            }
            
            response = requests.get(url, params=params)
            data = response.json()
            
            if not isinstance(data, list):
                logger.error(f"Unexpected CoinGecko response format for {symbol}")
                return pd.DataFrame()
            
            # Convert to DataFrame
            df_list = []
            for row in data:
                df_list.append({
                    'time': pd.to_datetime(row[0], unit='ms'),
                    'open': row[1],
                    'high': row[2],
                    'low': row[3],
                    'close': row[4],
                    'volume': 0  # CoinGecko OHLC doesn't include volume
                })
            
            df = pd.DataFrame(df_list)
            df.set_index('time', inplace=True)
            
            logger.info(f"Fetched {len(df)} rows of crypto data for {symbol}")
            return df
            
        except Exception as e:
            logger.error(f"Error fetching crypto data for {symbol}: {e}")
            return pd.DataFrame()
    
    def get_market_data(
        self, 
        symbol: str, 
        source: str = 'yfinance',
        **kwargs
    ) -> pd.DataFrame:
        """
        Universal method to get market data from any supported source.
        
        Args:
            symbol: Symbol to fetch
            source: Data source ('yfinance', 'alpha_vantage', 'crypto')
            **kwargs: Additional arguments for specific sources
        
        Returns:
            DataFrame with OHLCV data
        """
        if source == 'yfinance':
            return self.fetch_yfinance_data(symbol, **kwargs)
        elif source == 'alpha_vantage':
            return self.fetch_alpha_vantage_data(symbol, **kwargs)
        elif source == 'crypto':
            return self.fetch_crypto_data(symbol, **kwargs)
        else:
            logger.error(f"Unsupported data source: {source}")
            return pd.DataFrame()
    
    def save_data(self, data: pd.DataFrame, filepath: str):
        """
        Save market data to CSV file.
        
        Args:
            data: DataFrame with market data
            filepath: Path to save the data
        """
        try:
            # Ensure directory exists
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            
            # Reset index to include time column
            data_to_save = data.reset_index()
            data_to_save.to_csv(filepath, index=False)
            
            logger.info(f"Saved {len(data)} rows to {filepath}")
            
        except Exception as e:
            logger.error(f"Error saving data to {filepath}: {e}")

def fetch_popular_assets():
    """
    Fetch data for popular trading assets and save them.
    """
    fetcher = MarketDataFetcher()
    
    # Popular stocks
    stocks = ['AAPL', 'GOOGL', 'MSFT', 'AMZN', 'TSLA', 'NVDA', 'META', 'NFLX']
    
    # Popular crypto (using yfinance crypto symbols)
    crypto = ['BTC-USD', 'ETH-USD', 'BNB-USD', 'ADA-USD', 'XRP-USD', 'SOL-USD']
    
    # Major indices and commodities
    indices = ['^GSPC', '^IXIC', '^DJI', 'GC=F', 'CL=F']  # S&P500, NASDAQ, Dow Jones, Gold, Oil
    
    all_symbols = stocks + crypto + indices
    
    for symbol in all_symbols:
        try:
            logger.info(f"Fetching data for {symbol}...")
            data = fetcher.fetch_yfinance_data(symbol, period='2y', interval='1h')
            
            if not data.empty:
                # Clean symbol name for filename
                clean_symbol = symbol.replace('^', '').replace('=', '').replace('-', '')
                filepath = f'data/real/{clean_symbol}_data.csv'
                fetcher.save_data(data, filepath)
            else:
                logger.warning(f"No data retrieved for {symbol}")
                
        except Exception as e:
            logger.error(f"Failed to fetch {symbol}: {e}")
    
    logger.info("Completed fetching popular assets data")

if __name__ == '__main__':
    # Install required packages if not present
    try:
        import yfinance
    except ImportError:
        print("Installing yfinance...")
        os.system("pip install yfinance")
        import yfinance
    
    # Fetch popular assets
    fetch_popular_assets()
    
    # Example of fetching specific data
    fetcher = MarketDataFetcher()
    
    # Fetch Bitcoin data
    btc_data = fetcher.get_market_data('BTC-USD', source='yfinance', period='1y', interval='1h')
    if not btc_data.empty:
        fetcher.save_data(btc_data, 'data/BTC_live_data.csv')
        print(f"Bitcoin data: {len(btc_data)} rows, from {btc_data.index.min()} to {btc_data.index.max()}")
    
    print("Real-time data fetching setup complete!")