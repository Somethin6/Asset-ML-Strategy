#!/usr/bin/env python3
"""
Enhanced Data Loader - Universal CSV/Excel Loader with Comprehensive Technical Analysis
Supports both traditional and custom column formats with 50+ technical indicators
"""

import pandas as pd
import numpy as np
import talib
import ta
from typing import Dict, List, Optional, Union
import warnings
warnings.filterwarnings('ignore')

class EnhancedDataLoader:
    def __init__(self):
        self.data = None
        self.original_data = None
        self.indicators = {}
        self.support_resistance = {}
        self.fibonacci_levels = {}
        
    def load_data(self, file_path: str, format_type: str = 'auto') -> pd.DataFrame:
        """
        Load data from CSV or Excel with flexible column format detection
        
        Args:
            file_path: Path to the data file
            format_type: 'auto', 'traditional', 'custom'
        
        Returns:
            DataFrame with standardized columns
        """
        try:
            # Determine file type and load
            if file_path.endswith(('.xlsx', '.xls')):
                data = pd.read_excel(file_path)
            elif file_path.endswith('.csv'):
                data = pd.read_csv(file_path)
            else:
                raise ValueError("Unsupported file format. Use CSV or Excel files.")
            
            # Store original data
            self.original_data = data.copy()
            
            # Detect and standardize column format
            data = self._detect_and_standardize_columns(data, format_type)
            
            # Validate required columns
            self._validate_data(data)
            
            # Process data
            self.data = self._process_data(data)
            
            print(f"✅ Data loaded successfully: {len(self.data)} rows, {len(self.data.columns)} columns")
            print(f"📅 Date range: {self.data.index.min()} to {self.data.index.max()}")
            
            return self.data
            
        except Exception as e:
            raise Exception(f"Failed to load data: {str(e)}")
    
    def _detect_and_standardize_columns(self, data: pd.DataFrame, format_type: str) -> pd.DataFrame:
        """Detect column format and standardize column names"""
        
        columns_lower = [col.lower().strip() for col in data.columns]
        
        # Define column mappings for different formats
        traditional_mapping = {
            'date': ['date', 'datetime', 'time', 'timestamp'],
            'open': ['open', 'o'],
            'high': ['high', 'h'], 
            'low': ['low', 'l'],
            'close': ['close', 'c', 'adj close', 'adj_close'],
            'volume': ['volume', 'vol', 'v'],
            'trades': ['trades', 'trade_count', 'count', 'transactions']
        }
        
        custom_mapping = {
            'timestamp': ['timestamp', 'time', 'datetime', 'date'],
            'open': ['open', 'o'],
            'high': ['high', 'h'],
            'low': ['low', 'l'], 
            'close': ['close', 'c'],
            'volume': ['volume', 'vol', 'v'],
            'trades': ['trades', 'trade_count', 'count', 'transactions']
        }
        
        # Choose mapping based on format
        if format_type == 'custom':
            mapping = custom_mapping
            time_col = 'timestamp'
        else:
            mapping = traditional_mapping
            time_col = 'date'
        
        # Create standardized column mapping
        column_map = {}
        found_columns = set()
        
        for standard_col, possible_names in mapping.items():
            for possible_name in possible_names:
                for i, col in enumerate(columns_lower):
                    if possible_name in col and col not in found_columns:
                        column_map[data.columns[i]] = standard_col
                        found_columns.add(col)
                        break
                if standard_col in column_map.values():
                    break
        
        # Apply column mapping
        data_renamed = data.rename(columns=column_map)
        
        # Set time column as index
        time_column = None
        for col in [time_col, 'date', 'timestamp', 'datetime', 'time']:
            if col in data_renamed.columns:
                time_column = col
                break
        
        if time_column:
            data_renamed[time_column] = pd.to_datetime(data_renamed[time_column])
            data_renamed.set_index(time_column, inplace=True)
        
        return data_renamed
    
    def _validate_data(self, data: pd.DataFrame) -> None:
        """Validate that required columns exist"""
        required_cols = ['open', 'high', 'low', 'close', 'volume']
        missing_cols = [col for col in required_cols if col not in data.columns]
        
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        # Check for negative values where inappropriate
        for col in ['open', 'high', 'low', 'close', 'volume']:
            if (data[col] < 0).any():
                print(f"⚠️  Warning: Negative values found in {col} column")
    
    def _process_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Process and clean the data"""
        # Ensure OHLC relationships
        data['high'] = np.maximum.reduce([data['open'], data['high'], data['low'], data['close']])
        data['low'] = np.minimum.reduce([data['open'], data['high'], data['low'], data['close']])
        
        # Remove any NaN values
        data = data.dropna()
        
        # Sort by index (time)
        data = data.sort_index()
        
        return data
    
    def calculate_all_indicators(self) -> Dict:
        """Calculate comprehensive technical indicators (50+ indicators)"""
        if self.data is None:
            raise ValueError("No data loaded. Please load data first.")
        
        indicators = {}
        df = self.data.copy()
        
        print("🔄 Calculating comprehensive technical indicators...")
        
        # Price and Volume basics
        try:
            indicators.update(self._calculate_price_indicators(df))
            print("✅ Price indicators calculated")
        except Exception as e:
            print(f"⚠️  Warning: Error in price indicators: {e}")
        
        try:
            indicators.update(self._calculate_volume_indicators(df))
            print("✅ Volume indicators calculated")
        except Exception as e:
            print(f"⚠️  Warning: Error in volume indicators: {e}")
        
        try:
            indicators.update(self._calculate_momentum_indicators(df))
            print("✅ Momentum indicators calculated")
        except Exception as e:
            print(f"⚠️  Warning: Error in momentum indicators: {e}")
        
        try:
            indicators.update(self._calculate_volatility_indicators(df))
            print("✅ Volatility indicators calculated")
        except Exception as e:
            print(f"⚠️  Warning: Error in volatility indicators: {e}")
        
        try:
            indicators.update(self._calculate_statistical_indicators(df))
            print("✅ Statistical indicators calculated")
        except Exception as e:
            print(f"⚠️  Warning: Error in statistical indicators: {e}")
        
        # Skip trend indicators for now due to API issues
        # indicators.update(self._calculate_trend_indicators(df))
        # indicators.update(self._calculate_pattern_indicators(df))
        
        self.indicators = indicators
        print(f"✅ Calculated {len(indicators)} technical indicators")
        
        return indicators
    
    def _calculate_price_indicators(self, df: pd.DataFrame) -> Dict:
        """Basic price-based indicators"""
        indicators = {}
        
        # Simple Moving Averages
        for period in [5, 10, 20, 50, 100, 200]:
            indicators[f'SMA_{period}'] = ta.trend.sma_indicator(df['close'], window=period)
        
        # Exponential Moving Averages  
        for period in [5, 10, 20, 50]:
            indicators[f'EMA_{period}'] = ta.trend.ema_indicator(df['close'], window=period)
        
        # Weighted Moving Average
        indicators['WMA_14'] = talib.WMA(df['close'], timeperiod=14)
        
        # Hull Moving Average
        indicators['HMA_14'] = self._hull_moving_average(df['close'], 14)
        
        # Typical Price
        indicators['TYPICAL_PRICE'] = (df['high'] + df['low'] + df['close']) / 3
        
        # Median Price
        indicators['MEDIAN_PRICE'] = (df['high'] + df['low']) / 2
        
        # Price changes
        indicators['PRICE_CHANGE'] = df['close'].pct_change()
        indicators['PRICE_CHANGE_5'] = df['close'].pct_change(5)
        indicators['LOG_RETURNS'] = np.log(df['close'] / df['close'].shift(1))
        
        return indicators
    
    def _calculate_volume_indicators(self, df: pd.DataFrame) -> Dict:
        """Volume-based indicators"""
        indicators = {}
        
        # Volume Moving Averages
        indicators['VOLUME_SMA_10'] = df['volume'].rolling(window=10).mean()
        indicators['VOLUME_SMA_20'] = df['volume'].rolling(window=20).mean()
        
        # On Balance Volume
        indicators['OBV'] = ta.volume.on_balance_volume(df['close'], df['volume'])
        
        # Accumulation Distribution Line
        indicators['ADL'] = ta.volume.acc_dist_index(df['high'], df['low'], df['close'], df['volume'])
        
        # Chaikin Money Flow
        indicators['CMF'] = ta.volume.chaikin_money_flow(df['high'], df['low'], df['close'], df['volume'])
        
        # Volume Price Trend
        indicators['VPT'] = ta.volume.volume_price_trend(df['close'], df['volume'])
        
        # Money Flow Index
        indicators['MFI'] = ta.volume.money_flow_index(df['high'], df['low'], df['close'], df['volume'])
        
        # Volume Rate of Change
        indicators['VROC'] = df['volume'].pct_change(12)
        
        return indicators
    
    def _calculate_momentum_indicators(self, df: pd.DataFrame) -> Dict:
        """Momentum-based indicators"""
        indicators = {}
        
        # RSI with multiple periods
        for period in [7, 14, 21]:
            indicators[f'RSI_{period}'] = ta.momentum.rsi(df['close'], window=period)
        
        # Stochastic Oscillator
        stoch = ta.momentum.stoch(df['high'], df['low'], df['close'])
        indicators['STOCH_K'] = stoch
        indicators['STOCH_D'] = stoch.rolling(window=3).mean()
        
        # Williams %R
        indicators['WILLIAMS_R'] = ta.momentum.williams_r(df['high'], df['low'], df['close'])
        
        # Ultimate Oscillator
        indicators['ULTIMATE_OSC'] = ta.momentum.ultimate_oscillator(df['high'], df['low'], df['close'])
        
        # Rate of Change
        for period in [5, 10, 20]:
            indicators[f'ROC_{period}'] = ta.momentum.roc(df['close'], window=period)
        
        # Commodity Channel Index
        indicators['CCI'] = ta.trend.cci(df['high'], df['low'], df['close'])
        
        return indicators
    
    def _calculate_trend_indicators(self, df: pd.DataFrame) -> Dict:
        """Trend-based indicators"""
        indicators = {}
        
        # MACD
        macd = ta.trend.MACD(df['close'])
        indicators['MACD'] = macd.macd()
        indicators['MACD_SIGNAL'] = macd.macd_signal()
        indicators['MACD_HISTOGRAM'] = macd.macd_diff()
        
        # Average Directional Index
        indicators['ADX'] = ta.trend.adx(df['high'], df['low'], df['close'])
        indicators['ADX_POS'] = ta.trend.adx_pos(df['high'], df['low'], df['close'])
        indicators['ADX_NEG'] = ta.trend.adx_neg(df['high'], df['low'], df['close'])
        
        # Parabolic SAR
        indicators['SAR'] = ta.trend.psar_up(df['high'], df['low'], df['close'])
        
        # Aroon - fix the function call
        indicators['AROON_UP'] = ta.trend.aroon_up(df['high'], df['low'])
        indicators['AROON_DOWN'] = ta.trend.aroon_down(df['high'], df['low'])
        
        # Trix
        indicators['TRIX'] = ta.trend.trix(df['close'])
        
        # Mass Index
        indicators['MASS_INDEX'] = ta.trend.mass_index(df['high'], df['low'])
        
        return indicators
    
    def _calculate_volatility_indicators(self, df: pd.DataFrame) -> Dict:
        """Volatility-based indicators"""
        indicators = {}
        
        # Bollinger Bands
        bb = ta.volatility.BollingerBands(df['close'])
        indicators['BB_UPPER'] = bb.bollinger_hband()
        indicators['BB_MIDDLE'] = bb.bollinger_mavg()
        indicators['BB_LOWER'] = bb.bollinger_lband()
        indicators['BB_WIDTH'] = indicators['BB_UPPER'] - indicators['BB_LOWER']
        indicators['BB_PERCENT'] = (df['close'] - indicators['BB_LOWER']) / indicators['BB_WIDTH']
        
        # Average True Range
        indicators['ATR'] = ta.volatility.average_true_range(df['high'], df['low'], df['close'])
        
        # True Range
        indicators['TRUE_RANGE'] = ta.volatility.true_range(df['high'], df['low'], df['close'])
        
        # Keltner Channel
        keltner = ta.volatility.KeltnerChannel(df['high'], df['low'], df['close'])
        indicators['KELTNER_UPPER'] = keltner.keltner_channel_hband()
        indicators['KELTNER_LOWER'] = keltner.keltner_channel_lband()
        
        # Donchian Channel
        donchian = ta.volatility.DonchianChannel(df['high'], df['low'], df['close'])
        indicators['DONCHIAN_UPPER'] = donchian.donchian_channel_hband()
        indicators['DONCHIAN_LOWER'] = donchian.donchian_channel_lband()
        
        # Historical Volatility
        indicators['VOLATILITY_10'] = df['close'].rolling(10).std()
        indicators['VOLATILITY_20'] = df['close'].rolling(20).std()
        
        return indicators
    
    def _calculate_statistical_indicators(self, df: pd.DataFrame) -> Dict:
        """Statistical indicators"""
        indicators = {}
        
        # Z-Score
        indicators['ZSCORE_20'] = (df['close'] - df['close'].rolling(20).mean()) / df['close'].rolling(20).std()
        
        # Linear Regression
        indicators['LINREG_14'] = talib.LINEARREG(df['close'], timeperiod=14)
        indicators['LINREG_SLOPE'] = talib.LINEARREG_SLOPE(df['close'], timeperiod=14)
        
        # Standard Deviation
        indicators['STDDEV_20'] = talib.STDDEV(df['close'], timeperiod=20)
        
        # Variance
        indicators['VAR_14'] = talib.VAR(df['close'], timeperiod=14)
        
        # Correlation with volume
        indicators['PRICE_VOLUME_CORR'] = df['close'].rolling(20).corr(df['volume'])
        
        return indicators
    
    def _calculate_pattern_indicators(self, df: pd.DataFrame) -> Dict:
        """Pattern recognition indicators"""
        indicators = {}
        
        # Candlestick patterns (using talib)
        patterns = [
            'CDL2CROWS', 'CDL3BLACKCROWS', 'CDL3INSIDE', 'CDL3LINESTRIKE',
            'CDL3OUTSIDE', 'CDL3STARSINSOUTH', 'CDL3WHITESOLDIERS', 'CDLABANDONEDBABY',
            'CDLADVANCEBLOCK', 'CDLBELTHOLD', 'CDLBREAKAWAY', 'CDLCLOSINGMARUBOZU',
            'CDLCONCEALBABYSWALL', 'CDLCOUNTERATTACK', 'CDLDARKCLOUDCOVER', 'CDLDOJI',
            'CDLDOJISTAR', 'CDLDRAGONFLYDOJI', 'CDLENGULFING', 'CDLEVENINGDOJISTAR',
            'CDLEVENINGSTAR', 'CDLGAPSIDESIDEWHITE', 'CDLGRAVESTONEDOJI', 'CDLHAMMER',
            'CDLHANGINGMAN', 'CDLHARAMI', 'CDLHARAMICROSS', 'CDLHIGHWAVE', 'CDLHIKKAKE'
        ]
        
        for pattern in patterns:
            try:
                func = getattr(talib, pattern)
                indicators[pattern] = func(df['open'], df['high'], df['low'], df['close'])
            except:
                pass
        
        return indicators
    
    def _hull_moving_average(self, data: pd.Series, period: int) -> pd.Series:
        """Calculate Hull Moving Average"""
        wma_half = talib.WMA(data, timeperiod=int(period/2))
        wma_full = talib.WMA(data, timeperiod=period)
        diff = 2 * wma_half - wma_full
        hma = talib.WMA(diff, timeperiod=int(np.sqrt(period)))
        return hma
    
    def calculate_support_resistance(self, method: str = 'pivot_points') -> Dict:
        """Calculate support and resistance levels"""
        if self.data is None:
            raise ValueError("No data loaded. Please load data first.")
        
        df = self.data.copy()
        levels = {}
        
        if method == 'pivot_points':
            levels = self._calculate_pivot_points(df)
        elif method == 'swing_levels':
            levels = self._calculate_swing_levels(df)
        elif method == 'volume_profile':
            levels = self._calculate_volume_profile(df)
        
        self.support_resistance = levels
        return levels
    
    def _calculate_pivot_points(self, df: pd.DataFrame) -> Dict:
        """Calculate pivot point support and resistance levels"""
        high = df['high'].iloc[-1]
        low = df['low'].iloc[-1]
        close = df['close'].iloc[-1]
        
        pivot = (high + low + close) / 3
        
        levels = {
            'pivot': pivot,
            'resistance_1': 2 * pivot - low,
            'resistance_2': pivot + (high - low),
            'resistance_3': high + 2 * (pivot - low),
            'support_1': 2 * pivot - high,
            'support_2': pivot - (high - low),
            'support_3': low - 2 * (high - pivot)
        }
        
        return levels
    
    def _calculate_swing_levels(self, df: pd.DataFrame, window: int = 20) -> Dict:
        """Calculate swing high and low levels"""
        highs = df['high'].rolling(window=window, center=True).max()
        lows = df['low'].rolling(window=window, center=True).min()
        
        swing_highs = df.loc[df['high'] == highs, 'high'].dropna()
        swing_lows = df.loc[df['low'] == lows, 'low'].dropna()
        
        # Get recent levels
        recent_highs = swing_highs.tail(5).tolist()
        recent_lows = swing_lows.tail(5).tolist()
        
        return {
            'resistance_levels': recent_highs,
            'support_levels': recent_lows
        }
    
    def _calculate_volume_profile(self, df: pd.DataFrame, bins: int = 50) -> Dict:
        """Calculate volume profile-based support and resistance"""
        price_range = df['high'].max() - df['low'].min()
        bin_size = price_range / bins
        
        volume_profile = {}
        
        for _, row in df.iterrows():
            typical_price = (row['high'] + row['low'] + row['close']) / 3
            bin_level = int((typical_price - df['low'].min()) / bin_size)
            
            if bin_level in volume_profile:
                volume_profile[bin_level] += row['volume']
            else:
                volume_profile[bin_level] = row['volume']
        
        # Find high volume levels
        sorted_levels = sorted(volume_profile.items(), key=lambda x: x[1], reverse=True)
        top_levels = sorted_levels[:10]  # Top 10 volume levels
        
        support_resistance_prices = []
        for level, _ in top_levels:
            price = df['low'].min() + (level * bin_size)
            support_resistance_prices.append(price)
        
        return {
            'volume_levels': support_resistance_prices,
            'volume_profile': volume_profile
        }
    
    def calculate_fibonacci_retracements(self, period: int = 50) -> Dict:
        """Calculate Fibonacci retracement levels"""
        if self.data is None:
            raise ValueError("No data loaded. Please load data first.")
        
        df = self.data.tail(period)
        
        high = df['high'].max()
        low = df['low'].min()
        diff = high - low
        
        # Fibonacci ratios
        ratios = [0.236, 0.382, 0.5, 0.618, 0.786]
        
        levels = {
            'high': high,
            'low': low,
        }
        
        # Calculate retracement levels
        for ratio in ratios:
            levels[f'fib_{ratio}'] = high - (diff * ratio)
        
        # Extension levels
        extension_ratios = [1.272, 1.414, 1.618, 2.618]
        for ratio in extension_ratios:
            levels[f'ext_{ratio}'] = high + (diff * (ratio - 1))
        
        self.fibonacci_levels = levels
        return levels
    
    def get_data_summary(self) -> Dict:
        """Get comprehensive data summary"""
        if self.data is None:
            return {"error": "No data loaded"}
        
        df = self.data
        
        summary = {
            'basic_stats': {
                'rows': len(df),
                'columns': len(df.columns),
                'start_date': df.index.min(),
                'end_date': df.index.max(),
                'trading_days': len(df)
            },
            'price_stats': {
                'current_price': df['close'].iloc[-1],
                'highest_price': df['high'].max(),
                'lowest_price': df['low'].min(),
                'average_price': df['close'].mean(),
                'price_volatility': df['close'].std()
            },
            'volume_stats': {
                'total_volume': df['volume'].sum(),
                'average_volume': df['volume'].mean(),
                'highest_volume': df['volume'].max(),
                'volume_volatility': df['volume'].std()
            },
            'indicators_count': len(self.indicators),
            'support_resistance_count': len(self.support_resistance),
            'fibonacci_levels_count': len(self.fibonacci_levels)
        }
        
        return summary

# Example usage and testing
if __name__ == "__main__":
    # Test the enhanced data loader
    loader = EnhancedDataLoader()
    
    # You can test with sample data
    print("🚀 Enhanced Data Loader Test")
    print("=" * 50)
    
    # The loader will be integrated with the main GUI
    print("✅ Enhanced Data Loader initialized successfully")
    print("📊 Features available:")
    print("  • Universal CSV/Excel loading")
    print("  • 50+ Technical indicators")
    print("  • Support/Resistance detection") 
    print("  • Fibonacci retracements")
    print("  • Comprehensive data analysis")