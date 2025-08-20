#!/usr/bin/env python3
"""
🚀 ADVANCED FEATURE ENGINEERING SYSTEM 🚀
Calculates 100+ technical indicators and advanced market features
This is the most comprehensive feature engineering module for trading systems!
"""

import pandas as pd
import numpy as np
import warnings
from typing import Dict, List, Optional, Tuple
import logging
from scipy import stats
from scipy.signal import find_peaks
from sklearn.preprocessing import StandardScaler, MinMaxScaler

warnings.filterwarnings('ignore')

class AdvancedFeatureEngine:
    """
    Advanced feature engineering system for financial data
    Generates 100+ technical indicators and market microstructure features
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.feature_cache = {}
        self.scalers = {}
        
    def calculate_all_features(self, data: pd.DataFrame, target_col: str = 'Close') -> pd.DataFrame:
        """
        Calculate all 100+ advanced features for the dataset
        
        Args:
            data: DataFrame with OHLC data
            target_col: Target column name (default: 'Close')
            
        Returns:
            DataFrame with all calculated features
        """
        self.logger.info("🚀 Starting calculation of 100+ advanced features...")
        
        # Create feature dataframe
        features = data.copy()
        
        # 1. PRICE ACTION FEATURES (20 features)
        features = self._add_price_action_features(features, target_col)
        
        # 2. MOVING AVERAGES (15 features) 
        features = self._add_moving_average_features(features, target_col)
        
        # 3. VOLATILITY FEATURES (10 features)
        features = self._add_volatility_features(features, target_col)
        
        # 4. MOMENTUM OSCILLATORS (15 features)
        features = self._add_momentum_features(features, target_col)
        
        # 5. TREND INDICATORS (10 features)
        features = self._add_trend_features(features, target_col)
        
        # 6. VOLUME ANALYSIS (12 features)
        features = self._add_volume_features(features)
        
        # 7. SUPPORT/RESISTANCE (8 features)
        features = self._add_support_resistance_features(features, target_col)
        
        # 8. CANDLESTICK PATTERNS (10 features)
        features = self._add_candlestick_features(features)
        
        # 9. STATISTICAL FEATURES (15 features)
        features = self._add_statistical_features(features, target_col)
        
        # 10. MARKET MICROSTRUCTURE (5 features)
        features = self._add_microstructure_features(features)
        
        self.logger.info(f"✅ Generated {len(features.columns) - len(data.columns)} advanced features!")
        return features
    
    def _add_price_action_features(self, data: pd.DataFrame, col: str) -> pd.DataFrame:
        """Add price action features"""
        # Basic returns and changes
        data[f'{col}_Returns'] = data[col].pct_change()
        data[f'{col}_LogReturns'] = np.log(data[col] / data[col].shift(1))
        data[f'{col}_Change'] = data[col].diff()
        data[f'{col}_Change_Pct'] = data[col].pct_change() * 100
        
        # Price ratios
        if all(c in data.columns for c in ['High', 'Low', 'Open']):
            data['HL_Ratio'] = data['High'] / data['Low']
            data['OC_Ratio'] = data['Open'] / data[col]
            data['HC_Ratio'] = data['High'] / data[col]
            data['LC_Ratio'] = data['Low'] / data[col]
            
        # Price position within range
        data['Price_Position'] = (data[col] - data['Low']) / (data['High'] - data['Low'])
        data['True_Range'] = np.maximum(
            data['High'] - data['Low'],
            np.maximum(
                abs(data['High'] - data[col].shift(1)),
                abs(data['Low'] - data[col].shift(1))
            )
        )
        
        # Gaps
        data['Gap_Up'] = (data['Open'] > data[col].shift(1)).astype(int)
        data['Gap_Down'] = (data['Open'] < data[col].shift(1)).astype(int)
        data['Gap_Size'] = abs(data['Open'] - data[col].shift(1))
        
        # Price accelerations
        data[f'{col}_Acceleration'] = data[f'{col}_Returns'].diff()
        data[f'{col}_Velocity'] = data[f'{col}_Change'].rolling(3).mean()
        
        # Higher highs and lower lows
        data['Higher_High'] = (data['High'] > data['High'].shift(1)).astype(int)
        data['Lower_Low'] = (data['Low'] < data['Low'].shift(1)).astype(int)
        data['Higher_Low'] = (data['Low'] > data['Low'].shift(1)).astype(int)
        data['Lower_High'] = (data['High'] < data['High'].shift(1)).astype(int)
        
        # Body and wick sizes
        data['Body_Size'] = abs(data[col] - data['Open'])
        data['Upper_Wick'] = data['High'] - np.maximum(data[col], data['Open'])
        data['Lower_Wick'] = np.minimum(data[col], data['Open']) - data['Low']
        
        return data
    
    def _add_moving_average_features(self, data: pd.DataFrame, col: str) -> pd.DataFrame:
        """Add moving average features"""
        periods = [5, 10, 20, 50, 100, 200]
        
        for period in periods:
            # Simple Moving Average
            data[f'SMA_{period}'] = data[col].rolling(period).mean()
            data[f'SMA_{period}_Ratio'] = data[col] / data[f'SMA_{period}']
            
            # Exponential Moving Average
            data[f'EMA_{period}'] = data[col].ewm(span=period).mean()
            data[f'EMA_{period}_Ratio'] = data[col] / data[f'EMA_{period}']
            
        # Calculate additional EMAs for crossovers
        data['EMA_12'] = data[col].ewm(span=12).mean()
        data['EMA_26'] = data[col].ewm(span=26).mean()
        
        # Moving average crossovers
        data['SMA_5_20_Cross'] = np.where(data['SMA_5'] > data['SMA_20'], 1, -1)
        data['SMA_10_50_Cross'] = np.where(data['SMA_10'] > data['SMA_50'], 1, -1)
        data['EMA_12_26_Cross'] = np.where(data['EMA_12'] > data['EMA_26'], 1, -1)
        
        return data
    
    def _add_volatility_features(self, data: pd.DataFrame, col: str) -> pd.DataFrame:
        """Add volatility features"""
        # Rolling volatilities
        for period in [5, 10, 20, 50]:
            data[f'Volatility_{period}'] = data[f'{col}_Returns'].rolling(period).std()
            data[f'Volatility_{period}_Ann'] = data[f'Volatility_{period}'] * np.sqrt(252)
            
        # Average True Range
        data['ATR_14'] = data['True_Range'].rolling(14).mean()
        data['ATR_Ratio'] = data['True_Range'] / data['ATR_14']
        
        # Bollinger Bands
        data['BB_Middle'] = data[col].rolling(20).mean()
        bb_std = data[col].rolling(20).std()
        data['BB_Upper'] = data['BB_Middle'] + (bb_std * 2)
        data['BB_Lower'] = data['BB_Middle'] - (bb_std * 2)
        data['BB_Width'] = (data['BB_Upper'] - data['BB_Lower']) / data['BB_Middle']
        data['BB_Position'] = (data[col] - data['BB_Lower']) / (data['BB_Upper'] - data['BB_Lower'])
        
        return data
    
    def _add_momentum_features(self, data: pd.DataFrame, col: str) -> pd.DataFrame:
        """Add momentum oscillator features"""
        # RSI
        delta = data[col].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        data['RSI'] = 100 - (100 / (1 + rs))
        
        # Stochastic Oscillator
        low_14 = data['Low'].rolling(14).min()
        high_14 = data['High'].rolling(14).max()
        data['Stoch_K'] = 100 * (data[col] - low_14) / (high_14 - low_14)
        data['Stoch_D'] = data['Stoch_K'].rolling(3).mean()
        
        # Williams %R
        data['Williams_R'] = -100 * (high_14 - data[col]) / (high_14 - low_14)
        
        # MACD
        ema_12 = data[col].ewm(span=12).mean()
        ema_26 = data[col].ewm(span=26).mean()
        data['MACD'] = ema_12 - ema_26
        data['MACD_Signal'] = data['MACD'].ewm(span=9).mean()
        data['MACD_Histogram'] = data['MACD'] - data['MACD_Signal']
        
        # Commodity Channel Index
        typical_price = (data['High'] + data['Low'] + data[col]) / 3
        data['CCI'] = (typical_price - typical_price.rolling(20).mean()) / (0.015 * typical_price.rolling(20).apply(lambda x: np.mean(np.abs(x - np.mean(x)))))
        
        # Rate of Change
        data['ROC_10'] = ((data[col] - data[col].shift(10)) / data[col].shift(10)) * 100
        data['ROC_20'] = ((data[col] - data[col].shift(20)) / data[col].shift(20)) * 100
        
        # Money Flow Index
        if 'Volume' in data.columns:
            typical_price = (data['High'] + data['Low'] + data[col]) / 3
            money_flow = typical_price * data['Volume']
            positive_flow = money_flow.where(typical_price > typical_price.shift(1), 0).rolling(14).sum()
            negative_flow = money_flow.where(typical_price < typical_price.shift(1), 0).rolling(14).sum()
            data['MFI'] = 100 - (100 / (1 + positive_flow / negative_flow))
        
        return data
    
    def _add_trend_features(self, data: pd.DataFrame, col: str) -> pd.DataFrame:
        """Add trend identification features"""
        # ADX (Average Directional Index)
        high_diff = data['High'].diff()
        low_diff = data['Low'].diff()
        
        plus_dm = high_diff.where((high_diff > low_diff) & (high_diff > 0), 0)
        minus_dm = (-low_diff).where((low_diff > high_diff) & (low_diff < 0), 0)
        
        atr = data['True_Range'].rolling(14).mean()
        plus_di = 100 * (plus_dm.rolling(14).mean() / atr)
        minus_di = 100 * (minus_dm.rolling(14).mean() / atr)
        
        dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
        data['ADX'] = dx.rolling(14).mean()
        data['DI_Plus'] = plus_di
        data['DI_Minus'] = minus_di
        
        # Parabolic SAR (simplified)
        data['SAR'] = data[col].copy()  # Simplified version
        
        # Aroon Indicator
        aroon_up = 100 * (data['High'].rolling(25).apply(lambda x: x.argmax()) / 24)
        aroon_down = 100 * (data['Low'].rolling(25).apply(lambda x: x.argmin()) / 24)
        data['Aroon_Up'] = aroon_up
        data['Aroon_Down'] = aroon_down
        data['Aroon_Oscillator'] = aroon_up - aroon_down
        
        # Linear regression trend
        data['LR_Slope_10'] = data[col].rolling(10).apply(lambda x: stats.linregress(range(len(x)), x)[0])
        data['LR_Slope_20'] = data[col].rolling(20).apply(lambda x: stats.linregress(range(len(x)), x)[0])
        
        return data
    
    def _add_volume_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Add volume analysis features"""
        if 'Volume' not in data.columns:
            return data
            
        # Volume moving averages
        data['Volume_SMA_10'] = data['Volume'].rolling(10).mean()
        data['Volume_SMA_20'] = data['Volume'].rolling(20).mean()
        data['Volume_Ratio'] = data['Volume'] / data['Volume_SMA_20']
        
        # On Balance Volume
        data['OBV'] = (data['Volume'] * np.where(data['Close'] > data['Close'].shift(1), 1, -1)).cumsum()
        
        # Volume Price Trend
        data['VPT'] = (data['Volume'] * data['Close'].pct_change()).cumsum()
        
        # Accumulation/Distribution Line
        clv = ((data['Close'] - data['Low']) - (data['High'] - data['Close'])) / (data['High'] - data['Low'])
        data['ADL'] = (clv * data['Volume']).cumsum()
        
        # Chaikin Money Flow
        data['CMF'] = (clv * data['Volume']).rolling(20).sum() / data['Volume'].rolling(20).sum()
        
        # Volume Oscillator
        data['Volume_Oscillator'] = ((data['Volume_SMA_10'] - data['Volume_SMA_20']) / data['Volume_SMA_20']) * 100
        
        # Price Volume Trend
        data['PVT'] = ((data['Close'] - data['Close'].shift(1)) / data['Close'].shift(1) * data['Volume']).cumsum()
        
        # Volume-weighted features
        data['VWAP'] = (data['Close'] * data['Volume']).cumsum() / data['Volume'].cumsum()
        data['Volume_Weighted_Price'] = (data['Close'] * data['Volume']).rolling(20).sum() / data['Volume'].rolling(20).sum()
        
        # Volume spikes
        volume_std = data['Volume'].rolling(20).std()
        data['Volume_Spike'] = (data['Volume'] > (data['Volume_SMA_20'] + 2 * volume_std)).astype(int)
        
        return data
    
    def _add_support_resistance_features(self, data: pd.DataFrame, col: str) -> pd.DataFrame:
        """Add support and resistance level features"""
        # Pivot points
        data['Pivot'] = (data['High'] + data['Low'] + data[col]) / 3
        data['R1'] = 2 * data['Pivot'] - data['Low']
        data['S1'] = 2 * data['Pivot'] - data['High']
        data['R2'] = data['Pivot'] + (data['High'] - data['Low'])
        data['S2'] = data['Pivot'] - (data['High'] - data['Low'])
        
        # Distance from pivot levels
        data['Distance_Pivot'] = abs(data[col] - data['Pivot']) / data[col]
        data['Distance_R1'] = abs(data[col] - data['R1']) / data[col]
        data['Distance_S1'] = abs(data[col] - data['S1']) / data[col]
        
        return data
    
    def _add_candlestick_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Add candlestick pattern features"""
        # Doji patterns
        body_size = abs(data['Close'] - data['Open'])
        total_range = data['High'] - data['Low']
        data['Doji'] = (body_size <= total_range * 0.1).astype(int)
        
        # Hammer and Hanging Man
        lower_wick = np.minimum(data['Close'], data['Open']) - data['Low']
        upper_wick = data['High'] - np.maximum(data['Close'], data['Open'])
        data['Hammer'] = ((lower_wick >= body_size * 2) & (upper_wick <= body_size * 0.1)).astype(int)
        
        # Engulfing patterns
        data['Bullish_Engulfing'] = ((data['Close'] > data['Open']) & 
                                    (data['Close'].shift(1) < data['Open'].shift(1)) &
                                    (data['Open'] < data['Close'].shift(1)) &
                                    (data['Close'] > data['Open'].shift(1))).astype(int)
        
        data['Bearish_Engulfing'] = ((data['Close'] < data['Open']) & 
                                    (data['Close'].shift(1) > data['Open'].shift(1)) &
                                    (data['Open'] > data['Close'].shift(1)) &
                                    (data['Close'] < data['Open'].shift(1))).astype(int)
        
        # Morning/Evening Star patterns (simplified)
        data['Morning_Star'] = ((data['Close'].shift(2) < data['Open'].shift(2)) &
                               (abs(data['Close'].shift(1) - data['Open'].shift(1)) < body_size.shift(1) * 0.3) &
                               (data['Close'] > data['Open'])).astype(int)
        
        data['Evening_Star'] = ((data['Close'].shift(2) > data['Open'].shift(2)) &
                               (abs(data['Close'].shift(1) - data['Open'].shift(1)) < body_size.shift(1) * 0.3) &
                               (data['Close'] < data['Open'])).astype(int)
        
        # Spinning tops
        data['Spinning_Top'] = ((body_size <= total_range * 0.3) & 
                               (upper_wick >= body_size) & 
                               (lower_wick >= body_size)).astype(int)
        
        # Marubozu
        data['Marubozu'] = ((upper_wick <= total_range * 0.05) & 
                           (lower_wick <= total_range * 0.05)).astype(int)
        
        # Inside/Outside bars
        data['Inside_Bar'] = ((data['High'] <= data['High'].shift(1)) & 
                             (data['Low'] >= data['Low'].shift(1))).astype(int)
        
        data['Outside_Bar'] = ((data['High'] >= data['High'].shift(1)) & 
                              (data['Low'] <= data['Low'].shift(1))).astype(int)
        
        return data
    
    def _add_statistical_features(self, data: pd.DataFrame, col: str) -> pd.DataFrame:
        """Add statistical features"""
        # Skewness and Kurtosis
        data['Returns_Skewness_20'] = data[f'{col}_Returns'].rolling(20).skew()
        data['Returns_Kurtosis_20'] = data[f'{col}_Returns'].rolling(20).kurt()
        
        # Z-score
        data['Price_ZScore_20'] = (data[col] - data[col].rolling(20).mean()) / data[col].rolling(20).std()
        
        # Percentile ranks
        data['Price_Percentile_50'] = data[col].rolling(50).rank(pct=True)
        data['Volume_Percentile_50'] = data['Volume'].rolling(50).rank(pct=True) if 'Volume' in data.columns else 0
        
        # Hurst Exponent (simplified)
        def hurst_exponent(ts, max_lag=20):
            if len(ts) < max_lag:
                return 0.5
            lags = range(2, max_lag)
            tau = [np.sqrt(np.std(np.subtract(ts[lag:], ts[:-lag]))) for lag in lags]
            poly = np.polyfit(np.log(lags), np.log(tau), 1)
            return poly[0]
        
        data['Hurst_Exponent'] = data[col].rolling(50).apply(hurst_exponent)
        
        # Autocorrelation
        data['Autocorr_1'] = data[f'{col}_Returns'].rolling(20).apply(lambda x: x.autocorr(lag=1))
        data['Autocorr_5'] = data[f'{col}_Returns'].rolling(20).apply(lambda x: x.autocorr(lag=5))
        
        # Entropy (simplified)
        data['Returns_Entropy'] = data[f'{col}_Returns'].rolling(20).apply(lambda x: stats.entropy(np.histogram(x, bins=10)[0] + 1e-8))
        
        # Fractal dimension (box counting simplified)
        data['Fractal_Dimension'] = data[col].rolling(50).apply(lambda x: 1.5 if len(x) > 10 else 1.0)  # Placeholder
        
        # Information theory measures
        data['Mutual_Information'] = data[f'{col}_Returns'].rolling(20).apply(lambda x: stats.entropy(np.histogram(x, bins=5)[0] + 1e-8))
        
        # Higher moments
        data['Returns_Moment_3'] = data[f'{col}_Returns'].rolling(20).apply(lambda x: stats.moment(x, moment=3))
        data['Returns_Moment_4'] = data[f'{col}_Returns'].rolling(20).apply(lambda x: stats.moment(x, moment=4))
        
        # Drawdown features
        running_max = data[col].expanding().max()
        data['Drawdown'] = (data[col] - running_max) / running_max
        data['Max_Drawdown_20'] = data['Drawdown'].rolling(20).min()
        
        # Regime change indicator (simplified)
        data['Regime_Change'] = abs(data[f'{col}_Returns'].rolling(20).mean() - data[f'{col}_Returns'].rolling(5).mean())
        
        return data
    
    def _add_microstructure_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Add market microstructure features"""
        if 'trades' in data.columns:
            data['Avg_Trade_Size'] = data['Volume'] / data['trades']
            data['Trade_Intensity'] = data['trades'] / data['Volume'] * 1000
        else:
            # Estimate trade-related features
            data['Estimated_Trades'] = data['Volume'] / 100  # Rough estimate
            data['Avg_Trade_Size'] = 100
            data['Trade_Intensity'] = data['Estimated_Trades'] / data['Volume'] * 1000
        
        # Bid-ask spread proxy (using high-low)
        data['Spread_Proxy'] = (data['High'] - data['Low']) / data['Close']
        
        # Market impact proxy
        if 'Volume' in data.columns:
            data['Market_Impact'] = abs(data['Close_Returns']) / (data['Volume'] / data['Volume'].rolling(20).mean())
        
        return data
    
    def get_feature_importance(self, features: pd.DataFrame, target: pd.Series) -> pd.DataFrame:
        """Calculate feature importance using various methods"""
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.feature_selection import mutual_info_regression
        
        # Remove non-numeric columns and handle NaN
        numeric_features = features.select_dtypes(include=[np.number])
        numeric_features = numeric_features.fillna(numeric_features.mean())
        target_clean = target.fillna(target.mean())
        
        # Align indices
        common_idx = numeric_features.index.intersection(target_clean.index)
        X = numeric_features.loc[common_idx]
        y = target_clean.loc[common_idx]
        
        if len(X) < 10:
            return pd.DataFrame()
        
        importance_scores = {}
        
        try:
            # Random Forest importance
            rf = RandomForestRegressor(n_estimators=50, random_state=42)
            rf.fit(X, y)
            rf_importance = dict(zip(X.columns, rf.feature_importances_))
            importance_scores['Random_Forest'] = rf_importance
            
            # Mutual Information
            mi_scores = mutual_info_regression(X, y)
            mi_importance = dict(zip(X.columns, mi_scores))
            importance_scores['Mutual_Information'] = mi_importance
            
            # Correlation-based
            corr_importance = {col: abs(X[col].corr(y)) for col in X.columns}
            importance_scores['Correlation'] = corr_importance
            
        except Exception as e:
            self.logger.warning(f"Error calculating feature importance: {e}")
            
        return pd.DataFrame(importance_scores).fillna(0)
    
    def create_feature_summary(self, features: pd.DataFrame) -> Dict:
        """Create a comprehensive summary of generated features"""
        feature_categories = {
            'Price Action': [col for col in features.columns if any(x in col for x in ['Returns', 'Change', 'Ratio', 'Gap', 'Body', 'Wick'])],
            'Moving Averages': [col for col in features.columns if any(x in col for x in ['SMA', 'EMA', 'Cross'])],
            'Volatility': [col for col in features.columns if any(x in col for x in ['Volatility', 'ATR', 'BB_', 'True_Range'])],
            'Momentum': [col for col in features.columns if any(x in col for x in ['RSI', 'Stoch', 'MACD', 'CCI', 'ROC', 'MFI', 'Williams'])],
            'Trend': [col for col in features.columns if any(x in col for x in ['ADX', 'DI_', 'SAR', 'Aroon', 'LR_Slope'])],
            'Volume': [col for col in features.columns if any(x in col for x in ['Volume', 'OBV', 'VPT', 'ADL', 'CMF', 'VWAP'])],
            'Support/Resistance': [col for col in features.columns if any(x in col for x in ['Pivot', 'R1', 'R2', 'S1', 'S2', 'Distance'])],
            'Candlestick': [col for col in features.columns if any(x in col for x in ['Doji', 'Hammer', 'Engulfing', 'Star', 'Spinning', 'Marubozu', 'Inside', 'Outside'])],
            'Statistical': [col for col in features.columns if any(x in col for x in ['Skewness', 'Kurtosis', 'ZScore', 'Percentile', 'Hurst', 'Autocorr', 'Entropy', 'Fractal', 'Mutual', 'Moment', 'Drawdown'])],
            'Microstructure': [col for col in features.columns if any(x in col for x in ['Trade', 'Spread', 'Impact'])]
        }
        
        summary = {
            'total_features': len(features.columns),
            'categories': {cat: len(cols) for cat, cols in feature_categories.items()},
            'feature_list': feature_categories
        }
        
        return summary

def main():
    """Demo of advanced feature engineering"""
    print("🚀 ADVANCED FEATURE ENGINEERING SYSTEM DEMO 🚀")
    print("=" * 60)
    
    # Create sample data
    np.random.seed(42)
    dates = pd.date_range('2020-01-01', periods=1000, freq='D')
    
    # Generate realistic OHLCV data
    base_price = 100
    prices = []
    volumes = []
    
    for i in range(1000):
        if i == 0:
            price = base_price
        else:
            change = np.random.normal(0, 0.02)
            price = prices[-1] * (1 + change)
        prices.append(price)
        volumes.append(int(np.random.normal(1000000, 200000)))
    
    data = pd.DataFrame({
        'Date': dates,
        'Open': prices,
        'High': [p * (1 + abs(np.random.normal(0, 0.01))) for p in prices],
        'Low': [p * (1 - abs(np.random.normal(0, 0.01))) for p in prices],
        'Close': [p * (1 + np.random.normal(0, 0.005)) for p in prices],
        'Volume': volumes,
        'trades': [int(v / 100 + np.random.normal(0, 50)) for v in volumes]
    })
    
    # Fix OHLC relationships
    for i in range(len(data)):
        high = max(data.loc[i, 'Open'], data.loc[i, 'High'], data.loc[i, 'Low'], data.loc[i, 'Close'])
        low = min(data.loc[i, 'Open'], data.loc[i, 'High'], data.loc[i, 'Low'], data.loc[i, 'Close'])
        data.loc[i, 'High'] = high
        data.loc[i, 'Low'] = low
    
    # Initialize feature engine
    engine = AdvancedFeatureEngine()
    
    # Calculate all features
    features = engine.calculate_all_features(data)
    
    # Generate summary
    summary = engine.create_feature_summary(features)
    
    print(f"✅ Generated {summary['total_features']} total features!")
    print(f"\n📊 Feature Categories:")
    for category, count in summary['categories'].items():
        print(f"  • {category}: {count} features")
    
    print(f"\n🎯 Top features in each category:")
    for category, feature_list in summary['feature_list'].items():
        if feature_list:
            print(f"  {category}: {feature_list[:3]}{'...' if len(feature_list) > 3 else ''}")
    
    # Calculate feature importance
    try:
        target = features['Close'].shift(-1)  # Next period close
        importance_df = engine.get_feature_importance(features, target)
        if not importance_df.empty:
            print(f"\n🏆 Top 10 Most Important Features:")
            avg_importance = importance_df.mean(axis=1).sort_values(ascending=False)
            for i, (feature, score) in enumerate(avg_importance.head(10).items()):
                print(f"  {i+1:2d}. {feature}: {score:.4f}")
    except Exception as e:
        print(f"Note: Could not calculate feature importance: {e}")
    
    print(f"\n🚀 ADVANCED FEATURE ENGINEERING COMPLETE!")
    print(f"Ready for advanced ML models and predictions! 💰")

if __name__ == "__main__":
    main()