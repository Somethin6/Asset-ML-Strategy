#!/usr/bin/env python3
"""
Advanced ML Ensemble Trading System - The "MoneyPrinter"
Combines multiple ML models with advanced features for maximum profitability.
"""

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import xgboost as xgb
import lightgbm as lgb
from typing import List, Dict, Tuple, Optional
import logging
import warnings
warnings.filterwarnings('ignore')

from src.data_loader import load_data
from src.feature_engineering import add_features
from src.backtesting import Backtester
from risk_management import RiskManager, RiskMetrics

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AdvancedFeatureEngineer:
    """
    Advanced feature engineering for maximum alpha generation.
    """
    
    def __init__(self):
        self.scaler = None
    
    def add_market_microstructure_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add market microstructure features for high-frequency patterns.
        """
        # Price impact features
        df['price_impact'] = (df['close'] - df['open']) / df['volume']
        df['bid_ask_spread_proxy'] = (df['high'] - df['low']) / df['close']
        
        # Volume-weighted features
        df['vwap'] = (df['close'] * df['volume']).rolling(20).sum() / df['volume'].rolling(20).sum()
        df['price_vs_vwap'] = (df['close'] - df['vwap']) / df['vwap']
        
        # Tick-level features
        df['tick_direction'] = np.sign(df['close'].diff())
        df['tick_momentum'] = df['tick_direction'].rolling(10).sum()
        
        # Order flow imbalance proxy
        df['order_flow_imbalance'] = (df['high'] + df['low'] - 2 * df['close']) / (df['high'] - df['low'] + 1e-8)
        
        return df
    
    def add_regime_detection_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add market regime detection features.
        """
        # Volatility regimes
        returns = df['close'].pct_change()
        df['realized_vol'] = returns.rolling(20).std() * np.sqrt(252)
        df['vol_regime'] = pd.qcut(df['realized_vol'].rank(method='first'), q=3, labels=[0, 1, 2])
        
        # Trend regimes
        df['trend_strength'] = (df['close'] - df['close'].rolling(50).mean()) / df['close'].rolling(50).std()
        df['trend_regime'] = pd.qcut(df['trend_strength'].rank(method='first'), q=3, labels=[0, 1, 2])
        
        # Mean reversion vs momentum regimes
        df['momentum_20'] = df['close'] / df['close'].shift(20) - 1
        df['mean_reversion_20'] = (df['close'].rolling(20).mean() - df['close']) / df['close']
        
        return df
    
    def add_multi_timeframe_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add multi-timeframe analysis features.
        """
        # Short-term features (5 periods)
        df['ma_5'] = df['close'].rolling(5).mean()
        df['std_5'] = df['close'].rolling(5).std()
        df['rsi_5'] = self.calculate_rsi(df['close'], 5)
        
        # Medium-term features (20 periods)
        df['ma_20'] = df['close'].rolling(20).mean()
        df['std_20'] = df['close'].rolling(20).std()
        df['rsi_20'] = self.calculate_rsi(df['close'], 20)
        
        # Long-term features (50 periods)
        df['ma_50'] = df['close'].rolling(50).mean()
        df['std_50'] = df['close'].rolling(50).std()
        df['rsi_50'] = self.calculate_rsi(df['close'], 50)
        
        # Cross-timeframe relationships
        df['ma_ratio_5_20'] = df['ma_5'] / df['ma_20']
        df['ma_ratio_20_50'] = df['ma_20'] / df['ma_50']
        df['vol_ratio_5_20'] = df['std_5'] / df['std_20']
        
        return df
    
    def calculate_rsi(self, prices: pd.Series, window: int) -> pd.Series:
        """Calculate RSI indicator"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def add_advanced_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add all advanced features.
        """
        logger.info("Adding market microstructure features...")
        df = self.add_market_microstructure_features(df)
        
        logger.info("Adding regime detection features...")
        df = self.add_regime_detection_features(df)
        
        logger.info("Adding multi-timeframe features...")
        df = self.add_multi_timeframe_features(df)
        
        # Clean up any remaining NaNs
        df = df.ffill().bfill()
        
        return df

class EnsemblePredictor:
    """
    Advanced ensemble predictor combining multiple ML models.
    """
    
    def __init__(self):
        self.models = {}
        self.scalers = {}
        self.feature_importance = {}
        self.trained = False
    
    def initialize_models(self):
        """
        Initialize all ML models for the ensemble.
        """
        self.models = {
            'rf': RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42),
            'xgb': xgb.XGBClassifier(n_estimators=200, max_depth=6, learning_rate=0.1, random_state=42),
            'lgb': lgb.LGBMClassifier(n_estimators=200, max_depth=6, learning_rate=0.1, random_state=42),
            'gb': GradientBoostingClassifier(n_estimators=200, max_depth=6, learning_rate=0.1, random_state=42),
            'svm': SVC(kernel='rbf', probability=True, random_state=42),
            'lr': LogisticRegression(random_state=42, max_iter=1000)
        }
        
        # Initialize scalers for each model
        for model_name in self.models.keys():
            if model_name in ['svm', 'lr']:
                self.scalers[model_name] = StandardScaler()
            else:
                self.scalers[model_name] = RobustScaler()
    
    def train_ensemble(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, float]:
        """
        Train all models in the ensemble.
        
        Args:
            X: Feature matrix
            y: Target variable
        
        Returns:
            Dictionary of model accuracies
        """
        self.initialize_models()
        
        # Split data
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        
        model_scores = {}
        
        for model_name, model in self.models.items():
            logger.info(f"Training {model_name.upper()} model...")
            
            try:
                # Scale features if needed
                if model_name in self.scalers:
                    X_train_scaled = self.scalers[model_name].fit_transform(X_train)
                    X_val_scaled = self.scalers[model_name].transform(X_val)
                else:
                    X_train_scaled = X_train
                    X_val_scaled = X_val
                
                # Train model
                model.fit(X_train_scaled, y_train)
                
                # Validate
                y_pred = model.predict(X_val_scaled)
                accuracy = accuracy_score(y_val, y_pred)
                model_scores[model_name] = accuracy
                
                # Store feature importance if available
                if hasattr(model, 'feature_importances_'):
                    self.feature_importance[model_name] = dict(zip(X.columns, model.feature_importances_))
                elif hasattr(model, 'coef_'):
                    self.feature_importance[model_name] = dict(zip(X.columns, abs(model.coef_[0])))
                
                logger.info(f"{model_name.upper()} accuracy: {accuracy:.4f}")
                
            except Exception as e:
                logger.error(f"Error training {model_name}: {e}")
                model_scores[model_name] = 0.0
        
        self.trained = True
        return model_scores
    
    def predict_ensemble(self, X: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Make ensemble predictions.
        
        Args:
            X: Feature matrix
        
        Returns:
            Tuple of (predictions, probabilities)
        """
        if not self.trained:
            raise ValueError("Models not trained yet. Call train_ensemble first.")
        
        predictions = []
        probabilities = []
        
        for model_name, model in self.models.items():
            try:
                # Scale features if needed
                if model_name in self.scalers:
                    X_scaled = self.scalers[model_name].transform(X)
                else:
                    X_scaled = X
                
                # Get predictions and probabilities
                pred = model.predict(X_scaled)
                if hasattr(model, 'predict_proba'):
                    prob = model.predict_proba(X_scaled)[:, 1]  # Probability of class 1
                else:
                    prob = pred  # Use predictions as probabilities for models without predict_proba
                
                predictions.append(pred)
                probabilities.append(prob)
                
            except Exception as e:
                logger.error(f"Error predicting with {model_name}: {e}")
                # Use neutral predictions
                predictions.append(np.zeros(len(X)))
                probabilities.append(np.full(len(X), 0.5))
        
        # Ensemble voting
        predictions_array = np.array(predictions)
        probabilities_array = np.array(probabilities)
        
        # Weighted voting (equal weights for now, could be optimized)
        ensemble_predictions = np.round(np.mean(predictions_array, axis=0)).astype(int)
        ensemble_probabilities = np.mean(probabilities_array, axis=0)
        
        return ensemble_predictions, ensemble_probabilities

class MoneyPrinterStrategy:
    """
    The ultimate "MoneyPrinter" ML trading strategy.
    """
    
    def __init__(self, initial_capital: float = 100000.0):
        """
        Initialize the MoneyPrinter strategy.
        
        Args:
            initial_capital: Initial trading capital
        """
        self.initial_capital = initial_capital
        self.feature_engineer = AdvancedFeatureEngineer()
        self.ensemble_predictor = EnsemblePredictor()
        self.risk_manager = RiskManager(initial_capital)
        
        # Performance tracking
        self.performance_metrics = {}
        self.feature_importance_combined = {}
        
    def prepare_data(self, data_path: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Load and prepare data with advanced features.
        
        Args:
            data_path: Path to market data CSV
        
        Returns:
            Tuple of (features_df, full_df)
        """
        logger.info(f"Loading data from {data_path}...")
        df = load_data(data_path)
        
        logger.info("Adding basic technical features...")
        df = add_features(df)
        
        logger.info("Adding advanced features...")
        df = self.feature_engineer.add_advanced_features(df)
        
        # Create target variable (next period return > threshold)
        return_threshold = 0.001  # 0.1% return threshold
        df['future_return'] = df['close'].pct_change().shift(-1)
        df['target'] = (df['future_return'] > return_threshold).astype(int)
        
        # Remove NaNs
        df_clean = df.dropna()
        
        # Separate features and full data
        feature_cols = [col for col in df_clean.columns if col not in ['target', 'future_return']]
        features_df = df_clean[feature_cols]
        
        logger.info(f"Prepared data: {len(df_clean)} rows, {len(feature_cols)} features")
        
        return features_df, df_clean
    
    def train_models(self, features_df: pd.DataFrame, target: pd.Series) -> Dict[str, float]:
        """
        Train all ensemble models.
        
        Args:
            features_df: Feature matrix
            target: Target variable
        
        Returns:
            Model performance scores
        """
        logger.info("Training ensemble models...")
        model_scores = self.ensemble_predictor.train_ensemble(features_df, target)
        
        # Combine feature importance from all models
        all_features = set()
        for importance_dict in self.ensemble_predictor.feature_importance.values():
            all_features.update(importance_dict.keys())
        
        combined_importance = {}
        for feature in all_features:
            importance_sum = 0
            count = 0
            for importance_dict in self.ensemble_predictor.feature_importance.values():
                if feature in importance_dict:
                    importance_sum += importance_dict[feature]
                    count += 1
            combined_importance[feature] = importance_sum / count if count > 0 else 0
        
        # Sort by importance
        self.feature_importance_combined = dict(
            sorted(combined_importance.items(), key=lambda x: x[1], reverse=True)
        )
        
        logger.info("Top 10 most important features:")
        for i, (feature, importance) in enumerate(list(self.feature_importance_combined.items())[:10]):
            logger.info(f"  {i+1}. {feature}: {importance:.4f}")
        
        return model_scores
    
    def generate_signals(self, features_df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate trading signals using ensemble predictions.
        
        Args:
            features_df: Feature matrix
        
        Returns:
            DataFrame with signals and probabilities
        """
        logger.info("Generating trading signals...")
        
        predictions, probabilities = self.ensemble_predictor.predict_ensemble(features_df)
        
        signals_df = pd.DataFrame(index=features_df.index)
        signals_df['prediction'] = predictions
        signals_df['probability'] = probabilities
        
        # Enhanced signal generation with probability threshold
        prob_threshold = 0.6  # Only trade when confident
        signals_df['signal'] = 0  # 0: hold, 1: buy, 2: sell
        
        # Buy signals (high probability of positive return)
        signals_df.loc[
            (signals_df['prediction'] == 1) & (signals_df['probability'] > prob_threshold), 
            'signal'
        ] = 1
        
        # Sell signals (high probability of negative return)
        signals_df.loc[
            (signals_df['prediction'] == 0) & (signals_df['probability'] < (1 - prob_threshold)), 
            'signal'
        ] = 2
        
        logger.info(f"Generated signals: {(signals_df['signal'] == 1).sum()} buy, {(signals_df['signal'] == 2).sum()} sell")
        
        return signals_df
    
    def run_backtest(self, data_df: pd.DataFrame, signals_df: pd.DataFrame) -> Dict:
        """
        Run backtest with advanced risk management.
        
        Args:
            data_df: Full market data
            signals_df: Trading signals
        
        Returns:
            Backtest results
        """
        logger.info("Running backtest...")
        
        # Align data and signals
        aligned_data = data_df.loc[signals_df.index]
        signals_series = signals_df['signal']
        
        # Run standard backtest
        backtester = Backtester(
            data=aligned_data, 
            signals=signals_series,
            initial_capital=self.initial_capital
        )
        
        metrics = backtester.run()
        
        # Calculate additional metrics
        returns = backtester.portfolio['strategy_returns'].dropna()
        
        additional_metrics = {
            'Win Rate': f"{(returns > 0).mean():.2%}",
            'Loss Rate': f"{(returns < 0).mean():.2%}",
            'Average Win': f"{returns[returns > 0].mean():.4f}",
            'Average Loss': f"{returns[returns < 0].mean():.4f}",
            'Profit Factor': f"{returns[returns > 0].sum() / abs(returns[returns < 0].sum()):.2f}",
            'Max Consecutive Wins': str(self._max_consecutive(returns > 0)),
            'Max Consecutive Losses': str(self._max_consecutive(returns < 0)),
            'Recovery Factor': f"{float(metrics['Total Return'].rstrip('%')) / abs(float(metrics['Max Drawdown'].rstrip('%'))):.2f}",
        }
        
        # Combine all metrics
        all_metrics = {**metrics, **additional_metrics}
        
        self.performance_metrics = all_metrics
        
        return all_metrics
    
    def _max_consecutive(self, series: pd.Series) -> int:
        """Calculate maximum consecutive True values"""
        return series.astype(int).groupby((~series).cumsum()).sum().max()
    
    def run_full_strategy(self, data_path: str) -> Dict:
        """
        Run the complete MoneyPrinter strategy.
        
        Args:
            data_path: Path to market data
        
        Returns:
            Complete results and metrics
        """
        logger.info("🚀 Starting MoneyPrinter Strategy...")
        
        # 1. Prepare data
        features_df, full_df = self.prepare_data(data_path)
        
        # 2. Train models
        model_scores = self.train_models(features_df, full_df['target'])
        
        # 3. Generate signals
        signals_df = self.generate_signals(features_df)
        
        # 4. Run backtest
        backtest_results = self.run_backtest(full_df, signals_df)
        
        # 5. Compile results
        results = {
            'model_scores': model_scores,
            'backtest_results': backtest_results,
            'feature_importance': dict(list(self.feature_importance_combined.items())[:20]),  # Top 20
            'data_info': {
                'total_samples': len(full_df),
                'features_count': len(features_df.columns),
                'positive_signals': (signals_df['signal'] == 1).sum(),
                'negative_signals': (signals_df['signal'] == 2).sum(),
                'hold_signals': (signals_df['signal'] == 0).sum(),
            }
        }
        
        logger.info("💰 MoneyPrinter Strategy Complete!")
        self._print_results(results)
        
        return results
    
    def _print_results(self, results: Dict):
        """Print formatted results"""
        print("\n" + "="*60)
        print("🏆 MONEYPRINTER STRATEGY RESULTS 🏆")
        print("="*60)
        
        print("\n📊 Model Performance:")
        for model, score in results['model_scores'].items():
            print(f"  {model.upper()}: {score:.4f}")
        
        print(f"\n📈 Backtest Results:")
        for metric, value in results['backtest_results'].items():
            print(f"  {metric}: {value}")
        
        print(f"\n🔥 Top Features:")
        for i, (feature, importance) in enumerate(list(results['feature_importance'].items())[:10]):
            print(f"  {i+1}. {feature}: {importance:.4f}")
        
        print(f"\n📋 Data Summary:")
        for key, value in results['data_info'].items():
            print(f"  {key}: {value}")
        
        print("="*60)

if __name__ == '__main__':
    # Initialize the MoneyPrinter
    money_printer = MoneyPrinterStrategy(initial_capital=100000.0)
    
    # Run the complete strategy
    results = money_printer.run_full_strategy('data/market_data.csv')
    
    print("\n🎯 MoneyPrinter Strategy is now operational and ready to generate alpha! 💰")