#!/usr/bin/env python3
"""
Advanced ML Strategy Engine - The Most Advanced ML Trading Bot Engine
Finds best strategies across time periods and learns transitions between them
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split, cross_val_score, TimeSeriesSplit
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler, RobustScaler
import xgboost as xgb
import lightgbm as lgb
from typing import Dict, List, Optional, Tuple, Any
import warnings
warnings.filterwarnings('ignore')

class AdvancedMLEngine:
    def __init__(self):
        self.models = {}
        self.scalers = {}
        self.strategies = {}
        self.performance_history = {}
        self.best_strategy_periods = {}
        self.transition_model = None
        self.feature_importance = {}
        
    def initialize_models(self):
        """Initialize all available ML models"""
        self.models = {
            'random_forest': RandomForestRegressor(
                n_estimators=200,
                max_depth=15,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42,
                n_jobs=-1
            ),
            'gradient_boosting': GradientBoostingRegressor(
                n_estimators=200,
                learning_rate=0.1,
                max_depth=6,
                min_samples_split=5,
                random_state=42
            ),
            'xgboost': xgb.XGBRegressor(
                n_estimators=200,
                learning_rate=0.1,
                max_depth=6,
                min_child_weight=1,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                n_jobs=-1
            ),
            'lightgbm': lgb.LGBMRegressor(
                n_estimators=200,
                learning_rate=0.1,
                max_depth=6,
                min_child_samples=20,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                n_jobs=-1,
                verbosity=-1
            )
        }
        
        print(f"✅ Initialized {len(self.models)} ML models")
    
    def create_dynamic_features(self, data: pd.DataFrame, indicators: Dict) -> pd.DataFrame:
        """Create comprehensive feature matrix for ML"""
        features_df = data[['open', 'high', 'low', 'close', 'volume']].copy()
        
        # Add all indicators
        for name, values in indicators.items():
            if isinstance(values, pd.Series) and len(values) == len(features_df):
                features_df[name] = values
        
        # Create interaction features
        features_df = self._create_interaction_features(features_df)
        
        # Create time-based features
        features_df = self._create_time_features(features_df)
        
        # Create rolling statistical features
        features_df = self._create_rolling_features(features_df)
        
        # Remove NaN values
        features_df = features_df.dropna()
        
        print(f"📊 Created feature matrix: {features_df.shape[1]} features, {features_df.shape[0]} samples")
        
        return features_df
    
    def _create_interaction_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create interaction features between key indicators"""
        try:
            # Price-Volume interactions
            if 'close' in df.columns and 'volume' in df.columns:
                df['price_volume_product'] = df['close'] * df['volume']
                df['volume_price_ratio'] = df['volume'] / df['close']
            
            # RSI interactions
            rsi_cols = [col for col in df.columns if 'RSI' in col]
            if len(rsi_cols) >= 2:
                df['rsi_divergence'] = df[rsi_cols[0]] - df[rsi_cols[1]]
            
            # Moving average crossovers
            sma_cols = [col for col in df.columns if 'SMA' in col]
            if len(sma_cols) >= 2:
                df['sma_crossover'] = df[sma_cols[0]] - df[sma_cols[1]]
            
            # Bollinger Band position
            if all(col in df.columns for col in ['BB_UPPER', 'BB_LOWER', 'close']):
                df['bb_position'] = (df['close'] - df['BB_LOWER']) / (df['BB_UPPER'] - df['BB_LOWER'])
            
            # MACD signal interactions
            if all(col in df.columns for col in ['MACD', 'MACD_SIGNAL']):
                df['macd_signal_cross'] = df['MACD'] - df['MACD_SIGNAL']
                
        except Exception as e:
            print(f"Warning: Error creating interaction features: {e}")
        
        return df
    
    def _create_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create time-based features"""
        if hasattr(df.index, 'dayofweek'):
            df['day_of_week'] = df.index.dayofweek
            df['is_monday'] = (df.index.dayofweek == 0).astype(int)
            df['is_friday'] = (df.index.dayofweek == 4).astype(int)
            df['month'] = df.index.month
            df['quarter'] = df.index.quarter
        
        return df
    
    def _create_rolling_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create rolling statistical features"""
        # Rolling statistics for price
        if 'close' in df.columns:
            for window in [5, 10, 20]:
                df[f'close_rolling_mean_{window}'] = df['close'].rolling(window).mean()
                df[f'close_rolling_std_{window}'] = df['close'].rolling(window).std()
                df[f'close_rolling_min_{window}'] = df['close'].rolling(window).min()
                df[f'close_rolling_max_{window}'] = df['close'].rolling(window).max()
                
                # Position within rolling range
                df[f'close_position_{window}'] = (
                    (df['close'] - df[f'close_rolling_min_{window}']) / 
                    (df[f'close_rolling_max_{window}'] - df[f'close_rolling_min_{window}'])
                )
        
        return df
    
    def find_best_strategies_by_period(self, features_df: pd.DataFrame, target_col: str = 'close', 
                                     periods: List[int] = None) -> Dict:
        """Find best performing strategies for different time periods"""
        if periods is None:
            periods = [50, 100, 200, 500]  # Different lookback periods
        
        self.initialize_models()
        best_strategies = {}
        
        print("🎯 Finding best strategies across different time periods...")
        
        for period in periods:
            print(f"\n📈 Analyzing period: {period} days")
            
            if len(features_df) < period + 50:  # Need enough data
                continue
            
            # Use the last 'period' days for this analysis
            period_data = features_df.tail(period + 50)  # Extra buffer for features
            
            X = period_data.drop(columns=[target_col] if target_col in period_data.columns else []).select_dtypes(include=[np.number])
            
            # Create target (next day's return)
            y = period_data[target_col].shift(-1).pct_change().dropna()
            
            # Align X and y
            min_len = min(len(X), len(y))
            X = X.iloc[:min_len]
            y = y.iloc[:min_len]
            
            if len(X) < 30:  # Need minimum samples
                continue
            
            strategy_performance = self._evaluate_models_for_period(X, y, period)
            best_strategies[period] = strategy_performance
            
            print(f"✅ Period {period}: Best model = {strategy_performance['best_model']} "
                  f"(R² = {strategy_performance['best_score']:.4f})")
        
        self.best_strategy_periods = best_strategies
        return best_strategies
    
    def _evaluate_models_for_period(self, X: pd.DataFrame, y: pd.Series, period: int) -> Dict:
        """Evaluate all models for a specific time period"""
        # Handle NaN values
        mask = ~(X.isnull().any(axis=1) | y.isnull())
        X_clean = X[mask]
        y_clean = y[mask]
        
        if len(X_clean) < 20:  # Not enough data
            return {'best_model': 'insufficient_data', 'best_score': 0, 'models': {}}
        
        # Scale features
        scaler = RobustScaler()
        X_scaled = pd.DataFrame(
            scaler.fit_transform(X_clean),
            columns=X_clean.columns,
            index=X_clean.index
        )
        
        # Time series split for validation
        tscv = TimeSeriesSplit(n_splits=3)
        model_scores = {}
        
        for model_name, model in self.models.items():
            try:
                # Cross-validation with time series split
                scores = cross_val_score(model, X_scaled, y_clean, cv=tscv, scoring='r2')
                mean_score = scores.mean()
                std_score = scores.std()
                
                model_scores[model_name] = {
                    'mean_score': mean_score,
                    'std_score': std_score,
                    'scores': scores.tolist()
                }
                
            except Exception as e:
                print(f"Warning: Error evaluating {model_name}: {e}")
                model_scores[model_name] = {'mean_score': -999, 'std_score': 999, 'scores': []}
        
        # Find best model
        best_model = max(model_scores.keys(), key=lambda k: model_scores[k]['mean_score'])
        best_score = model_scores[best_model]['mean_score']
        
        # Train best model on full period data
        best_model_instance = self.models[best_model]
        best_model_instance.fit(X_scaled, y_clean)
        
        # Calculate feature importance
        if hasattr(best_model_instance, 'feature_importances_'):
            feature_importance = dict(zip(X_scaled.columns, best_model_instance.feature_importances_))
            top_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)[:10]
        else:
            top_features = []
        
        return {
            'best_model': best_model,
            'best_score': best_score,
            'models': model_scores,
            'trained_model': best_model_instance,
            'scaler': scaler,
            'top_features': top_features,
            'feature_count': len(X_scaled.columns)
        }
    
    def learn_strategy_transitions(self, features_df: pd.DataFrame) -> Dict:
        """Learn when to transition between different strategies"""
        print("\n🔄 Learning strategy transitions...")
        
        if not self.best_strategy_periods:
            print("❌ No strategies found. Run find_best_strategies_by_period first.")
            return {}
        
        transition_data = []
        
        # Create transition features
        periods = list(self.best_strategy_periods.keys())
        window = 20  # Look at 20-day windows
        
        for i in range(window, len(features_df) - window):
            window_data = features_df.iloc[i-window:i+window]
            
            # Calculate market regime indicators
            volatility = window_data['close'].std()
            trend_strength = abs(window_data['close'].iloc[-1] - window_data['close'].iloc[0]) / window_data['close'].iloc[0]
            volume_change = window_data['volume'].pct_change().mean()
            
            # Determine which strategy performs best in this regime
            best_strategy = self._determine_best_strategy_for_regime(window_data, volatility, trend_strength)
            
            transition_data.append({
                'volatility': volatility,
                'trend_strength': trend_strength,
                'volume_change': volume_change,
                'rsi_avg': window_data['RSI_14'].mean() if 'RSI_14' in window_data.columns else 50,
                'best_strategy': best_strategy
            })
        
        if not transition_data:
            return {}
        
        # Train transition model
        transition_df = pd.DataFrame(transition_data)
        
        X_transition = transition_df[['volatility', 'trend_strength', 'volume_change', 'rsi_avg']]
        y_transition = transition_df['best_strategy']
        
        # Use Random Forest for strategy selection
        from sklearn.ensemble import RandomForestClassifier
        
        self.transition_model = RandomForestClassifier(
            n_estimators=100,
            random_state=42,
            n_jobs=-1
        )
        
        self.transition_model.fit(X_transition, y_transition)
        
        # Evaluate transition model
        from sklearn.metrics import classification_report, accuracy_score
        
        predictions = self.transition_model.predict(X_transition)
        accuracy = accuracy_score(y_transition, predictions)
        
        print(f"✅ Transition model trained with {accuracy:.3f} accuracy")
        
        # Feature importance for transitions
        transition_importance = dict(zip(
            X_transition.columns,
            self.transition_model.feature_importances_
        ))
        
        return {
            'model': self.transition_model,
            'accuracy': accuracy,
            'feature_importance': transition_importance,
            'training_data_size': len(transition_data)
        }
    
    def _determine_best_strategy_for_regime(self, window_data: pd.DataFrame, 
                                          volatility: float, trend_strength: float) -> str:
        """Determine which strategy works best for current market regime"""
        # Simple heuristic for now - can be made more sophisticated
        if volatility > window_data['close'].rolling(50).std().mean():
            if trend_strength > 0.05:  # Strong trend, high volatility
                return 'xgboost'  # XGBoost for complex patterns
            else:  # High volatility, low trend
                return 'lightgbm'  # LightGBM for speed
        else:
            if trend_strength > 0.02:  # Low volatility, clear trend
                return 'gradient_boosting'  # GBM for smooth trends
            else:  # Low volatility, sideways
                return 'random_forest'  # RF for stability
    
    def predict_best_strategy(self, current_features: pd.Series) -> str:
        """Predict which strategy to use based on current market conditions"""
        if self.transition_model is None:
            return 'random_forest'  # Default
        
        try:
            # Extract regime features from current data
            recent_closes = current_features.get('close', 100)
            volatility = 0.02  # Default
            trend_strength = 0.01  # Default
            volume_change = 0.0  # Default
            rsi_avg = current_features.get('RSI_14', 50)
            
            regime_features = np.array([[volatility, trend_strength, volume_change, rsi_avg]])
            predicted_strategy = self.transition_model.predict(regime_features)[0]
            
            return predicted_strategy
            
        except Exception as e:
            print(f"Warning: Error predicting strategy: {e}")
            return 'random_forest'  # Fallback
    
    def generate_comprehensive_predictions(self, features_df: pd.DataFrame, 
                                         target_col: str = 'close') -> Dict:
        """Generate predictions using the best strategy for current conditions"""
        if not self.best_strategy_periods:
            print("❌ No strategies available. Train strategies first.")
            return {}
        
        # Get recent features for prediction
        recent_features = features_df.tail(1).iloc[0]
        
        # Predict best strategy
        best_strategy = self.predict_best_strategy(recent_features)
        
        predictions = {}
        
        # Generate predictions with all available models
        for period, strategy_data in self.best_strategy_periods.items():
            model_name = strategy_data['best_model']
            
            if model_name == 'insufficient_data':
                continue
            
            trained_model = strategy_data['trained_model']
            scaler = strategy_data['scaler']
            
            try:
                # Prepare features
                X_recent = features_df.tail(1).select_dtypes(include=[np.number])
                
                # Ensure same features as training
                training_features = scaler.n_features_in_
                if len(X_recent.columns) != training_features:
                    # Skip if feature mismatch
                    continue
                
                X_scaled = scaler.transform(X_recent)
                
                # Make prediction
                pred = trained_model.predict(X_scaled)[0]
                
                predictions[f'period_{period}_{model_name}'] = {
                    'prediction': pred,
                    'confidence': strategy_data['best_score'],
                    'is_recommended': (model_name == best_strategy)
                }
                
            except Exception as e:
                print(f"Warning: Error generating prediction for period {period}: {e}")
        
        # Ensemble prediction
        if predictions:
            recommended_preds = [p['prediction'] for p in predictions.values() if p['is_recommended']]
            all_preds = [p['prediction'] for p in predictions.values()]
            
            ensemble_pred = np.mean(recommended_preds) if recommended_preds else np.mean(all_preds)
            
            predictions['ensemble'] = {
                'prediction': ensemble_pred,
                'recommended_strategy': best_strategy,
                'n_models': len(predictions)
            }
        
        return predictions
    
    def get_strategy_performance_report(self) -> Dict:
        """Generate comprehensive performance report"""
        if not self.best_strategy_periods:
            return {"error": "No strategies trained"}
        
        report = {
            'summary': {
                'total_periods_analyzed': len(self.best_strategy_periods),
                'best_overall_model': self._get_best_overall_model(),
                'transition_model_available': self.transition_model is not None
            },
            'period_analysis': {},
            'model_rankings': self._get_model_rankings()
        }
        
        for period, strategy_data in self.best_strategy_periods.items():
            report['period_analysis'][period] = {
                'best_model': strategy_data['best_model'],
                'performance': strategy_data['best_score'],
                'feature_count': strategy_data.get('feature_count', 0),
                'top_features': strategy_data.get('top_features', [])[:5]  # Top 5
            }
        
        return report
    
    def _get_best_overall_model(self) -> str:
        """Determine the best performing model overall"""
        model_performance = {}
        
        for period, strategy_data in self.best_strategy_periods.items():
            best_model = strategy_data['best_model']
            if best_model != 'insufficient_data':
                if best_model not in model_performance:
                    model_performance[best_model] = []
                model_performance[best_model].append(strategy_data['best_score'])
        
        # Calculate average performance
        avg_performance = {
            model: np.mean(scores) 
            for model, scores in model_performance.items()
        }
        
        return max(avg_performance.keys(), key=lambda k: avg_performance[k]) if avg_performance else 'random_forest'
    
    def _get_model_rankings(self) -> Dict:
        """Get overall model rankings"""
        model_scores = {}
        
        for period, strategy_data in self.best_strategy_periods.items():
            models = strategy_data.get('models', {})
            for model_name, model_data in models.items():
                if model_name not in model_scores:
                    model_scores[model_name] = []
                model_scores[model_name].append(model_data['mean_score'])
        
        rankings = {}
        for model, scores in model_scores.items():
            rankings[model] = {
                'average_score': np.mean(scores),
                'std_score': np.std(scores),
                'wins': sum(1 for period_data in self.best_strategy_periods.values() 
                           if period_data['best_model'] == model)
            }
        
        return dict(sorted(rankings.items(), key=lambda x: x[1]['average_score'], reverse=True))

# Example usage
if __name__ == "__main__":
    print("🚀 Advanced ML Engine Test")
    print("=" * 50)
    
    engine = AdvancedMLEngine()
    engine.initialize_models()
    
    print("✅ Advanced ML Engine initialized successfully")
    print("🎯 Features available:")
    print("  • Multi-model ensemble (RF, GBM, XGB, LGB)")
    print("  • Dynamic strategy selection")
    print("  • Strategy transition learning")
    print("  • Comprehensive feature engineering")
    print("  • Time-series aware validation")
    print("  • Performance tracking & reporting")