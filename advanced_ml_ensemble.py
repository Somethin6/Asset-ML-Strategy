#!/usr/bin/env python3
"""
🚀 ADVANCED ML ENSEMBLE SYSTEM 🚀
Implements 15+ advanced ML models with intelligent ensemble methods
The most sophisticated ML trading system available!
"""

import pandas as pd
import numpy as np
import warnings
from typing import Dict, List, Optional, Tuple, Any
import logging
from sklearn.ensemble import (
    RandomForestRegressor, GradientBoostingRegressor, 
    ExtraTreesRegressor, AdaBoostRegressor, BaggingRegressor
)
from sklearn.linear_model import (
    LinearRegression, Ridge, Lasso, ElasticNet, 
    HuberRegressor, RANSACRegressor
)
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import joblib
import time

warnings.filterwarnings('ignore')

class AdvancedMLEnsemble:
    """
    Advanced ML ensemble system with 15+ models and intelligent voting
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.models = {}
        self.scalers = {}
        self.model_performances = {}
        self.ensemble_weights = {}
        self.feature_importance = {}
        self.training_history = []
        
    def initialize_models(self):
        """Initialize all 15+ advanced ML models"""
        self.logger.info("🚀 Initializing 15+ advanced ML models...")
        
        # Try to import advanced models
        try:
            import xgboost as xgb
            xgb_available = True
        except ImportError:
            xgb_available = False
            self.logger.warning("XGBoost not available - install with: pip install xgboost")
        
        try:
            import lightgbm as lgb
            lgb_available = True
        except ImportError:
            lgb_available = False
            self.logger.warning("LightGBM not available - install with: pip install lightgbm")
        
        # Tree-based models
        self.models.update({
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
            'extra_trees': ExtraTreesRegressor(
                n_estimators=200,
                max_depth=15,
                min_samples_split=5,
                random_state=42,
                n_jobs=-1
            ),
            'ada_boost': AdaBoostRegressor(
                n_estimators=100,
                learning_rate=0.1,
                random_state=42
            )
        })
        
        # Add XGBoost if available
        if xgb_available:
            self.models['xgboost'] = xgb.XGBRegressor(
                n_estimators=200,
                learning_rate=0.1,
                max_depth=6,
                min_child_weight=1,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                n_jobs=-1
            )
        
        # Add LightGBM if available
        if lgb_available:
            self.models['lightgbm'] = lgb.LGBMRegressor(
                n_estimators=200,
                learning_rate=0.1,
                max_depth=6,
                num_leaves=31,
                min_child_samples=20,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                n_jobs=-1,
                verbosity=-1
            )
        
        # Linear models
        self.models.update({
            'linear_regression': LinearRegression(),
            'ridge': Ridge(alpha=1.0, random_state=42),
            'lasso': Lasso(alpha=0.1, random_state=42, max_iter=2000),
            'elastic_net': ElasticNet(alpha=0.1, l1_ratio=0.5, random_state=42, max_iter=2000),
            'huber': HuberRegressor(epsilon=1.35, max_iter=500),
            'ransac': RANSACRegressor(random_state=42, max_trials=100)
        })
        
        # Support Vector Machine
        self.models['svr'] = SVR(kernel='rbf', gamma='scale', C=1.0)
        
        # K-Nearest Neighbors
        self.models['knn'] = KNeighborsRegressor(n_neighbors=5, weights='distance')
        
        # Neural Network
        self.models['neural_network'] = MLPRegressor(
            hidden_layer_sizes=(100, 50),
            activation='relu',
            solver='adam',
            alpha=0.001,
            learning_rate='adaptive',
            max_iter=500,
            random_state=42
        )
        
        # Decision Tree
        self.models['decision_tree'] = DecisionTreeRegressor(
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42
        )
        
        # Gaussian Process (for smaller datasets)
        self.models['gaussian_process'] = GaussianProcessRegressor(
            alpha=1e-6,
            normalize_y=True,
            random_state=42
        )
        
        # Bagging ensemble
        self.models['bagging'] = BaggingRegressor(
            n_estimators=50,
            random_state=42,
            n_jobs=-1
        )
        
        self.logger.info(f"✅ Initialized {len(self.models)} ML models!")
        
    def prepare_data(self, X: pd.DataFrame, y: pd.Series, test_size: float = 0.2) -> Tuple:
        """Prepare and scale data for training"""
        # Handle missing values
        X_clean = X.fillna(X.mean())
        y_clean = y.fillna(y.mean())
        
        # Align indices
        common_idx = X_clean.index.intersection(y_clean.index)
        X_clean = X_clean.loc[common_idx]
        y_clean = y_clean.loc[common_idx]
        
        # Time-series split
        split_idx = int(len(X_clean) * (1 - test_size))
        
        X_train = X_clean.iloc[:split_idx]
        X_test = X_clean.iloc[split_idx:]
        y_train = y_clean.iloc[:split_idx]
        y_test = y_clean.iloc[split_idx:]
        
        # Scale features for models that need it
        scaler_models = ['neural_network', 'svr', 'knn', 'gaussian_process']
        
        self.scalers = {
            'standard': StandardScaler(),
            'robust': RobustScaler(),
            'minmax': MinMaxScaler()
        }
        
        # Fit scalers on training data
        for scaler_name, scaler in self.scalers.items():
            scaler.fit(X_train)
        
        return X_train, X_test, y_train, y_test
    
    def train_all_models(self, X_train: pd.DataFrame, y_train: pd.Series, 
                        X_test: pd.DataFrame = None, y_test: pd.Series = None) -> Dict:
        """Train all models and evaluate performance"""
        self.logger.info("🔥 Training all ML models...")
        
        results = {}
        training_start = time.time()
        
        for model_name, model in self.models.items():
            try:
                start_time = time.time()
                self.logger.info(f"Training {model_name}...")
                
                # Determine if scaling is needed
                if model_name in ['neural_network', 'svr', 'knn', 'gaussian_process']:
                    X_train_scaled = self.scalers['standard'].transform(X_train)
                    X_test_scaled = self.scalers['standard'].transform(X_test) if X_test is not None else None
                else:
                    X_train_scaled = X_train
                    X_test_scaled = X_test
                
                # Train model
                if model_name == 'gaussian_process' and len(X_train) > 1000:
                    # Skip GP for large datasets
                    self.logger.warning(f"Skipping {model_name} - dataset too large")
                    continue
                    
                model.fit(X_train_scaled, y_train)
                
                # Make predictions
                y_train_pred = model.predict(X_train_scaled)
                
                # Calculate training metrics
                train_r2 = r2_score(y_train, y_train_pred)
                train_mse = mean_squared_error(y_train, y_train_pred)
                train_mae = mean_absolute_error(y_train, y_train_pred)
                
                # Test metrics if test data provided
                test_r2, test_mse, test_mae = None, None, None
                if X_test is not None and y_test is not None:
                    y_test_pred = model.predict(X_test_scaled)
                    test_r2 = r2_score(y_test, y_test_pred)
                    test_mse = mean_squared_error(y_test, y_test_pred)
                    test_mae = mean_absolute_error(y_test, y_test_pred)
                
                # Cross-validation score
                try:
                    cv_scores = cross_val_score(
                        model, X_train_scaled, y_train, 
                        cv=TimeSeriesSplit(n_splits=3), 
                        scoring='r2', n_jobs=1
                    )
                    cv_mean = cv_scores.mean()
                    cv_std = cv_scores.std()
                except:
                    cv_mean, cv_std = None, None
                
                # Feature importance (if available)
                feature_importance = None
                if hasattr(model, 'feature_importances_'):
                    feature_importance = dict(zip(X_train.columns, model.feature_importances_))
                elif hasattr(model, 'coef_'):
                    feature_importance = dict(zip(X_train.columns, abs(model.coef_)))
                
                training_time = time.time() - start_time
                
                # Store results
                results[model_name] = {
                    'model': model,
                    'train_r2': train_r2,
                    'train_mse': train_mse,
                    'train_mae': train_mae,
                    'test_r2': test_r2,
                    'test_mse': test_mse,
                    'test_mae': test_mae,
                    'cv_mean': cv_mean,
                    'cv_std': cv_std,
                    'feature_importance': feature_importance,
                    'training_time': training_time,
                    'status': 'success'
                }
                
                self.logger.info(f"✅ {model_name}: R²={train_r2:.4f}, Time={training_time:.2f}s")
                
            except Exception as e:
                self.logger.error(f"❌ {model_name} failed: {e}")
                results[model_name] = {
                    'status': 'failed',
                    'error': str(e)
                }
        
        total_time = time.time() - training_start
        self.logger.info(f"🎉 All models trained in {total_time:.2f} seconds!")
        
        # Store results
        self.model_performances = results
        
        return results
    
    def create_ensemble(self, X_test: pd.DataFrame = None, y_test: pd.Series = None, 
                       method: str = 'weighted_avg') -> Dict:
        """Create ensemble predictions using different methods"""
        self.logger.info(f"🎯 Creating ensemble using {method} method...")
        
        successful_models = {k: v for k, v in self.model_performances.items() 
                           if v.get('status') == 'success'}
        
        if len(successful_models) < 2:
            self.logger.error("Need at least 2 successful models for ensemble")
            return {}
        
        # Calculate ensemble weights based on performance
        if method == 'weighted_avg':
            weights = {}
            
            # Use cross-validation scores if available, otherwise use test R²
            for model_name, results in successful_models.items():
                if results.get('cv_mean') is not None:
                    score = max(0, results['cv_mean'])  # Ensure non-negative
                elif results.get('test_r2') is not None:
                    score = max(0, results['test_r2'])
                else:
                    score = max(0, results.get('train_r2', 0))
                
                weights[model_name] = score
            
            # Normalize weights
            total_weight = sum(weights.values())
            if total_weight > 0:
                weights = {k: v/total_weight for k, v in weights.items()}
            else:
                weights = {k: 1/len(weights) for k in weights.keys()}
        
        elif method == 'equal_weight':
            weights = {k: 1/len(successful_models) for k in successful_models.keys()}
        
        elif method == 'top_performers':
            # Use only top 5 performers
            if 'cv_mean' in list(successful_models.values())[0]:
                sorted_models = sorted(successful_models.items(), 
                                     key=lambda x: x[1].get('cv_mean', 0), reverse=True)
            else:
                sorted_models = sorted(successful_models.items(), 
                                     key=lambda x: x[1].get('test_r2', x[1].get('train_r2', 0)), reverse=True)
            
            top_models = dict(sorted_models[:5])
            weights = {k: 1/len(top_models) for k in top_models.keys()}
        
        self.ensemble_weights = weights
        
        # Create ensemble performance summary
        ensemble_info = {
            'method': method,
            'weights': weights,
            'num_models': len(weights),
            'models_used': list(weights.keys())
        }
        
        self.logger.info(f"✅ Ensemble created with {len(weights)} models")
        self.logger.info(f"Top contributors: {sorted(weights.items(), key=lambda x: x[1], reverse=True)[:3]}")
        
        return ensemble_info
    
    def predict_ensemble(self, X: pd.DataFrame) -> np.ndarray:
        """Make ensemble predictions"""
        if not self.ensemble_weights:
            raise ValueError("Ensemble not created. Call create_ensemble() first.")
        
        predictions = {}
        
        # Get predictions from each model
        for model_name, weight in self.ensemble_weights.items():
            if model_name in self.model_performances and self.model_performances[model_name].get('status') == 'success':
                model = self.model_performances[model_name]['model']
                
                # Scale data if needed
                if model_name in ['neural_network', 'svr', 'knn', 'gaussian_process']:
                    X_scaled = self.scalers['standard'].transform(X)
                else:
                    X_scaled = X
                
                pred = model.predict(X_scaled)
                predictions[model_name] = pred * weight
        
        # Combine weighted predictions
        ensemble_pred = np.sum(list(predictions.values()), axis=0)
        
        return ensemble_pred
    
    def get_model_rankings(self) -> pd.DataFrame:
        """Get ranking of models by performance"""
        successful_models = {k: v for k, v in self.model_performances.items() 
                           if v.get('status') == 'success'}
        
        rankings = []
        for model_name, results in successful_models.items():
            rankings.append({
                'Model': model_name,
                'Train_R2': results.get('train_r2', 0),
                'Test_R2': results.get('test_r2', 0),
                'CV_Mean': results.get('cv_mean', 0),
                'CV_Std': results.get('cv_std', 0),
                'Training_Time': results.get('training_time', 0),
                'Ensemble_Weight': self.ensemble_weights.get(model_name, 0)
            })
        
        df = pd.DataFrame(rankings)
        
        # Sort by CV score if available, otherwise by test R²
        if df['CV_Mean'].notna().any():
            df = df.sort_values('CV_Mean', ascending=False)
        else:
            df = df.sort_values('Test_R2', ascending=False)
        
        return df
    
    def generate_model_report(self) -> str:
        """Generate comprehensive model performance report"""
        rankings = self.get_model_rankings()
        
        report = f"""
🚀 ADVANCED ML ENSEMBLE SYSTEM REPORT 🚀
{'='*60}

📊 MODEL PERFORMANCE SUMMARY:
Total Models Trained: {len(self.model_performances)}
Successful Models: {len([v for v in self.model_performances.values() if v.get('status') == 'success'])}
Failed Models: {len([v for v in self.model_performances.values() if v.get('status') == 'failed'])}

🏆 TOP PERFORMING MODELS:
"""
        
        for i, (_, row) in enumerate(rankings.head(10).iterrows()):
            cv_score = f"{row['CV_Mean']:.4f}±{row['CV_Std']:.4f}" if pd.notna(row['CV_Mean']) else "N/A"
            report += f"{i+1:2d}. {row['Model']:<18} | R²: {row['Test_R2']:.4f} | CV: {cv_score} | Weight: {row['Ensemble_Weight']:.4f}\n"
        
        report += f"\n🎯 ENSEMBLE CONFIGURATION:\n"
        if self.ensemble_weights:
            report += f"Method: Weighted Average\n"
            report += f"Models Used: {len(self.ensemble_weights)}\n"
            report += f"Top Contributors:\n"
            for model, weight in sorted(self.ensemble_weights.items(), key=lambda x: x[1], reverse=True)[:5]:
                report += f"  • {model}: {weight:.4f}\n"
        else:
            report += "Ensemble not created yet.\n"
        
        # Add failed models if any
        failed_models = [k for k, v in self.model_performances.items() if v.get('status') == 'failed']
        if failed_models:
            report += f"\n❌ FAILED MODELS:\n"
            for model in failed_models:
                error = self.model_performances[model].get('error', 'Unknown error')
                report += f"  • {model}: {error}\n"
        
        report += f"\n✅ SYSTEM READY FOR ADVANCED PREDICTIONS! 🚀\n"
        
        return report
    
    def save_ensemble(self, filepath: str):
        """Save the entire ensemble system"""
        ensemble_data = {
            'models': self.model_performances,
            'scalers': self.scalers,
            'ensemble_weights': self.ensemble_weights,
            'training_history': self.training_history
        }
        joblib.dump(ensemble_data, filepath)
        self.logger.info(f"Ensemble saved to {filepath}")
    
    def load_ensemble(self, filepath: str):
        """Load a saved ensemble system"""
        ensemble_data = joblib.load(filepath)
        self.model_performances = ensemble_data['models']
        self.scalers = ensemble_data['scalers']
        self.ensemble_weights = ensemble_data['ensemble_weights']
        self.training_history = ensemble_data.get('training_history', [])
        self.logger.info(f"Ensemble loaded from {filepath}")


def main():
    """Demo of advanced ML ensemble system"""
    print("🚀 ADVANCED ML ENSEMBLE SYSTEM DEMO 🚀")
    print("=" * 60)
    
    # Create sample data with features
    np.random.seed(42)
    n_samples = 1000
    n_features = 20
    
    # Generate feature matrix
    X = pd.DataFrame(
        np.random.randn(n_samples, n_features),
        columns=[f'Feature_{i}' for i in range(n_features)]
    )
    
    # Generate target with some relationship to features
    y = (X.iloc[:, :5].sum(axis=1) + 
         0.5 * X.iloc[:, 5:10].sum(axis=1) + 
         np.random.randn(n_samples) * 0.1)
    y.name = 'Target'
    
    print(f"📊 Dataset: {X.shape[0]} samples, {X.shape[1]} features")
    
    # Initialize ensemble system
    ensemble = AdvancedMLEnsemble()
    ensemble.initialize_models()
    
    # Prepare data
    X_train, X_test, y_train, y_test = ensemble.prepare_data(X, y, test_size=0.2)
    print(f"Train: {len(X_train)}, Test: {len(X_test)}")
    
    # Train all models
    results = ensemble.train_all_models(X_train, y_train, X_test, y_test)
    
    # Create ensemble
    ensemble_info = ensemble.create_ensemble(X_test, y_test, method='weighted_avg')
    
    # Make ensemble predictions
    ensemble_pred = ensemble.predict_ensemble(X_test)
    ensemble_r2 = r2_score(y_test, ensemble_pred)
    ensemble_mse = mean_squared_error(y_test, ensemble_pred)
    
    print(f"\n🎯 ENSEMBLE PERFORMANCE:")
    print(f"R² Score: {ensemble_r2:.4f}")
    print(f"MSE: {ensemble_mse:.4f}")
    
    # Generate report
    report = ensemble.generate_model_report()
    print(report)
    
    print("\n🚀 ADVANCED ML ENSEMBLE SYSTEM COMPLETE!")
    print("Ready for production trading! 💰")

if __name__ == "__main__":
    main()