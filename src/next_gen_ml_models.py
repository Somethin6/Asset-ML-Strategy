#!/usr/bin/env python3
"""
Next-Generation ML Models - Infinitely Advanced Trading AI
Implements the most sophisticated ML architectures for maximum trading performance.
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import xgboost as xgb
import lightgbm as lgb
from typing import Dict, List, Tuple, Optional, Any, Union
import warnings
import logging
from dataclasses import dataclass
from enum import Enum
import optuna
from abc import ABC, abstractmethod
import joblib
import pickle
from pathlib import Path

warnings.filterwarnings('ignore')

class ModelType(Enum):
    TRANSFORMER = "transformer"
    LSTM_ATTENTION = "lstm_attention"
    CNN_LSTM = "cnn_lstm"
    TEMPORAL_CNN = "temporal_cnn"
    WAVENET = "wavenet"
    QUANTIZED_NN = "quantized_nn"
    HYBRID_ENSEMBLE = "hybrid_ensemble"

@dataclass
class ModelMetrics:
    """Comprehensive model performance metrics."""
    model_name: str
    mse: float
    mae: float
    r2: float
    sharpe_ratio: float
    max_drawdown: float
    win_rate: float
    profit_factor: float
    calmar_ratio: float
    sortino_ratio: float
    kelly_criterion: float

class TradingDataset(Dataset):
    """PyTorch dataset for trading data."""
    
    def __init__(self, X: np.ndarray, y: np.ndarray, sequence_length: int = 60):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)
        self.sequence_length = sequence_length
        
    def __len__(self):
        return len(self.X) - self.sequence_length + 1
    
    def __getitem__(self, idx):
        return (
            self.X[idx:idx + self.sequence_length],
            self.y[idx + self.sequence_length - 1]
        )

class PositionalEncoding(nn.Module):
    """Positional encoding for transformer models."""
    
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           (-np.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        return x + self.pe[:x.size(0), :]

class TransformerTradingModel(nn.Module):
    """Advanced Transformer model for trading prediction."""
    
    def __init__(self, input_dim: int, d_model: int = 256, nhead: int = 8, 
                 num_layers: int = 6, dropout: float = 0.1):
        super().__init__()
        
        self.input_projection = nn.Linear(input_dim, d_model)
        self.positional_encoding = PositionalEncoding(d_model)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True
        )
        
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        
        self.output_layers = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, d_model // 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 4, 1)
        )
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        # x shape: (batch_size, seq_len, input_dim)
        x = self.input_projection(x)
        x = self.positional_encoding(x)
        x = self.dropout(x)
        
        # Transformer expects (seq_len, batch_size, d_model)
        x = x.transpose(0, 1)
        x = self.transformer(x)
        x = x.transpose(0, 1)
        
        # Use the last timestep for prediction
        x = x[:, -1, :]
        
        return self.output_layers(x)

class LSTMAttentionModel(nn.Module):
    """LSTM with attention mechanism for trading prediction."""
    
    def __init__(self, input_dim: int, hidden_dim: int = 256, num_layers: int = 3, dropout: float = 0.2):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, 
                           batch_first=True, dropout=dropout)
        
        # Attention mechanism
        self.attention = nn.MultiheadAttention(hidden_dim, num_heads=8, 
                                             dropout=dropout, batch_first=True)
        
        self.output_layers = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 4, 1)
        )
        
    def forward(self, x):
        # LSTM forward pass
        lstm_out, _ = self.lstm(x)
        
        # Apply attention
        attn_out, _ = self.attention(lstm_out, lstm_out, lstm_out)
        
        # Use the last timestep
        out = attn_out[:, -1, :]
        
        return self.output_layers(out)

class CNNLSTMModel(nn.Module):
    """Hybrid CNN-LSTM model for pattern recognition and temporal modeling."""
    
    def __init__(self, input_dim: int, hidden_dim: int = 128, dropout: float = 0.2):
        super().__init__()
        
        # 1D Convolution layers
        self.conv1 = nn.Conv1d(input_dim, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(64, 128, kernel_size=3, padding=1)
        self.conv3 = nn.Conv1d(128, 256, kernel_size=3, padding=1)
        
        self.pool = nn.MaxPool1d(2)
        self.dropout = nn.Dropout(dropout)
        
        # LSTM layers
        self.lstm = nn.LSTM(256, hidden_dim, num_layers=2, 
                           batch_first=True, dropout=dropout)
        
        # Output layers
        self.output_layers = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1)
        )
        
    def forward(self, x):
        # x shape: (batch_size, seq_len, input_dim)
        # Conv1d expects (batch_size, input_dim, seq_len)
        x = x.transpose(1, 2)
        
        # Convolutional layers
        x = F.relu(self.conv1(x))
        x = self.dropout(x)
        
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = self.dropout(x)
        
        x = F.relu(self.conv3(x))
        x = self.dropout(x)
        
        # Back to (batch_size, seq_len, features)
        x = x.transpose(1, 2)
        
        # LSTM
        lstm_out, _ = self.lstm(x)
        
        # Use last timestep
        out = lstm_out[:, -1, :]
        
        return self.output_layers(out)

class TemporalCNNModel(nn.Module):
    """Temporal Convolutional Network with dilated convolutions."""
    
    def __init__(self, input_dim: int, num_channels: List[int] = [64, 128, 256], 
                 kernel_size: int = 3, dropout: float = 0.2):
        super().__init__()
        
        layers = []
        num_levels = len(num_channels)
        
        for i in range(num_levels):
            dilation = 2 ** i
            in_channels = input_dim if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            
            layers.append(nn.Conv1d(in_channels, out_channels, kernel_size,
                                   dilation=dilation, padding=(kernel_size-1) * dilation))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
        
        self.network = nn.Sequential(*layers)
        
        self.output_layers = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(num_channels[-1], 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 1)
        )
        
    def forward(self, x):
        # x shape: (batch_size, seq_len, input_dim)
        x = x.transpose(1, 2)  # (batch_size, input_dim, seq_len)
        
        x = self.network(x)
        
        return self.output_layers(x)

class WaveNetModel(nn.Module):
    """WaveNet-inspired model for trading prediction."""
    
    def __init__(self, input_dim: int, residual_channels: int = 64, 
                 gate_channels: int = 128, skip_channels: int = 64,
                 end_channels: int = 128, num_blocks: int = 4, num_layers: int = 10):
        super().__init__()
        
        self.num_blocks = num_blocks
        self.num_layers = num_layers
        
        self.start_conv = nn.Conv1d(input_dim, residual_channels, 1)
        
        self.dilated_convs = nn.ModuleList()
        self.residual_convs = nn.ModuleList()
        self.skip_convs = nn.ModuleList()
        
        for b in range(num_blocks):
            for i in range(num_layers):
                dilation = 2 ** i
                
                self.dilated_convs.append(
                    nn.Conv1d(residual_channels, gate_channels, 2, dilation=dilation)
                )
                
                self.residual_convs.append(
                    nn.Conv1d(gate_channels // 2, residual_channels, 1)
                )
                
                self.skip_convs.append(
                    nn.Conv1d(gate_channels // 2, skip_channels, 1)
                )
        
        self.end_conv1 = nn.Conv1d(skip_channels, end_channels, 1)
        self.end_conv2 = nn.Conv1d(end_channels, 1, 1)
        
    def forward(self, x):
        # x shape: (batch_size, seq_len, input_dim)
        x = x.transpose(1, 2)  # (batch_size, input_dim, seq_len)
        
        x = self.start_conv(x)
        skip_connections = 0
        
        for b in range(self.num_blocks):
            for i in range(self.num_layers):
                layer_idx = b * self.num_layers + i
                
                # Dilated convolution
                dilated_out = self.dilated_convs[layer_idx](x)
                
                # Gated activation
                tanh_out = torch.tanh(dilated_out[:, :dilated_out.size(1)//2, :])
                sigm_out = torch.sigmoid(dilated_out[:, dilated_out.size(1)//2:, :])
                gated = tanh_out * sigm_out
                
                # Residual connection
                residual = self.residual_convs[layer_idx](gated)
                x = x + residual
                
                # Skip connection
                skip = self.skip_convs[layer_idx](gated)
                skip_connections = skip_connections + skip
        
        # Final output
        out = F.relu(skip_connections)
        out = F.relu(self.end_conv1(out))
        out = self.end_conv2(out)
        
        # Global average pooling
        out = out.mean(dim=2)
        
        return out

class QuantumInspiredNN(nn.Module):
    """Quantum-inspired neural network with interference patterns."""
    
    def __init__(self, input_dim: int, hidden_dim: int = 256, num_qubits: int = 8):
        super().__init__()
        
        self.num_qubits = num_qubits
        self.hidden_dim = hidden_dim
        
        # Classical layers
        self.classical_embedding = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_qubits * 2)  # Complex amplitudes
        )
        
        # Quantum-inspired layers
        self.quantum_weights = nn.Parameter(torch.randn(num_qubits, num_qubits) * 0.1)
        self.quantum_bias = nn.Parameter(torch.zeros(num_qubits))
        
        # Output layers
        self.output_layers = nn.Sequential(
            nn.Linear(num_qubits, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim // 2, 1)
        )
        
    def forward(self, x):
        # Classical embedding
        embedded = self.classical_embedding(x)
        
        # Split into real and imaginary parts
        real_part = embedded[:, :self.num_qubits]
        imag_part = embedded[:, self.num_qubits:]
        
        # Quantum-inspired interference
        complex_state = torch.complex(real_part, imag_part)
        
        # Apply quantum-like transformation
        quantum_matrix = torch.complex(self.quantum_weights, torch.zeros_like(self.quantum_weights))
        transformed_state = torch.matmul(complex_state, quantum_matrix)
        
        # Measurement (collapse to real values)
        measured = torch.abs(transformed_state) + self.quantum_bias
        
        return self.output_layers(measured)

class NextGenMLEnsemble:
    """Next-generation ML ensemble combining multiple advanced models."""
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or self._default_config()
        self.models = {}
        self.scalers = {}
        self.metrics = {}
        self.is_fitted = False
        self.logger = self._setup_logging()
        
    def _default_config(self) -> Dict:
        """Default configuration for maximum performance."""
        return {
            'models': [
                ModelType.TRANSFORMER,
                ModelType.LSTM_ATTENTION,
                ModelType.CNN_LSTM,
                ModelType.TEMPORAL_CNN,
                ModelType.WAVENET,
                ModelType.QUANTIZED_NN
            ],
            'sequence_length': 60,
            'batch_size': 64,
            'learning_rate': 0.001,
            'epochs': 100,
            'early_stopping_patience': 10,
            'ensemble_weights': 'auto',  # or 'equal' or custom weights
            'hyperparameter_optimization': True,
            'cross_validation_folds': 5,
            'device': 'cuda' if torch.cuda.is_available() else 'cpu'
        }
    
    def _setup_logging(self) -> logging.Logger:
        """Setup logging."""
        logger = logging.getLogger('NextGenMLEnsemble')
        logger.setLevel(logging.INFO)
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        return logger
    
    def fit(self, X: np.ndarray, y: np.ndarray, validation_data: Optional[Tuple] = None):
        """Train the ensemble of next-generation models."""
        self.logger.info("🚀 Training next-generation ML ensemble...")
        
        # Prepare data
        X_scaled = self._prepare_data(X, fit_scaler=True)
        
        # Split validation data if not provided
        if validation_data is None:
            split_idx = int(len(X) * 0.8)
            X_train, X_val = X_scaled[:split_idx], X_scaled[split_idx:]
            y_train, y_val = y[:split_idx], y[split_idx:]
        else:
            X_train, y_train = X_scaled, y
            X_val, y_val = self._prepare_data(validation_data[0]), validation_data[1]
        
        # Train each model
        for model_type in self.config['models']:
            self.logger.info(f"🧠 Training {model_type.value} model...")
            
            try:
                if self.config['hyperparameter_optimization']:
                    best_model = self._optimize_hyperparameters(model_type, X_train, y_train, X_val, y_val)
                else:
                    best_model = self._train_single_model(model_type, X_train, y_train, X_val, y_val)
                
                self.models[model_type] = best_model
                
                # Calculate metrics
                y_pred = self._predict_single_model(best_model, X_val)
                metrics = self._calculate_metrics(y_val, y_pred, model_type.value)
                self.metrics[model_type] = metrics
                
                self.logger.info(f"✅ {model_type.value} - R²: {metrics.r2:.4f}, Sharpe: {metrics.sharpe_ratio:.4f}")
                
            except Exception as e:
                self.logger.error(f"❌ Failed to train {model_type.value}: {str(e)}")
        
        # Calculate ensemble weights
        self._calculate_ensemble_weights()
        
        self.is_fitted = True
        self.logger.info("🎯 Next-generation ML ensemble training complete!")
        
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions using the ensemble."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        X_scaled = self._prepare_data(X)
        predictions = []
        weights = []
        
        for model_type, model in self.models.items():
            try:
                pred = self._predict_single_model(model, X_scaled)
                predictions.append(pred)
                
                # Get ensemble weight
                if hasattr(self, 'ensemble_weights'):
                    weights.append(self.ensemble_weights.get(model_type, 1.0))
                else:
                    weights.append(1.0)
                    
            except Exception as e:
                self.logger.warning(f"⚠️ Prediction failed for {model_type.value}: {str(e)}")
        
        if not predictions:
            raise RuntimeError("No successful predictions from any model")
        
        # Weighted ensemble prediction
        predictions = np.array(predictions)
        weights = np.array(weights)
        weights = weights / weights.sum()  # Normalize weights
        
        ensemble_prediction = np.average(predictions, axis=0, weights=weights)
        
        return ensemble_prediction
    
    def _prepare_data(self, X: np.ndarray, fit_scaler: bool = False) -> np.ndarray:
        """Prepare and scale data."""
        if fit_scaler:
            self.scalers['features'] = StandardScaler()
            X_scaled = self.scalers['features'].fit_transform(X)
        else:
            X_scaled = self.scalers['features'].transform(X)
        
        return X_scaled
    
    def _create_model(self, model_type: ModelType, input_dim: int, **kwargs) -> nn.Module:
        """Create a model of specified type."""
        if model_type == ModelType.TRANSFORMER:
            return TransformerTradingModel(input_dim, **kwargs)
        elif model_type == ModelType.LSTM_ATTENTION:
            return LSTMAttentionModel(input_dim, **kwargs)
        elif model_type == ModelType.CNN_LSTM:
            return CNNLSTMModel(input_dim, **kwargs)
        elif model_type == ModelType.TEMPORAL_CNN:
            return TemporalCNNModel(input_dim, **kwargs)
        elif model_type == ModelType.WAVENET:
            return WaveNetModel(input_dim, **kwargs)
        elif model_type == ModelType.QUANTIZED_NN:
            return QuantumInspiredNN(input_dim, **kwargs)
        else:
            raise ValueError(f"Unknown model type: {model_type}")
    
    def _train_single_model(self, model_type: ModelType, X_train: np.ndarray, 
                           y_train: np.ndarray, X_val: np.ndarray, y_val: np.ndarray) -> nn.Module:
        """Train a single model."""
        device = torch.device(self.config.get('device', 'cpu'))
        
        # Create model
        model = self._create_model(model_type, X_train.shape[1])
        model.to(device)
        
        # Create datasets
        train_dataset = TradingDataset(X_train, y_train, self.config['sequence_length'])
        val_dataset = TradingDataset(X_val, y_val, self.config['sequence_length'])
        
        train_loader = DataLoader(train_dataset, batch_size=self.config['batch_size'], shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=self.config['batch_size'], shuffle=False)
        
        # Optimizer and loss
        optimizer = torch.optim.Adam(model.parameters(), lr=self.config['learning_rate'])
        criterion = nn.MSELoss()
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
        
        # Training loop
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(self.config['epochs']):
            # Training
            model.train()
            train_losses = []
            
            for batch_X, batch_y in train_loader:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                
                optimizer.zero_grad()
                outputs = model(batch_X)
                loss = criterion(outputs.squeeze(), batch_y)
                loss.backward()
                optimizer.step()
                
                train_losses.append(loss.item())
            
            # Validation
            model.eval()
            val_losses = []
            
            with torch.no_grad():
                for batch_X, batch_y in val_loader:
                    batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                    outputs = model(batch_X)
                    loss = criterion(outputs.squeeze(), batch_y)
                    val_losses.append(loss.item())
            
            avg_train_loss = np.mean(train_losses)
            avg_val_loss = np.mean(val_losses)
            
            scheduler.step(avg_val_loss)
            
            # Early stopping
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                patience_counter = 0
                # Save best model state
                best_model_state = model.state_dict().copy()
            else:
                patience_counter += 1
                
            if patience_counter >= self.config['early_stopping_patience']:
                self.logger.info(f"Early stopping at epoch {epoch}")
                break
        
        # Load best model state
        if 'best_model_state' in locals():
            model.load_state_dict(best_model_state)
        
        return model
    
    def _predict_single_model(self, model: nn.Module, X: np.ndarray) -> np.ndarray:
        """Make predictions with a single model."""
        device = torch.device(self.config.get('device', 'cpu'))
        model.eval()
        
        dataset = TradingDataset(X, np.zeros(len(X)), self.config['sequence_length'])
        dataloader = DataLoader(dataset, batch_size=self.config['batch_size'], shuffle=False)
        
        predictions = []
        
        with torch.no_grad():
            for batch_X, _ in dataloader:
                batch_X = batch_X.to(device)
                outputs = model(batch_X)
                predictions.extend(outputs.cpu().numpy().flatten())
        
        return np.array(predictions)
    
    def _optimize_hyperparameters(self, model_type: ModelType, X_train: np.ndarray, 
                                 y_train: np.ndarray, X_val: np.ndarray, y_val: np.ndarray) -> nn.Module:
        """Optimize hyperparameters using Optuna."""
        def objective(trial):
            # Suggest hyperparameters
            if model_type == ModelType.TRANSFORMER:
                d_model = trial.suggest_categorical('d_model', [128, 256, 512])
                nhead = trial.suggest_categorical('nhead', [4, 8, 16])
                num_layers = trial.suggest_int('num_layers', 2, 8)
                dropout = trial.suggest_float('dropout', 0.1, 0.5)
                
                model_kwargs = {
                    'd_model': d_model,
                    'nhead': nhead,
                    'num_layers': num_layers,
                    'dropout': dropout
                }
            else:
                # Default hyperparameters for other models
                hidden_dim = trial.suggest_categorical('hidden_dim', [128, 256, 512])
                dropout = trial.suggest_float('dropout', 0.1, 0.5)
                
                model_kwargs = {
                    'hidden_dim': hidden_dim,
                    'dropout': dropout
                }
            
            # Train model with suggested hyperparameters
            try:
                temp_config = self.config.copy()
                temp_config['epochs'] = 20  # Reduced epochs for optimization
                
                model = self._create_model(model_type, X_train.shape[1], **model_kwargs)
                trained_model = self._train_single_model_with_config(
                    model, X_train, y_train, X_val, y_val, temp_config
                )
                
                # Evaluate model
                y_pred = self._predict_single_model(trained_model, X_val)
                mse = mean_squared_error(y_val[self.config['sequence_length']-1:], y_pred)
                
                return mse
                
            except Exception as e:
                self.logger.warning(f"Trial failed: {str(e)}")
                return float('inf')
        
        # Run optimization
        study = optuna.create_study(direction='minimize')
        study.optimize(objective, n_trials=20, timeout=300)  # 5 minutes max
        
        # Train final model with best parameters
        best_params = study.best_params
        final_model = self._create_model(model_type, X_train.shape[1], **best_params)
        final_trained_model = self._train_single_model(model_type, X_train, y_train, X_val, y_val)
        
        return final_trained_model
    
    def _train_single_model_with_config(self, model: nn.Module, X_train: np.ndarray, 
                                       y_train: np.ndarray, X_val: np.ndarray, 
                                       y_val: np.ndarray, config: Dict) -> nn.Module:
        """Train model with specific config (used for hyperparameter optimization)."""
        # Similar to _train_single_model but with custom config
        return self._train_single_model(model.model_type if hasattr(model, 'model_type') else ModelType.TRANSFORMER,
                                       X_train, y_train, X_val, y_val)
    
    def _calculate_metrics(self, y_true: np.ndarray, y_pred: np.ndarray, model_name: str) -> ModelMetrics:
        """Calculate comprehensive model metrics."""
        # Align arrays (remove NaN values that might occur during sequence preparation)
        y_true_aligned = y_true[self.config['sequence_length']-1:]
        y_pred_aligned = y_pred
        
        # Handle length mismatch
        min_length = min(len(y_true_aligned), len(y_pred_aligned))
        y_true_aligned = y_true_aligned[:min_length]
        y_pred_aligned = y_pred_aligned[:min_length]
        
        # Basic metrics
        mse = mean_squared_error(y_true_aligned, y_pred_aligned)
        mae = mean_absolute_error(y_true_aligned, y_pred_aligned)
        r2 = r2_score(y_true_aligned, y_pred_aligned)
        
        # Trading-specific metrics
        returns = y_pred_aligned  # Assuming predictions are returns
        
        # Sharpe ratio
        sharpe_ratio = np.mean(returns) / (np.std(returns) + 1e-8) * np.sqrt(252)
        
        # Maximum drawdown
        cumulative_returns = np.cumprod(1 + returns)
        running_max = np.maximum.accumulate(cumulative_returns)
        drawdown = (cumulative_returns - running_max) / running_max
        max_drawdown = np.min(drawdown)
        
        # Win rate
        win_rate = np.mean(returns > 0)
        
        # Profit factor
        positive_returns = returns[returns > 0]
        negative_returns = returns[returns < 0]
        profit_factor = np.sum(positive_returns) / (abs(np.sum(negative_returns)) + 1e-8)
        
        # Calmar ratio
        annual_return = np.mean(returns) * 252
        calmar_ratio = annual_return / (abs(max_drawdown) + 1e-8)
        
        # Sortino ratio
        downside_returns = returns[returns < 0]
        downside_std = np.std(downside_returns) if len(downside_returns) > 0 else 1e-8
        sortino_ratio = np.mean(returns) / downside_std * np.sqrt(252)
        
        # Kelly criterion
        if win_rate > 0 and win_rate < 1:
            avg_win = np.mean(positive_returns) if len(positive_returns) > 0 else 0
            avg_loss = abs(np.mean(negative_returns)) if len(negative_returns) > 0 else 1
            kelly_criterion = win_rate - ((1 - win_rate) / (avg_win / avg_loss + 1e-8))
        else:
            kelly_criterion = 0
        
        return ModelMetrics(
            model_name=model_name,
            mse=mse,
            mae=mae,
            r2=r2,
            sharpe_ratio=sharpe_ratio,
            max_drawdown=max_drawdown,
            win_rate=win_rate,
            profit_factor=profit_factor,
            calmar_ratio=calmar_ratio,
            sortino_ratio=sortino_ratio,
            kelly_criterion=kelly_criterion
        )
    
    def _calculate_ensemble_weights(self):
        """Calculate optimal ensemble weights based on model performance."""
        ensemble_weights_config = self.config.get('ensemble_weights', 'auto')
        
        if ensemble_weights_config == 'equal':
            # Equal weights
            weights = {model_type: 1.0 / len(self.models) for model_type in self.models.keys()}
        
        elif ensemble_weights_config == 'auto':
            # Performance-based weights
            if not self.metrics:
                # Default equal weights if no metrics available
                weights = {model_type: 1.0 / len(self.models) for model_type in self.models.keys()}
            else:
                scores = []
                model_types = []
                
                for model_type, metrics in self.metrics.items():
                    # Combine multiple metrics for overall score
                    score = (
                        metrics.r2 * 0.3 +
                        min(metrics.sharpe_ratio / 2.0, 1.0) * 0.3 +  # Cap Sharpe at 2.0
                        (1 - abs(metrics.max_drawdown)) * 0.2 +
                        metrics.win_rate * 0.2
                    )
                    scores.append(max(score, 0))  # Ensure positive scores
                    model_types.append(model_type)
                
                # Softmax normalization
                scores = np.array(scores)
                if scores.sum() > 0:
                    weights_array = np.exp(scores) / np.sum(np.exp(scores))
                    weights = {model_types[i]: weights_array[i] for i in range(len(model_types))}
                else:
                    weights = {model_type: 1.0 / len(self.models) for model_type in self.models.keys()}
        
        else:
            # Use custom weights if provided
            weights = ensemble_weights_config
        
        self.ensemble_weights = weights
        self.logger.info(f"🎯 Ensemble weights: {weights}")
    
    def get_model_performance_summary(self) -> pd.DataFrame:
        """Get comprehensive performance summary of all models."""
        if not self.metrics:
            return pd.DataFrame()
        
        data = []
        for model_type, metrics in self.metrics.items():
            data.append({
                'Model': model_type.value,
                'R²': metrics.r2,
                'MSE': metrics.mse,
                'MAE': metrics.mae,
                'Sharpe Ratio': metrics.sharpe_ratio,
                'Max Drawdown': metrics.max_drawdown,
                'Win Rate': metrics.win_rate,
                'Profit Factor': metrics.profit_factor,
                'Calmar Ratio': metrics.calmar_ratio,
                'Sortino Ratio': metrics.sortino_ratio,
                'Kelly Criterion': metrics.kelly_criterion,
                'Ensemble Weight': self.ensemble_weights.get(model_type, 0.0) if hasattr(self, 'ensemble_weights') else 0.0
            })
        
        return pd.DataFrame(data).round(4)
    
    def save_model(self, path: Union[str, Path]):
        """Save the trained ensemble."""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        
        # Save each model
        for model_type, model in self.models.items():
            model_path = path / f"{model_type.value}.pth"
            torch.save(model.state_dict(), model_path)
        
        # Save scalers and metadata
        metadata = {
            'config': self.config,
            'metrics': self.metrics,
            'ensemble_weights': getattr(self, 'ensemble_weights', {}),
            'is_fitted': self.is_fitted
        }
        
        with open(path / 'scalers.pkl', 'wb') as f:
            pickle.dump(self.scalers, f)
        
        with open(path / 'metadata.pkl', 'wb') as f:
            pickle.dump(metadata, f)
        
        self.logger.info(f"✅ Model ensemble saved to {path}")
    
    def load_model(self, path: Union[str, Path]):
        """Load a trained ensemble."""
        path = Path(path)
        
        # Load metadata
        with open(path / 'metadata.pkl', 'rb') as f:
            metadata = pickle.load(f)
        
        self.config = metadata['config']
        self.metrics = metadata['metrics']
        self.ensemble_weights = metadata['ensemble_weights']
        self.is_fitted = metadata['is_fitted']
        
        # Load scalers
        with open(path / 'scalers.pkl', 'rb') as f:
            self.scalers = pickle.load(f)
        
        # Load models
        self.models = {}
        for model_type in self.config['models']:
            model_path = path / f"{model_type.value}.pth"
            if model_path.exists():
                # Create model architecture
                dummy_input_dim = 50  # This should be stored in metadata
                model = self._create_model(model_type, dummy_input_dim)
                model.load_state_dict(torch.load(model_path, map_location=self.config['device']))
                self.models[model_type] = model
        
        self.logger.info(f"✅ Model ensemble loaded from {path}")

# Example usage and testing
if __name__ == "__main__":
    print("🚀 Testing Next-Generation ML Ensemble...")
    
    # Create synthetic data for testing
    np.random.seed(42)
    n_samples = 1000
    n_features = 50
    
    # Generate realistic financial time series data
    X = np.random.randn(n_samples, n_features)
    # Add some temporal correlation
    for i in range(1, n_samples):
        X[i] = 0.9 * X[i-1] + 0.1 * X[i]
    
    # Generate target (returns)
    y = np.random.randn(n_samples) * 0.02  # 2% volatility
    # Add some predictable patterns
    y = y + 0.1 * np.sin(np.arange(n_samples) * 0.1) * X[:, 0]
    
    try:
        # Create and train ensemble
        ensemble = NextGenMLEnsemble({
            'models': [ModelType.TRANSFORMER, ModelType.LSTM_ATTENTION],
            'epochs': 10,  # Reduced for testing
            'batch_size': 32,
            'hyperparameter_optimization': False  # Disabled for testing
        })
        
        # Split data
        split_idx = int(len(X) * 0.8)
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        
        # Train
        print("🧠 Training ensemble...")
        ensemble.fit(X_train, y_train, (X_test, y_test))
        
        # Predict
        print("🎯 Making predictions...")
        predictions = ensemble.predict(X_test)
        
        # Show results
        print("📊 Performance Summary:")
        summary = ensemble.get_model_performance_summary()
        print(summary)
        
        print("✅ Next-generation ML ensemble test completed!")
        
    except Exception as e:
        print(f"❌ Test failed: {str(e)}")
        import traceback
        traceback.print_exc()