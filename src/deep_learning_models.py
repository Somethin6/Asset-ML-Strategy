#!/usr/bin/env python3
"""
Advanced Deep Learning Models for Time Series Trading
Implements LSTM, Transformer, and CNN models for ultimate trading performance.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import (
    LSTM, Dense, Dropout, Conv1D, MaxPooling1D, 
    Flatten, MultiHeadAttention, LayerNormalization,
    Input, GlobalAveragePooling1D, Add
)
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import pandas as pd
import logging
from typing import Tuple, List, Dict, Optional

logger = logging.getLogger(__name__)

class TimeSeriesDataset(Dataset):
    """PyTorch Dataset for time series data."""
    
    def __init__(self, X, y, sequence_length=60):
        self.X = X
        self.y = y
        self.sequence_length = sequence_length
    
    def __len__(self):
        return len(self.X) - self.sequence_length + 1
    
    def __getitem__(self, idx):
        return (
            torch.FloatTensor(self.X[idx:idx + self.sequence_length]),
            torch.FloatTensor([self.y[idx + self.sequence_length - 1]])
        )

class QuantumInspiredLSTM(nn.Module):
    """
    Quantum-inspired LSTM with entanglement-like connections.
    """
    
    def __init__(self, input_size, hidden_size=128, num_layers=3, dropout=0.2):
        super(QuantumInspiredLSTM, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # Quantum-inspired components
        self.quantum_gate = nn.Linear(input_size, input_size)
        self.entanglement_layer = nn.Linear(hidden_size, hidden_size)
        
        # LSTM layers with residual connections
        self.lstm_layers = nn.ModuleList([
            nn.LSTM(input_size if i == 0 else hidden_size, 
                   hidden_size, batch_first=True, dropout=dropout)
            for i in range(num_layers)
        ])
        
        # Attention mechanism
        self.attention = nn.MultiheadAttention(hidden_size, num_heads=8)
        
        # Output layers
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(hidden_size, hidden_size // 2)
        self.fc2 = nn.Linear(hidden_size // 2, hidden_size // 4)
        self.fc3 = nn.Linear(hidden_size // 4, 1)
        
        # Quantum superposition simulation
        self.superposition_weights = nn.Parameter(torch.randn(hidden_size, hidden_size))
        
    def forward(self, x):
        batch_size, seq_len, features = x.size()
        
        # Quantum-inspired preprocessing
        x = torch.tanh(self.quantum_gate(x))  # Quantum gate simulation
        
        # Multi-layer LSTM with residual connections
        hidden_states = []
        h_n, c_n = None, None
        
        for i, lstm in enumerate(self.lstm_layers):
            if i == 0:
                out, (h_n, c_n) = lstm(x)
            else:
                # Residual connection
                out, (h_n, c_n) = lstm(out)
                if out.size() == hidden_states[-1].size():
                    out = out + hidden_states[-1]  # Residual connection
            
            hidden_states.append(out)
        
        # Quantum entanglement simulation
        entangled_output = torch.matmul(out, self.superposition_weights)
        entangled_output = self.entanglement_layer(entangled_output)
        
        # Self-attention mechanism
        out = out.transpose(0, 1)  # (seq_len, batch, features)
        attended_out, _ = self.attention(out, out, out)
        attended_out = attended_out.transpose(0, 1)  # Back to (batch, seq, features)
        
        # Combine quantum and attention outputs
        final_output = attended_out + entangled_output
        
        # Take the last time step
        final_output = final_output[:, -1, :]
        
        # Dense layers with advanced activation
        out = self.dropout(torch.relu(self.fc1(final_output)))
        out = self.dropout(torch.relu(self.fc2(out)))
        out = torch.sigmoid(self.fc3(out))  # Probability output
        
        return out

class TransformerTimeSeriesModel:
    """
    Advanced Transformer model for time series prediction.
    """
    
    def __init__(self, sequence_length=60, n_features=122, d_model=256, n_heads=8, n_layers=6):
        self.sequence_length = sequence_length
        self.n_features = n_features
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.model = None
        self.scaler = MinMaxScaler()
        
    def build_model(self):
        """Build the Transformer model."""
        # Input layer
        inputs = Input(shape=(self.sequence_length, self.n_features))
        
        # Positional encoding
        x = Dense(self.d_model)(inputs)
        
        # Multi-layer Transformer
        for _ in range(self.n_layers):
            # Multi-head attention
            attn_output = MultiHeadAttention(
                num_heads=self.n_heads, 
                key_dim=self.d_model // self.n_heads,
                dropout=0.1
            )(x, x)
            
            # Residual connection and layer norm
            x = Add()([x, attn_output])
            x = LayerNormalization()(x)
            
            # Feed forward network
            ff_output = Dense(self.d_model * 4, activation='relu')(x)
            ff_output = Dropout(0.1)(ff_output)
            ff_output = Dense(self.d_model)(ff_output)
            
            # Another residual connection
            x = Add()([x, ff_output])
            x = LayerNormalization()(x)
        
        # Global pooling and output
        x = GlobalAveragePooling1D()(x)
        x = Dense(128, activation='relu')(x)
        x = Dropout(0.2)(x)
        x = Dense(64, activation='relu')(x)
        outputs = Dense(1, activation='sigmoid')(x)
        
        self.model = Model(inputs, outputs)
        self.model.compile(
            optimizer=Adam(learning_rate=1e-4),
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        
        return self.model
    
    def prepare_data(self, X, y):
        """Prepare data for Transformer training."""
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Create sequences
        sequences = []
        labels = []
        
        for i in range(self.sequence_length, len(X_scaled)):
            sequences.append(X_scaled[i-self.sequence_length:i])
            labels.append(y.iloc[i])
        
        return np.array(sequences), np.array(labels)

class CNNTimeSeriesModel:
    """
    Convolutional Neural Network for time series pattern recognition.
    """
    
    def __init__(self, sequence_length=60, n_features=122):
        self.sequence_length = sequence_length
        self.n_features = n_features
        self.model = None
        self.scaler = MinMaxScaler()
        
    def build_model(self):
        """Build multi-scale CNN model."""
        inputs = Input(shape=(self.sequence_length, self.n_features))
        
        # Multi-scale convolution branches
        branches = []
        
        # Short-term patterns (small kernels)
        conv1 = Conv1D(64, kernel_size=3, activation='relu', padding='same')(inputs)
        conv1 = Conv1D(64, kernel_size=3, activation='relu', padding='same')(conv1)
        pool1 = MaxPooling1D(pool_size=2)(conv1)
        branches.append(pool1)
        
        # Medium-term patterns
        conv2 = Conv1D(64, kernel_size=7, activation='relu', padding='same')(inputs)
        conv2 = Conv1D(64, kernel_size=7, activation='relu', padding='same')(conv2)
        pool2 = MaxPooling1D(pool_size=2)(conv2)
        branches.append(pool2)
        
        # Long-term patterns
        conv3 = Conv1D(64, kernel_size=15, activation='relu', padding='same')(inputs)
        conv3 = Conv1D(64, kernel_size=15, activation='relu', padding='same')(conv3)
        pool3 = MaxPooling1D(pool_size=2)(conv3)
        branches.append(pool3)
        
        # Combine branches
        if len(branches) > 1:
            combined = tf.keras.layers.Concatenate(axis=-1)(branches)
        else:
            combined = branches[0]
        
        # Additional CNN layers
        x = Conv1D(128, kernel_size=5, activation='relu')(combined)
        x = Dropout(0.3)(x)
        x = Conv1D(256, kernel_size=3, activation='relu')(x)
        x = Dropout(0.3)(x)
        
        # Global pooling and dense layers
        x = GlobalAveragePooling1D()(x)
        x = Dense(256, activation='relu')(x)
        x = Dropout(0.4)(x)
        x = Dense(128, activation='relu')(x)
        x = Dropout(0.3)(x)
        outputs = Dense(1, activation='sigmoid')(x)
        
        self.model = Model(inputs, outputs)
        self.model.compile(
            optimizer=Adam(learning_rate=1e-3),
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        
        return self.model

class AdvancedDeepLearningEnsemble:
    """
    Ensemble of deep learning models for maximum performance.
    """
    
    def __init__(self, sequence_length=60, n_features=122):
        self.sequence_length = sequence_length
        self.n_features = n_features
        
        # Initialize models
        self.quantum_lstm = None
        self.transformer = TransformerTimeSeriesModel(sequence_length, n_features)
        self.cnn = CNNTimeSeriesModel(sequence_length, n_features)
        
        self.models = {}
        self.trained = False
        
    def build_models(self):
        """Build all deep learning models."""
        logger.info("Building advanced deep learning models...")
        
        # Build Quantum-inspired LSTM (PyTorch)
        self.quantum_lstm = QuantumInspiredLSTM(
            input_size=self.n_features,
            hidden_size=128,
            num_layers=3,
            dropout=0.2
        )
        
        # Build Transformer (Keras)
        self.transformer.build_model()
        
        # Build CNN (Keras)  
        self.cnn.build_model()
        
        self.models = {
            'quantum_lstm': self.quantum_lstm,
            'transformer': self.transformer,
            'cnn': self.cnn
        }
        
        logger.info("Advanced deep learning models built successfully!")
    
    def train_models(self, X: np.ndarray, y: np.ndarray, epochs=50, batch_size=32):
        """Train all deep learning models."""
        if not self.models:
            self.build_models()
        
        results = {}
        
        # Train Transformer
        logger.info("Training Transformer model...")
        try:
            X_trans, y_trans = self.transformer.prepare_data(
                pd.DataFrame(X), pd.Series(y)
            )
            
            history = self.transformer.model.fit(
                X_trans, y_trans,
                epochs=min(epochs, 20),  # Reduce epochs for efficiency
                batch_size=batch_size,
                validation_split=0.2,
                verbose=0
            )
            results['transformer'] = history.history['val_accuracy'][-1]
            logger.info(f"Transformer training completed. Accuracy: {results['transformer']:.4f}")
        except Exception as e:
            logger.error(f"Error training Transformer: {e}")
            results['transformer'] = 0.5
        
        # Train CNN
        logger.info("Training CNN model...")
        try:
            X_cnn, y_cnn = self.cnn.prepare_data(
                pd.DataFrame(X), pd.Series(y)
            )
            
            history = self.cnn.model.fit(
                X_cnn, y_cnn,
                epochs=min(epochs, 15),  # Reduce epochs for efficiency
                batch_size=batch_size,
                validation_split=0.2,
                verbose=0
            )
            results['cnn'] = history.history['val_accuracy'][-1]
            logger.info(f"CNN training completed. Accuracy: {results['cnn']:.4f}")
        except Exception as e:
            logger.error(f"Error training CNN: {e}")
            results['cnn'] = 0.5
        
        # Train Quantum LSTM (simplified for efficiency)
        logger.info("Training Quantum-inspired LSTM...")
        try:
            # Use a smaller subset for PyTorch model to save time
            subset_size = min(1000, len(X))
            X_subset = X[-subset_size:]
            y_subset = y[-subset_size:]
            
            dataset = TimeSeriesDataset(X_subset, y_subset, self.sequence_length)
            
            if len(dataset) > 0:
                dataloader = DataLoader(dataset, batch_size=16, shuffle=True)
                
                optimizer = optim.Adam(self.quantum_lstm.parameters(), lr=1e-3)
                criterion = nn.BCELoss()
                
                self.quantum_lstm.train()
                for epoch in range(min(10, epochs)):  # Fewer epochs
                    epoch_loss = 0
                    for batch_X, batch_y in dataloader:
                        optimizer.zero_grad()
                        outputs = self.quantum_lstm(batch_X)
                        loss = criterion(outputs, batch_y)
                        loss.backward()
                        optimizer.step()
                        epoch_loss += loss.item()
                
                # Simple validation
                self.quantum_lstm.eval()
                with torch.no_grad():
                    total_correct = 0
                    total_samples = 0
                    for batch_X, batch_y in dataloader:
                        outputs = self.quantum_lstm(batch_X)
                        predicted = (outputs > 0.5).float()
                        total_correct += (predicted == batch_y).sum().item()
                        total_samples += batch_y.size(0)
                    
                    results['quantum_lstm'] = total_correct / total_samples if total_samples > 0 else 0.5
            else:
                results['quantum_lstm'] = 0.5
                
            logger.info(f"Quantum LSTM training completed. Accuracy: {results['quantum_lstm']:.4f}")
        except Exception as e:
            logger.error(f"Error training Quantum LSTM: {e}")
            results['quantum_lstm'] = 0.5
        
        self.trained = True
        return results
    
    def predict_ensemble(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Make ensemble predictions from all deep learning models."""
        if not self.trained:
            raise ValueError("Models not trained yet. Call train_models first.")
        
        predictions = []
        probabilities = []
        
        # Transformer predictions
        try:
            X_trans = self.transformer.scaler.transform(X)
            sequences = []
            for i in range(self.sequence_length, len(X_trans)):
                sequences.append(X_trans[i-self.sequence_length:i])
            
            if sequences:
                X_seq = np.array(sequences)
                trans_probs = self.transformer.model.predict(X_seq, verbose=0)
                trans_pred = (trans_probs > 0.5).astype(int).flatten()
                
                # Pad to match original length
                trans_pred = np.concatenate([
                    np.zeros(self.sequence_length), trans_pred
                ])[:len(X)]
                trans_probs = np.concatenate([
                    np.full(self.sequence_length, 0.5), trans_probs.flatten()
                ])[:len(X)]
                
                predictions.append(trans_pred)
                probabilities.append(trans_probs)
        except Exception as e:
            logger.error(f"Error in Transformer prediction: {e}")
            predictions.append(np.zeros(len(X)))
            probabilities.append(np.full(len(X), 0.5))
        
        # CNN predictions (similar logic)
        try:
            X_cnn = self.cnn.scaler.transform(X) if hasattr(self.cnn, 'scaler') and self.cnn.scaler else X
            sequences = []
            for i in range(self.sequence_length, len(X_cnn)):
                sequences.append(X_cnn[i-self.sequence_length:i])
            
            if sequences:
                X_seq = np.array(sequences)
                cnn_probs = self.cnn.model.predict(X_seq, verbose=0)
                cnn_pred = (cnn_probs > 0.5).astype(int).flatten()
                
                # Pad to match original length
                cnn_pred = np.concatenate([
                    np.zeros(self.sequence_length), cnn_pred
                ])[:len(X)]
                cnn_probs = np.concatenate([
                    np.full(self.sequence_length, 0.5), cnn_probs.flatten()
                ])[:len(X)]
                
                predictions.append(cnn_pred)
                probabilities.append(cnn_probs)
        except Exception as e:
            logger.error(f"Error in CNN prediction: {e}")
            predictions.append(np.zeros(len(X)))
            probabilities.append(np.full(len(X), 0.5))
        
        # Quantum LSTM predictions
        try:
            # Use last part of data for prediction
            pred_length = min(len(X), 100)  # Limit for efficiency
            X_subset = X[-pred_length:]
            
            self.quantum_lstm.eval()
            with torch.no_grad():
                sequences = []
                for i in range(self.sequence_length, len(X_subset)):
                    sequences.append(X_subset[i-self.sequence_length:i])
                
                if sequences:
                    X_tensor = torch.FloatTensor(np.array(sequences))
                    lstm_probs = self.quantum_lstm(X_tensor).cpu().numpy().flatten()
                    lstm_pred = (lstm_probs > 0.5).astype(int)
                    
                    # Pad to match original length  
                    lstm_pred = np.concatenate([
                        np.zeros(len(X) - len(lstm_pred)), lstm_pred
                    ])
                    lstm_probs = np.concatenate([
                        np.full(len(X) - len(lstm_probs), 0.5), lstm_probs
                    ])
                    
                    predictions.append(lstm_pred)
                    probabilities.append(lstm_probs)
                else:
                    predictions.append(np.zeros(len(X)))
                    probabilities.append(np.full(len(X), 0.5))
        except Exception as e:
            logger.error(f"Error in Quantum LSTM prediction: {e}")
            predictions.append(np.zeros(len(X)))
            probabilities.append(np.full(len(X), 0.5))
        
        if not predictions:
            # Fallback if all models failed
            return np.zeros(len(X)), np.full(len(X), 0.5)
        
        # Ensemble voting
        predictions_array = np.array(predictions)
        probabilities_array = np.array(probabilities)
        
        # Weighted ensemble (equal weights for now)
        ensemble_predictions = np.round(np.mean(predictions_array, axis=0)).astype(int)
        ensemble_probabilities = np.mean(probabilities_array, axis=0)
        
        return ensemble_predictions, ensemble_probabilities
    
    def prepare_data(self, X: np.ndarray, y: np.ndarray):
        """Prepare data for all models consistently.""" 
        # For efficiency, use a subset for deep learning
        subset_size = min(5000, len(X))  # Use last 5000 samples
        X_subset = X[-subset_size:]
        y_subset = y[-subset_size:]
        
        return X_subset, y_subset

if __name__ == '__main__':
    # Test the advanced deep learning models
    logger.info("Testing Advanced Deep Learning Ensemble...")
    
    # Generate sample data
    X_sample = np.random.randn(1000, 122)
    y_sample = np.random.randint(0, 2, 1000)
    
    # Initialize and train ensemble
    ensemble = AdvancedDeepLearningEnsemble()
    
    # Train models
    results = ensemble.train_models(X_sample, y_sample, epochs=5)
    print("Training Results:", results)
    
    # Make predictions
    predictions, probabilities = ensemble.predict_ensemble(X_sample)
    print(f"Predictions shape: {predictions.shape}")
    print(f"Probabilities shape: {probabilities.shape}")
    
    logger.info("Advanced Deep Learning Ensemble test completed!")