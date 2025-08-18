import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# --- Custom Dataset for Time Series ---

class TimeSeriesDataset(Dataset):
    def __init__(self, X, y, sequence_length=60):
        self.X = torch.tensor(X.values, dtype=torch.float32)
        self.y = torch.tensor(y.values, dtype=torch.long)
        self.sequence_length = sequence_length

    def __len__(self):
        return len(self.X) - self.sequence_length

    def __getitem__(self, idx):
        return (self.X[idx:idx+self.sequence_length], self.y[idx+self.sequence_length])

# --- Positional Encoding ---

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

# --- Transformer Model ---

class TransformerModel(nn.Module):
    def __init__(self, input_dim, d_model, nhead, num_encoder_layers, num_classes, dropout=0.5):
        super(TransformerModel, self).__init__()
        self.model_type = 'Transformer'
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        encoder_layers = nn.TransformerEncoderLayer(d_model, nhead, d_model*4, dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_encoder_layers)
        self.encoder = nn.Linear(input_dim, d_model)
        self.d_model = d_model
        self.decoder = nn.Linear(d_model, num_classes)
        self.init_weights()

    def init_weights(self):
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, src):
        src = self.encoder(src) * np.sqrt(self.d_model)
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src)
        output = output.mean(dim=0) # Average over sequence length
        output = self.decoder(output)
        return output

# --- Training and Prediction ---

def train_model(X: pd.DataFrame, y: pd.Series, sequence_length=60, epochs=10, lr=1e-4):
    """
    Trains the Transformer model.
    """
    # Create dataset and dataloader
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False) # Time series data should not be shuffled
    train_dataset = TimeSeriesDataset(X_train, y_train, sequence_length)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

    # Model setup
    input_dim = X.shape[1]
    model = TransformerModel(input_dim=input_dim, d_model=128, nhead=8, num_encoder_layers=3, num_classes=2)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # Training loop
    model.train()
    for epoch in range(epochs):
        for batch_X, batch_y in train_loader:
            batch_X = batch_X.permute(1, 0, 2) # (seq_len, batch, features)
            optimizer.zero_grad()
            output = model(batch_X)
            loss = criterion(output, batch_y)
            loss.backward()
            optimizer.step()
        print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item()}")

    return model.cpu()

def predict(model, X: pd.DataFrame, sequence_length=60) -> pd.Series:
    """
    Makes predictions using the trained Transformer model.
    """
    model.eval()
    all_preds = []

    # Create a dataset for prediction (without labels)
    X_tensor = torch.tensor(X.values, dtype=torch.float32)

    with torch.no_grad():
        for i in range(len(X) - sequence_length):
            seq = X_tensor[i:i+sequence_length].unsqueeze(1) # Add batch dimension
            seq = seq.permute(1, 0, 2) # (seq_len, batch, features) -> This is wrong, should be (seq_len, batch, dim) -> (60, 1, num_features)
            seq = seq.permute(1,0,2) # so from (batch, seq, feature) to (seq, batch, feature)

            # The above permutation is a bit confusing. Let me fix it.
            # The input to the transformer should be (sequence_length, batch_size, input_dim)
            # The dataset gives (batch_size, sequence_length, input_dim)
            # So we need to permute.

            # Let's create the sequence and predict one by one for simplicity in prediction
            input_seq = X_tensor[i:i+sequence_length].unsqueeze(0).permute(1,0,2) # (seq_len, batch=1, features)
            output = model(input_seq)
            pred = torch.argmax(output, dim=1)
            all_preds.append(pred.item())

    # The predictions are for the data points starting from `sequence_length`
    # So the length of predictions is len(X) - sequence_length
    # We need to return a pandas Series with the correct index
    pred_series = pd.Series(all_preds, index=X.index[sequence_length:])
    return pred_series
