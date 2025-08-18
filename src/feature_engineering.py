import pandas as pd
import numpy as np
from ta import add_all_ta_features
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# --- Fractional Differentiation ---

def get_weights_ffd(d, thres):
    """
    Computes the weights for fractional differentiation.
    """
    w, k = [1.], 1
    while True:
        w_ = -w[-1] / k * (d - k + 1)
        if abs(w_) < thres:
            break
        w.append(w_)
        k += 1
    return np.array(w[::-1]).reshape(-1, 1)

def frac_diff_ffd(series, d, thres=1e-5):
    """
    Computes the fractionally differentiated series.
    Uses a fixed-width window approach.
    """
    w = get_weights_ffd(d, thres)
    width = len(w) - 1
    df = {}
    for name in series.columns:
        seriesF = series[[name]].fillna(method='ffill').dropna()
        df_ = pd.Series(dtype=float)
        for iloc1 in range(width, len(seriesF)):
            loc0, loc1 = seriesF.index[iloc1 - width], seriesF.index[iloc1]
            if not np.isfinite(series.loc[loc1, name]):
                continue
            df_[loc1] = np.dot(w.T, seriesF.loc[loc0:loc1, name])[0]
        df[name] = df_
    df = pd.concat(df, axis=1)
    return df

# --- Autoencoder for Feature Extraction ---

class Autoencoder(nn.Module):
    def __init__(self, input_dim, encoding_dim):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, encoding_dim)
        )
        self.decoder = nn.Sequential(
            nn.Linear(encoding_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, input_dim),
            nn.Sigmoid() # to keep output between 0 and 1
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

    def get_features(self, x):
        return self.encoder(x)

def train_autoencoder(data, input_dim, encoding_dim, epochs=50, lr=1e-3):
    # Scale data to be between 0 and 1 for the sigmoid output layer
    scaler = lambda x: (x - x.min()) / (x.max() - x.min())
    scaled_data = data.apply(scaler)

    dataset = TensorDataset(torch.tensor(scaled_data.values, dtype=torch.float32))
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True)

    model = Autoencoder(input_dim, encoding_dim)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        for batch in dataloader:
            inputs = batch[0]
            outputs = model(inputs)
            loss = criterion(outputs, inputs)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    return model.cpu() # return model on cpu

def get_autoencoder_features(df, window_size=20, encoding_dim=5):
    # Use returns for training the autoencoder
    returns = df['close'].pct_change().dropna()

    # Create rolling windows of returns
    windows = returns.rolling(window=window_size).apply(lambda x: x.tolist(), raw=False).dropna()
    data_for_autoencoder = pd.DataFrame(windows.tolist(), index=windows.index)

    # Train the autoencoder
    trained_model = train_autoencoder(data_for_autoencoder, input_dim=window_size, encoding_dim=encoding_dim)

    # Generate features for the entire dataset
    with torch.no_grad():
        all_windows_tensor = torch.tensor(data_for_autoencoder.values, dtype=torch.float32)
        features = trained_model.get_features(all_windows_tensor).numpy()

    feature_df = pd.DataFrame(features, index=data_for_autoencoder.index, columns=[f'ae_feat_{i}' for i in range(encoding_dim)])
    return feature_df

# --- Main Feature Engineering Function ---

def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds a comprehensive set of features to the OHLCVT data.
    """
    # 1. Add standard technical indicators
    df_with_ta = add_all_ta_features(
        df, open="open", high="high", low="low", close="close", volume="volume", fillna=True
    )

    # 2. Add fractionally differentiated close price
    close_price = df_with_ta[['close']]
    frac_diff_close = frac_diff_ffd(close_price, d=0.5)
    frac_diff_close.rename(columns={'close': 'close_frac_diff'}, inplace=True)
    df_with_features = df_with_ta.join(frac_diff_close)

    # 3. Add autoencoder features
    ae_features = get_autoencoder_features(df)
    df_with_features = df_with_features.join(ae_features)

    # Handle NaNs from merging and feature creation
    df_with_features.fillna(method='ffill', inplace=True)
    df_with_features.fillna(method='bfill', inplace=True)
    df_with_features.dropna(inplace=True) # drop any remaining NaNs

    print("Features added successfully.")
    return df_with_features
