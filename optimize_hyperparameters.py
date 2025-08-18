import optuna
import pandas as pd
from src.data_loader import load_data
from src.feature_engineering import add_features
from src.model import train_model as train_transformer_model, predict as predict_transformer
from src.backtesting import Backtester

def objective(trial: optuna.Trial) -> float:
    """
    The objective function for Optuna to optimize.
    It trains a model with a given set of hyperparameters and returns
    a performance metric (e.g., Sharpe Ratio).
    """
    # --- Hyperparameter Search Space ---

    # Feature Engineering
    frac_diff_d = trial.suggest_float('frac_diff_d', 0.1, 0.9)
    ae_window_size = trial.suggest_int('ae_window_size', 10, 50)
    ae_encoding_dim = trial.suggest_int('ae_encoding_dim', 2, 10)

    # Transformer Model
    transformer_d_model = trial.suggest_categorical('transformer_d_model', [64, 128, 256])
    transformer_nhead = trial.suggest_categorical('transformer_nhead', [2, 4, 8])
    transformer_layers = trial.suggest_int('transformer_layers', 1, 4)
    transformer_lr = trial.suggest_loguniform('transformer_lr', 1e-5, 1e-3)

    # --- Pipeline Execution ---

    try:
        # 1. Load Data
        df = load_data('data/dummy_data.csv')

        # 2. Feature Engineering
        # We need to create a new add_features function that accepts hyperparameters
        # For now, let's assume we can modify the global state or pass them somehow.
        # A better way would be to refactor add_features to accept params.
        # For now, I will create a temporary version of add_features here.

        from src.feature_engineering import frac_diff_ffd, get_autoencoder_features, add_all_ta_features

        def add_features_optimized(df, d, window, encoding):
            df_with_ta = add_all_ta_features(df, open="open", high="high", low="low", close="close", volume="volume", fillna=True)
            close_price = df_with_ta[['close']]
            frac_diff_close = frac_diff_ffd(close_price, d=d)
            frac_diff_close.rename(columns={'close': 'close_frac_diff'}, inplace=True)
            df_with_features = df_with_ta.join(frac_diff_close)
            ae_features = get_autoencoder_features(df, window_size=window, encoding_dim=encoding)
            df_with_features = df_with_features.join(ae_features)
            df_with_features.fillna(method='ffill', inplace=True)
            df_with_features.fillna(method='bfill', inplace=True)
            df_with_features.dropna(inplace=True)
            return df_with_features

        df_features = add_features_optimized(df, frac_diff_d, ae_window_size, ae_encoding_dim)

        # 3. Target variable (simple up/down classification)
        df_features['target'] = (df_features['close'].shift(-1) > df_features['close']).astype(int)
        df_features.dropna(inplace=True)

        X = df_features.drop('target', axis=1)
        y = df_features['target']

        # 4. Train Model
        # We also need to refactor train_model to accept hyperparameters
        # I'll create a temporary wrapper here as well

        from src.model import TimeSeriesDataset, TransformerModel
        from torch.utils.data import DataLoader
        import torch

        def train_model_optimized(X, y, d_model, nhead, layers, lr):
            train_dataset = TimeSeriesDataset(X, y)
            train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
            model = TransformerModel(input_dim=X.shape[1], d_model=d_model, nhead=nhead, num_encoder_layers=layers, num_classes=2)
            criterion = torch.nn.CrossEntropyLoss()
            optimizer = torch.optim.Adam(model.parameters(), lr=lr)
            model.train()
            for epoch in range(5): # shorter training for optimization
                for batch_X, batch_y in train_loader:
                    batch_X = batch_X.permute(1, 0, 2)
                    optimizer.zero_grad()
                    output = model(batch_X)
                    loss = criterion(output, batch_y)
                    loss.backward()
                    optimizer.step()
            return model.cpu()

        model = train_model_optimized(X, y, transformer_d_model, transformer_nhead, transformer_layers, transformer_lr)

        # 5. Generate Signals
        predictions = predict_transformer(model, X)
        signals = pd.Series(0, index=X.index)
        signals[predictions == 1] = 1 # Buy
        signals[predictions == 0] = 2 # Sell (or hold, depending on interpretation) - let's say exit position

        # 6. Backtest
        backtester = Backtester(data=df.loc[X.index], signals=signals)
        metrics = backtester.run()

        # 7. Return Sharpe Ratio
        sharpe_ratio = float(metrics.get('Sharpe Ratio', '0.0').replace('%',''))

        return sharpe_ratio

    except Exception as e:
        print(f"An error occurred during the trial: {e}")
        return -1.0 # Return a bad value if the trial fails

if __name__ == '__main__':
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=10) # Using a small number of trials for demonstration

    print("Best trial:")
    trial = study.best_trial
    print(f"  Value: {trial.value}")
    print("  Params: ")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")
