import pandas as pd
from src.data_loader import load_data
from src.feature_engineering import add_features
from src.rl_env import TradingEnv
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

def train_agent():
    """
    The main function to train the RL agent.
    """
    # 1. Load and preprocess data
    print("Loading and preprocessing data...")
    df = load_data('data/dummy_data.csv')

    # 2. Add features
    print("Adding features...")
    df_with_features = add_features(df)

    # The feature engineering might create NaNs at the beginning, so we drop them
    df_with_features.dropna(inplace=True)

    # 3. Create the trading environment
    print("Creating trading environment...")
    env = DummyVecEnv([lambda: TradingEnv(df_with_features)])

    # 4. Instantiate and train the PPO agent
    print("Training PPO agent...")
    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        tensorboard_log="./tensorboard_logs/"
    )

    model.learn(total_timesteps=20000) # Using a small number for demonstration

    # 5. Save the trained model
    print("Saving trained model...")
    model.save("ppo_trading_agent")

    print("Training complete!")

if __name__ == '__main__':
    train_agent()
