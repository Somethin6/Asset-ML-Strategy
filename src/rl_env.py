import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd

class TradingEnv(gym.Env):
    """
    A custom trading environment for reinforcement learning.
    It follows the OpenAI Gym interface.
    """
    metadata = {'render_modes': ['human']}

    def __init__(self, df: pd.DataFrame, initial_capital=100000, transaction_cost_pct=0.001):
        super(TradingEnv, self).__init__()

        self.df = df
        self.initial_capital = initial_capital
        self.transaction_cost_pct = transaction_cost_pct

        # Define action space: 0: hold, 1: buy, 2: sell
        self.action_space = spaces.Discrete(3)

        # Define observation space
        # It consists of the market data from the df + portfolio info
        # The portfolio info will be: current portfolio value, shares held
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(len(df.columns) + 2,), # df columns + portfolio value + shares held
            dtype=np.float32
        )

        self.current_step = 0
        self.done = False
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0
        self.capital = self.initial_capital
        self.shares_held = 0
        self.portfolio_value = self.initial_capital
        self.done = False

        obs = self._get_obs()
        info = self._get_info()
        return obs, info

    def step(self, action):
        if self.done:
            return self.reset()

        current_price = self.df['close'].iloc[self.current_step]

        # Execute action
        if action == 1: # Buy
            # Buy with all available capital
            if self.capital > 0:
                cost = self.capital * (1 - self.transaction_cost_pct)
                self.shares_held += cost / current_price
                self.capital = 0

        elif action == 2: # Sell
            # Sell all held shares
            if self.shares_held > 0:
                revenue = self.shares_held * current_price * (1 - self.transaction_cost_pct)
                self.capital += revenue
                self.shares_held = 0

        # Hold action (action == 0) does nothing

        # Update portfolio value
        prev_portfolio_value = self.portfolio_value
        self.portfolio_value = self.capital + self.shares_held * current_price

        # Calculate reward
        reward = self.portfolio_value - prev_portfolio_value

        # Move to the next step
        self.current_step += 1
        if self.current_step >= len(self.df) - 1:
            self.done = True

        obs = self._get_obs()
        info = self._get_info()

        return obs, reward, self.done, False, info # Gymnasium returns 5 values

    def _get_obs(self):
        obs = self.df.iloc[self.current_step].values
        portfolio_state = np.array([self.portfolio_value, self.shares_held])
        return np.concatenate([obs, portfolio_state]).astype(np.float32)

    def _get_info(self):
        return {
            'portfolio_value': self.portfolio_value,
            'capital': self.capital,
            'shares_held': self.shares_held,
            'current_step': self.current_step
        }

    def render(self, mode='human'):
        if mode == 'human':
            print(f"Step: {self.current_step}")
            print(f"Portfolio Value: {self.portfolio_value}")
            print(f"Capital: {self.capital}")
            print(f"Shares Held: {self.shares_held}")
            print("-" * 20)

    def close(self):
        pass
