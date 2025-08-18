import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

class Backtester:
    """
    A comprehensive backtesting engine to evaluate trading strategies.
    """
    def __init__(self, data: pd.DataFrame, signals: pd.Series, initial_capital=100000.0, transaction_cost_pct=0.001, slippage_pct=0.0005):
        self.data = data
        self.signals = signals
        self.initial_capital = initial_capital
        self.transaction_cost_pct = transaction_cost_pct
        self.slippage_pct = slippage_pct
        self.positions = self._generate_positions()
        self.portfolio = pd.DataFrame(index=self.data.index)

    def _generate_positions(self):
        """Generates positions based on signals."""
        positions = pd.Series(index=self.signals.index, dtype=float).fillna(0.0)
        positions[self.signals == 1] = 1  # Buy signal -> long position
        positions[self.signals == 2] = -1 # Sell signal -> short position (or exit long)
        # For simplicity, we assume we are either fully in or fully out.
        # A more complex model could handle partial positions.
        return positions.ffill().fillna(0) # Forward fill to hold positions

    def run(self):
        """Runs the backtest."""
        self.portfolio['positions'] = self.positions
        self.portfolio['market_returns'] = self.data['close'].pct_change()

        # Calculate strategy returns considering costs and slippage
        trades = self.portfolio['positions'].diff().fillna(0)
        trade_prices = self.data['open'].shift(-1) # Assume trade at next day's open

        # Apply slippage
        buy_slippage = trade_prices * self.slippage_pct
        sell_slippage = -trade_prices * self.slippage_pct
        trade_prices_with_slippage = trade_prices + (buy_slippage * (trades > 0)) + (sell_slippage * (trades < 0))

        # Calculate portfolio returns
        self.portfolio['strategy_returns'] = self.portfolio['positions'].shift(1) * self.portfolio['market_returns']

        # Deduct transaction costs
        costs = abs(trades) * self.transaction_cost_pct
        self.portfolio['strategy_returns'] -= costs

        self.portfolio['cumulative_market_returns'] = (1 + self.portfolio['market_returns']).cumprod()
        self.portfolio['cumulative_strategy_returns'] = (1 + self.portfolio['strategy_returns']).cumprod()

        self.portfolio['portfolio_value'] = self.initial_capital * self.portfolio['cumulative_strategy_returns']
        self.portfolio.dropna(inplace=True)

        return self.calculate_metrics()

    def calculate_metrics(self):
        """Calculates performance metrics."""
        metrics = {}

        total_return = self.portfolio['cumulative_strategy_returns'].iloc[-1] - 1
        metrics['Total Return'] = f"{total_return:.2%}"

        days = len(self.portfolio)
        annualized_return = (1 + total_return) ** (252.0 / days) - 1
        metrics['Annualized Return'] = f"{annualized_return:.2%}"

        annualized_volatility = self.portfolio['strategy_returns'].std() * np.sqrt(252)
        metrics['Annualized Volatility'] = f"{annualized_volatility:.2%}"

        sharpe_ratio = annualized_return / annualized_volatility if annualized_volatility != 0 else 0
        metrics['Sharpe Ratio'] = f"{sharpe_ratio:.2f}"

        # Max Drawdown
        rolling_max = self.portfolio['cumulative_strategy_returns'].cummax()
        drawdown = self.portfolio['cumulative_strategy_returns'] / rolling_max - 1.0
        max_drawdown = drawdown.min()
        metrics['Max Drawdown'] = f"{max_drawdown:.2%}"

        calmar_ratio = annualized_return / abs(max_drawdown) if max_drawdown != 0 else 0
        metrics['Calmar Ratio'] = f"{calmar_ratio:.2f}"

        # Win Rate and Profit Factor
        trades = self.portfolio['strategy_returns'][self.portfolio['strategy_returns'] != 0]
        win_rate = (trades > 0).sum() / len(trades) if len(trades) > 0 else 0
        metrics['Win Rate'] = f"{win_rate:.2%}"

        gains = trades[trades > 0].sum()
        losses = abs(trades[trades < 0].sum())
        profit_factor = gains / losses if losses != 0 else np.inf
        metrics['Profit Factor'] = f"{profit_factor:.2f}"

        self.metrics = metrics
        return metrics

    def plot_performance(self):
        """Plots the performance of the strategy."""
        plt.figure(figsize=(12, 8))
        plt.plot(self.portfolio['cumulative_strategy_returns'], label='Strategy')
        plt.plot(self.portfolio['cumulative_market_returns'], label='Market (Buy and Hold)')
        plt.title('Backtest Performance')
        plt.xlabel('Date')
        plt.ylabel('Cumulative Returns')
        plt.legend()
        plt.grid(True)
        plt.show()
