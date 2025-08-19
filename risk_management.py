#!/usr/bin/env python3
"""
Advanced Risk Management System for ML Trading Strategy.
Implements position sizing, drawdown controls, and portfolio heat management.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
import logging
from dataclasses import dataclass

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class RiskMetrics:
    """Data class for risk metrics"""
    max_position_size: float = 0.1  # Maximum position size as % of portfolio
    max_daily_loss: float = 0.02    # Maximum daily loss as % of portfolio
    max_drawdown: float = 0.1       # Maximum drawdown before stopping
    kelly_multiplier: float = 0.25  # Kelly criterion multiplier (conservative)
    portfolio_heat: float = 0.0     # Current portfolio heat
    var_95: float = 0.0            # Value at Risk 95%
    sharpe_ratio: float = 0.0      # Sharpe ratio
    win_rate: float = 0.0          # Win rate

@dataclass
class Position:
    """Data class for a trading position"""
    symbol: str
    size: float
    entry_price: float
    entry_time: pd.Timestamp
    side: str  # 'long' or 'short'
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    current_price: float = 0.0
    unrealized_pnl: float = 0.0
    
    def update_pnl(self, current_price: float):
        """Update unrealized PnL"""
        self.current_price = current_price
        if self.side == 'long':
            self.unrealized_pnl = (current_price - self.entry_price) * self.size
        else:
            self.unrealized_pnl = (self.entry_price - current_price) * self.size

class RiskManager:
    """
    Advanced risk management system for trading strategies.
    """
    
    def __init__(self, initial_capital: float = 100000.0, risk_metrics: Optional[RiskMetrics] = None):
        """
        Initialize the risk manager.
        
        Args:
            initial_capital: Initial portfolio capital
            risk_metrics: Risk management parameters
        """
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        self.risk_metrics = risk_metrics or RiskMetrics()
        
        # Trading state
        self.positions: Dict[str, Position] = {}
        self.closed_trades: List[Dict] = []
        self.daily_pnl: List[float] = []
        self.portfolio_value_history: List[float] = [initial_capital]
        
        # Risk monitoring
        self.max_portfolio_value = initial_capital
        self.current_drawdown = 0.0
        self.daily_loss = 0.0
        self.trading_enabled = True
        
        logger.info(f"Risk Manager initialized with ${initial_capital:,.2f} capital")
    
    def calculate_kelly_position_size(
        self, 
        win_rate: float, 
        avg_win: float, 
        avg_loss: float, 
        current_price: float
    ) -> float:
        """
        Calculate position size using Kelly criterion.
        
        Args:
            win_rate: Historical win rate (0-1)
            avg_win: Average winning trade amount
            avg_loss: Average losing trade amount (positive value)
            current_price: Current asset price
        
        Returns:
            Recommended position size (number of shares)
        """
        if avg_loss <= 0 or win_rate <= 0:
            return 0.0
        
        # Kelly criterion: f = (bp - q) / b
        # Where b = avg_win/avg_loss, p = win_rate, q = 1-win_rate
        b = avg_win / avg_loss
        p = win_rate
        q = 1 - win_rate
        
        kelly_fraction = (b * p - q) / b
        
        # Apply conservative multiplier
        kelly_fraction *= self.risk_metrics.kelly_multiplier
        
        # Cap at maximum position size
        kelly_fraction = min(kelly_fraction, self.risk_metrics.max_position_size)
        kelly_fraction = max(kelly_fraction, 0)  # No negative positions from Kelly
        
        # Convert to number of shares
        max_capital_to_risk = self.current_capital * kelly_fraction
        position_size = max_capital_to_risk / current_price if current_price > 0 else 0
        
        return position_size
    
    def calculate_atr_position_size(
        self, 
        data: pd.DataFrame, 
        atr_period: int = 14, 
        risk_per_trade: float = 0.01
    ) -> float:
        """
        Calculate position size based on Average True Range (ATR).
        
        Args:
            data: OHLC data
            atr_period: ATR calculation period
            risk_per_trade: Risk per trade as % of portfolio
        
        Returns:
            Position size based on ATR
        """
        if len(data) < atr_period:
            return 0.0
        
        # Calculate ATR
        high_low = data['high'] - data['low']
        high_close = np.abs(data['high'] - data['close'].shift())
        low_close = np.abs(data['low'] - data['close'].shift())
        
        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        atr = true_range.rolling(atr_period).mean().iloc[-1]
        
        if atr <= 0:
            return 0.0
        
        # Position size = Risk amount / ATR
        risk_amount = self.current_capital * risk_per_trade
        current_price = data['close'].iloc[-1]
        
        # Use ATR as stop loss distance
        position_size = risk_amount / atr if current_price > 0 else 0
        
        return position_size
    
    def check_risk_limits(self, symbol: str, position_size: float, price: float) -> Tuple[bool, str]:
        """
        Check if a new position would violate risk limits.
        
        Args:
            symbol: Trading symbol
            position_size: Requested position size
            price: Current price
        
        Returns:
            Tuple of (allowed, reason)
        """
        if not self.trading_enabled:
            return False, "Trading disabled due to risk limits"
        
        # Check if daily loss limit exceeded
        if self.daily_loss >= self.risk_metrics.max_daily_loss:
            return False, f"Daily loss limit exceeded: {self.daily_loss:.2%}"
        
        # Check if maximum drawdown exceeded
        if self.current_drawdown >= self.risk_metrics.max_drawdown:
            return False, f"Maximum drawdown exceeded: {self.current_drawdown:.2%}"
        
        # Check position size limit
        position_value = position_size * price
        position_pct = position_value / self.current_capital
        
        if position_pct > self.risk_metrics.max_position_size:
            return False, f"Position size too large: {position_pct:.2%} > {self.risk_metrics.max_position_size:.2%}"
        
        # Check if symbol already has position (simple implementation)
        if symbol in self.positions:
            return False, f"Already have position in {symbol}"
        
        return True, "Risk checks passed"
    
    def open_position(
        self, 
        symbol: str, 
        size: float, 
        price: float, 
        side: str = 'long',
        stop_loss: Optional[float] = None,
        take_profit: Optional[float] = None
    ) -> bool:
        """
        Open a new trading position.
        
        Args:
            symbol: Trading symbol
            size: Position size (number of shares)
            price: Entry price
            side: 'long' or 'short'
            stop_loss: Stop loss price
            take_profit: Take profit price
        
        Returns:
            True if position opened successfully
        """
        # Check risk limits
        allowed, reason = self.check_risk_limits(symbol, size, price)
        if not allowed:
            logger.warning(f"Position not opened for {symbol}: {reason}")
            return False
        
        # Create position
        position = Position(
            symbol=symbol,
            size=size,
            entry_price=price,
            entry_time=pd.Timestamp.now(),
            side=side,
            stop_loss=stop_loss,
            take_profit=take_profit
        )
        
        self.positions[symbol] = position
        
        # Update capital (assuming we use cash for the position)
        position_value = size * price
        self.current_capital -= position_value
        
        logger.info(f"Opened {side} position: {size:.2f} shares of {symbol} at ${price:.2f}")
        return True
    
    def close_position(self, symbol: str, price: float, reason: str = "Manual") -> bool:
        """
        Close an existing position.
        
        Args:
            symbol: Trading symbol
            price: Exit price
            reason: Reason for closing
        
        Returns:
            True if position closed successfully
        """
        if symbol not in self.positions:
            logger.warning(f"No position found for {symbol}")
            return False
        
        position = self.positions[symbol]
        
        # Calculate P&L
        if position.side == 'long':
            pnl = (price - position.entry_price) * position.size
        else:
            pnl = (position.entry_price - price) * position.size
        
        # Update capital
        position_value = position.size * price
        self.current_capital += position_value
        
        # Record closed trade
        trade_record = {
            'symbol': symbol,
            'side': position.side,
            'size': position.size,
            'entry_price': position.entry_price,
            'exit_price': price,
            'entry_time': position.entry_time,
            'exit_time': pd.Timestamp.now(),
            'pnl': pnl,
            'reason': reason
        }
        
        self.closed_trades.append(trade_record)
        
        # Remove position
        del self.positions[symbol]
        
        logger.info(f"Closed position: {symbol} for ${pnl:.2f} P&L ({reason})")
        return True
    
    def update_positions(self, market_data: Dict[str, float]):
        """
        Update all positions with current market prices.
        
        Args:
            market_data: Dictionary of symbol -> current_price
        """
        for symbol, position in self.positions.items():
            if symbol in market_data:
                current_price = market_data[symbol]
                position.update_pnl(current_price)
                
                # Check stop loss and take profit
                if position.side == 'long':
                    if position.stop_loss and current_price <= position.stop_loss:
                        self.close_position(symbol, current_price, "Stop Loss")
                    elif position.take_profit and current_price >= position.take_profit:
                        self.close_position(symbol, current_price, "Take Profit")
                
                elif position.side == 'short':
                    if position.stop_loss and current_price >= position.stop_loss:
                        self.close_position(symbol, current_price, "Stop Loss")
                    elif position.take_profit and current_price <= position.take_profit:
                        self.close_position(symbol, current_price, "Take Profit")
    
    def calculate_portfolio_metrics(self) -> RiskMetrics:
        """
        Calculate current portfolio risk metrics.
        
        Returns:
            Updated risk metrics
        """
        # Current portfolio value
        portfolio_value = self.current_capital
        for position in self.positions.values():
            portfolio_value += position.current_price * position.size
        
        self.portfolio_value_history.append(portfolio_value)
        
        # Update maximum portfolio value and drawdown
        if portfolio_value > self.max_portfolio_value:
            self.max_portfolio_value = portfolio_value
        
        self.current_drawdown = (self.max_portfolio_value - portfolio_value) / self.max_portfolio_value
        
        # Calculate metrics from closed trades
        if self.closed_trades:
            pnls = [trade['pnl'] for trade in self.closed_trades]
            wins = [pnl for pnl in pnls if pnl > 0]
            losses = [pnl for pnl in pnls if pnl < 0]
            
            win_rate = len(wins) / len(pnls) if pnls else 0
            avg_win = np.mean(wins) if wins else 0
            avg_loss = abs(np.mean(losses)) if losses else 0
            
            # Calculate Sharpe ratio (simplified)
            if len(self.portfolio_value_history) > 1:
                returns = pd.Series(self.portfolio_value_history).pct_change().dropna()
                sharpe_ratio = returns.mean() / returns.std() * np.sqrt(252) if returns.std() > 0 else 0
            else:
                sharpe_ratio = 0
            
            # Update risk metrics
            self.risk_metrics.win_rate = win_rate
            self.risk_metrics.sharpe_ratio = sharpe_ratio
            self.risk_metrics.portfolio_heat = sum(abs(pos.unrealized_pnl) for pos in self.positions.values()) / portfolio_value
        
        # Check if we need to disable trading
        if self.current_drawdown >= self.risk_metrics.max_drawdown:
            self.trading_enabled = False
            logger.warning(f"Trading disabled: Maximum drawdown reached ({self.current_drawdown:.2%})")
        
        return self.risk_metrics
    
    def get_portfolio_summary(self) -> Dict:
        """
        Get a summary of the current portfolio state.
        
        Returns:
            Dictionary with portfolio summary
        """
        portfolio_value = self.current_capital
        unrealized_pnl = 0
        
        for position in self.positions.values():
            position_value = position.current_price * position.size
            portfolio_value += position_value
            unrealized_pnl += position.unrealized_pnl
        
        realized_pnl = sum(trade['pnl'] for trade in self.closed_trades)
        total_pnl = realized_pnl + unrealized_pnl
        
        return {
            'initial_capital': self.initial_capital,
            'current_capital': self.current_capital,
            'portfolio_value': portfolio_value,
            'total_pnl': total_pnl,
            'realized_pnl': realized_pnl,
            'unrealized_pnl': unrealized_pnl,
            'total_return': (portfolio_value - self.initial_capital) / self.initial_capital,
            'current_drawdown': self.current_drawdown,
            'max_drawdown': max(self.current_drawdown, self.risk_metrics.max_drawdown),
            'open_positions': len(self.positions),
            'total_trades': len(self.closed_trades),
            'trading_enabled': self.trading_enabled
        }

def calculate_optimal_stop_loss(data: pd.DataFrame, atr_multiplier: float = 2.0) -> float:
    """
    Calculate optimal stop loss based on ATR.
    
    Args:
        data: OHLC data
        atr_multiplier: Multiplier for ATR-based stop loss
    
    Returns:
        Stop loss distance as percentage of current price
    """
    if len(data) < 14:
        return 0.02  # Default 2% if not enough data
    
    # Calculate ATR
    high_low = data['high'] - data['low']
    high_close = np.abs(data['high'] - data['close'].shift())
    low_close = np.abs(data['low'] - data['close'].shift())
    
    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    atr = true_range.rolling(14).mean().iloc[-1]
    
    current_price = data['close'].iloc[-1]
    
    # Stop loss as percentage of current price
    stop_loss_pct = (atr * atr_multiplier) / current_price if current_price > 0 else 0.02
    
    # Cap between 1% and 10%
    return max(0.01, min(0.10, stop_loss_pct))

if __name__ == '__main__':
    # Example usage
    risk_manager = RiskManager(initial_capital=100000.0)
    
    # Simulate some trades
    risk_manager.open_position('AAPL', 100, 150.0, 'long', stop_loss=145.0, take_profit=160.0)
    risk_manager.open_position('GOOGL', 50, 120.0, 'long', stop_loss=115.0, take_profit=130.0)
    
    # Update with market data
    market_data = {'AAPL': 155.0, 'GOOGL': 125.0}
    risk_manager.update_positions(market_data)
    
    # Get portfolio summary
    summary = risk_manager.get_portfolio_summary()
    print("Portfolio Summary:")
    for key, value in summary.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.2f}")
        else:
            print(f"  {key}: {value}")
    
    print("Advanced Risk Management System ready!")