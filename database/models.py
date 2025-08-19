"""
Database models for Asset-ML-Strategy
Production-grade database schema with proper relationships and indexes
"""

from datetime import datetime
from decimal import Decimal
from typing import Optional
from sqlalchemy import Column, Integer, String, DateTime, Boolean, Numeric, Text, ForeignKey, Index, Enum
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
import enum

Base = declarative_base()

class UserRole(enum.Enum):
    ADMIN = "admin"
    TRADER = "trader"
    VIEWER = "viewer"

class TradeAction(enum.Enum):
    BUY = "buy"
    SELL = "sell"

class TradeStatus(enum.Enum):
    PENDING = "pending"
    EXECUTED = "executed"
    CANCELLED = "cancelled"
    REJECTED = "rejected"

class StrategyStatus(enum.Enum):
    ACTIVE = "active"
    INACTIVE = "inactive"
    PAUSED = "paused"

class User(Base):
    """User accounts with role-based access control"""
    __tablename__ = "users"
    
    id = Column(Integer, primary_key=True, index=True)
    email = Column(String(255), unique=True, index=True, nullable=False)
    hashed_password = Column(String(255), nullable=False)
    full_name = Column(String(255), nullable=False)
    role = Column(Enum(UserRole), default=UserRole.TRADER, nullable=False)
    is_active = Column(Boolean, default=True)
    is_verified = Column(Boolean, default=False)
    subscription_tier = Column(String(50), default="basic")
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    last_login = Column(DateTime)
    
    # Relationships
    trading_accounts = relationship("TradingAccount", back_populates="user", cascade="all, delete-orphan")
    strategies = relationship("Strategy", back_populates="user", cascade="all, delete-orphan")
    trades = relationship("Trade", back_populates="user", cascade="all, delete-orphan")

class TradingAccount(Base):
    """Trading accounts with balance and risk limits"""
    __tablename__ = "trading_accounts"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    account_name = Column(String(255), nullable=False)
    account_type = Column(String(50), default="paper")  # paper, live
    broker = Column(String(100))
    initial_balance = Column(Numeric(15, 2), nullable=False)
    current_balance = Column(Numeric(15, 2), nullable=False)
    available_balance = Column(Numeric(15, 2), nullable=False)
    total_pnl = Column(Numeric(15, 2), default=0)
    daily_pnl = Column(Numeric(15, 2), default=0)
    max_daily_loss = Column(Numeric(15, 2), default=1000)
    max_position_size = Column(Numeric(5, 4), default=0.1)  # As percentage
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    user = relationship("User", back_populates="trading_accounts")
    trades = relationship("Trade", back_populates="account", cascade="all, delete-orphan")
    positions = relationship("Position", back_populates="account", cascade="all, delete-orphan")
    
    # Indexes
    __table_args__ = (
        Index('ix_trading_accounts_user_active', 'user_id', 'is_active'),
    )

class Strategy(Base):
    """Trading strategies with parameters and performance metrics"""
    __tablename__ = "strategies"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    name = Column(String(255), nullable=False)
    description = Column(Text)
    strategy_type = Column(String(100), default="ml_ensemble")
    parameters = Column(Text)  # JSON string
    risk_parameters = Column(Text)  # JSON string
    status = Column(Enum(StrategyStatus), default=StrategyStatus.ACTIVE)
    
    # Performance metrics
    total_return = Column(Numeric(10, 4), default=0)
    sharpe_ratio = Column(Numeric(10, 4), default=0)
    max_drawdown = Column(Numeric(10, 4), default=0)
    win_rate = Column(Numeric(5, 4), default=0)
    profit_factor = Column(Numeric(10, 4), default=0)
    total_trades = Column(Integer, default=0)
    
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    user = relationship("User", back_populates="strategies")
    trades = relationship("Trade", back_populates="strategy")
    
    # Indexes
    __table_args__ = (
        Index('ix_strategies_user_status', 'user_id', 'status'),
    )

class Trade(Base):
    """Individual trades with full audit trail"""
    __tablename__ = "trades"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    account_id = Column(Integer, ForeignKey("trading_accounts.id"), nullable=False)
    strategy_id = Column(Integer, ForeignKey("strategies.id"), nullable=True)
    
    # Trade details
    symbol = Column(String(20), nullable=False, index=True)
    action = Column(Enum(TradeAction), nullable=False)
    quantity = Column(Numeric(15, 4), nullable=False)
    order_price = Column(Numeric(15, 4))
    executed_price = Column(Numeric(15, 4))
    commission = Column(Numeric(10, 2), default=0)
    slippage = Column(Numeric(10, 4), default=0)
    
    # Status and timing
    status = Column(Enum(TradeStatus), default=TradeStatus.PENDING)
    order_time = Column(DateTime, default=datetime.utcnow)
    execution_time = Column(DateTime)
    
    # P&L
    unrealized_pnl = Column(Numeric(15, 2), default=0)
    realized_pnl = Column(Numeric(15, 2), default=0)
    
    # Signal information
    signal_confidence = Column(Numeric(5, 4))
    signal_strength = Column(Numeric(5, 4))
    model_predictions = Column(Text)  # JSON string
    
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    user = relationship("User", back_populates="trades")
    account = relationship("TradingAccount", back_populates="trades")
    strategy = relationship("Strategy", back_populates="trades")
    
    # Indexes
    __table_args__ = (
        Index('ix_trades_symbol_time', 'symbol', 'order_time'),
        Index('ix_trades_account_time', 'account_id', 'order_time'),
        Index('ix_trades_strategy_time', 'strategy_id', 'order_time'),
    )

class Position(Base):
    """Current positions and their status"""
    __tablename__ = "positions"
    
    id = Column(Integer, primary_key=True, index=True)
    account_id = Column(Integer, ForeignKey("trading_accounts.id"), nullable=False)
    symbol = Column(String(20), nullable=False, index=True)
    quantity = Column(Numeric(15, 4), nullable=False)
    average_price = Column(Numeric(15, 4), nullable=False)
    current_price = Column(Numeric(15, 4))
    market_value = Column(Numeric(15, 2))
    unrealized_pnl = Column(Numeric(15, 2), default=0)
    unrealized_pnl_percent = Column(Numeric(10, 4), default=0)
    
    # Risk management
    stop_loss = Column(Numeric(15, 4))
    take_profit = Column(Numeric(15, 4))
    
    opened_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    account = relationship("TradingAccount", back_populates="positions")
    
    # Unique constraint
    __table_args__ = (
        Index('ix_positions_account_symbol', 'account_id', 'symbol', unique=True),
    )

class MarketData(Base):
    """Market data storage for backtesting and analysis"""
    __tablename__ = "market_data"
    
    id = Column(Integer, primary_key=True, index=True)
    symbol = Column(String(20), nullable=False, index=True)
    timestamp = Column(DateTime, nullable=False, index=True)
    timeframe = Column(String(10), nullable=False)  # 1m, 5m, 1h, 1d, etc.
    
    open_price = Column(Numeric(15, 4), nullable=False)
    high_price = Column(Numeric(15, 4), nullable=False)
    low_price = Column(Numeric(15, 4), nullable=False)
    close_price = Column(Numeric(15, 4), nullable=False)
    volume = Column(Numeric(20, 4), default=0)
    
    # Additional fields
    adjusted_close = Column(Numeric(15, 4))
    dividend_amount = Column(Numeric(10, 4))
    split_coefficient = Column(Numeric(10, 4))
    
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Indexes
    __table_args__ = (
        Index('ix_market_data_symbol_timeframe_timestamp', 'symbol', 'timeframe', 'timestamp', unique=True),
    )

class Signal(Base):
    """Trading signals generated by strategies"""
    __tablename__ = "signals"
    
    id = Column(Integer, primary_key=True, index=True)
    strategy_id = Column(Integer, ForeignKey("strategies.id"), nullable=False)
    symbol = Column(String(20), nullable=False, index=True)
    action = Column(Enum(TradeAction), nullable=False)
    confidence = Column(Numeric(5, 4), nullable=False)
    strength = Column(Numeric(5, 4), nullable=False)
    target_price = Column(Numeric(15, 4))
    stop_loss = Column(Numeric(15, 4))
    take_profit = Column(Numeric(15, 4))
    
    # Model outputs
    model_predictions = Column(Text)  # JSON string
    feature_importance = Column(Text)  # JSON string
    
    # Status
    is_executed = Column(Boolean, default=False)
    execution_notes = Column(Text)
    
    created_at = Column(DateTime, default=datetime.utcnow, index=True)
    expires_at = Column(DateTime)
    
    # Indexes
    __table_args__ = (
        Index('ix_signals_symbol_created', 'symbol', 'created_at'),
        Index('ix_signals_strategy_created', 'strategy_id', 'created_at'),
    )

class BacktestResult(Base):
    """Backtest results for strategies"""
    __tablename__ = "backtest_results"
    
    id = Column(Integer, primary_key=True, index=True)
    strategy_id = Column(Integer, ForeignKey("strategies.id"), nullable=False)
    name = Column(String(255), nullable=False)
    description = Column(Text)
    
    # Test parameters
    start_date = Column(DateTime, nullable=False)
    end_date = Column(DateTime, nullable=False)
    initial_capital = Column(Numeric(15, 2), nullable=False)
    symbols = Column(Text)  # JSON array
    parameters = Column(Text)  # JSON string
    
    # Results
    total_return = Column(Numeric(10, 4))
    annualized_return = Column(Numeric(10, 4))
    volatility = Column(Numeric(10, 4))
    sharpe_ratio = Column(Numeric(10, 4))
    sortino_ratio = Column(Numeric(10, 4))
    calmar_ratio = Column(Numeric(10, 4))
    max_drawdown = Column(Numeric(10, 4))
    win_rate = Column(Numeric(5, 4))
    profit_factor = Column(Numeric(10, 4))
    total_trades = Column(Integer)
    
    # Detailed results
    performance_chart = Column(Text)  # Base64 or path
    trade_history = Column(Text)  # JSON
    daily_returns = Column(Text)  # JSON
    
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    strategy = relationship("Strategy")

class SystemLog(Base):
    """System logs for monitoring and debugging"""
    __tablename__ = "system_logs"
    
    id = Column(Integer, primary_key=True, index=True)
    level = Column(String(20), nullable=False, index=True)  # INFO, WARNING, ERROR, etc.
    module = Column(String(100), nullable=False, index=True)
    message = Column(Text, nullable=False)
    details = Column(Text)  # JSON string for additional context
    user_id = Column(Integer, ForeignKey("users.id"))
    
    created_at = Column(DateTime, default=datetime.utcnow, index=True)
    
    # Indexes
    __table_args__ = (
        Index('ix_system_logs_level_created', 'level', 'created_at'),
        Index('ix_system_logs_module_created', 'module', 'created_at'),
    )