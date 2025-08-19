#!/usr/bin/env python3
"""
Live Trading Interface for MoneyPrinter
Handles real-time trading, monitoring, and execution.
"""

import asyncio
import threading
import time
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Callable
import pandas as pd
import numpy as np
import json
from dataclasses import dataclass, asdict
from enum import Enum

from config_manager import ConfigManager, MoneyPrinterConfig
from risk_management import RiskManager, Position
from moneyprinter_strategy import MoneyPrinterStrategy, AdvancedFeatureEngineer

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TradingState(Enum):
    """Trading system states"""
    STOPPED = "stopped"
    STARTING = "starting"
    RUNNING = "running"
    PAUSED = "paused"
    ERROR = "error"
    EMERGENCY_STOP = "emergency_stop"

@dataclass
class LiveTrade:
    """Live trading record"""
    timestamp: datetime
    symbol: str
    action: str  # 'BUY', 'SELL', 'HOLD'
    quantity: float
    price: float
    confidence: float
    reason: str
    execution_time: float
    status: str  # 'PENDING', 'EXECUTED', 'FAILED'

@dataclass
class SystemMetrics:
    """System performance metrics"""
    timestamp: datetime
    portfolio_value: float
    total_pnl: float
    daily_pnl: float
    drawdown: float
    positions_count: int
    model_confidence: float
    execution_latency: float
    system_health: float

class AlertSystem:
    """
    Alert and notification system for live trading.
    """
    
    def __init__(self, config: MoneyPrinterConfig):
        self.config = config
        self.alerts_sent = []
        
    def send_alert(self, message: str, level: str = "INFO", data: Optional[Dict] = None):
        """
        Send alert notification.
        
        Args:
            message: Alert message
            level: Alert level (INFO, WARNING, ERROR, CRITICAL)
            data: Additional data to include
        """
        alert = {
            'timestamp': datetime.now().isoformat(),
            'level': level,
            'message': message,
            'data': data or {}
        }
        
        self.alerts_sent.append(alert)
        
        # Log the alert
        logger.log(getattr(logging, level), f"ALERT: {message}")
        
        # Send email alerts if configured
        if self.config.alerts.enable_email_alerts and level in ['ERROR', 'CRITICAL']:
            self._send_email_alert(alert)
        
        # Send webhook alerts if configured
        if self.config.alerts.enable_webhook_alerts:
            self._send_webhook_alert(alert)
    
    def _send_email_alert(self, alert: Dict):
        """Send email alert (placeholder implementation)"""
        logger.info(f"Would send email alert: {alert['message']}")
    
    def _send_webhook_alert(self, alert: Dict):
        """Send webhook alert (placeholder implementation)"""
        logger.info(f"Would send webhook alert: {alert['message']}")

class LiveTradingEngine:
    """
    Main live trading engine that orchestrates all components.
    """
    
    def __init__(self, config_file: str = 'config/moneyprinter_config.yaml'):
        """
        Initialize the live trading engine.
        
        Args:
            config_file: Path to configuration file
        """
        self.config_manager = ConfigManager(config_file)
        self.config = self.config_manager.config
        self.state = TradingState.STOPPED
        
        # Initialize components
        self.risk_manager = RiskManager(
            initial_capital=self.config.trading.initial_capital,
            risk_metrics=self._create_risk_metrics()
        )
        
        self.strategy = MoneyPrinterStrategy(
            initial_capital=self.config.trading.initial_capital
        )
        
        self.alert_system = AlertSystem(self.config)
        
        # Trading state
        self.current_data = {}
        self.live_trades: List[LiveTrade] = []
        self.system_metrics: List[SystemMetrics] = []
        self.callbacks: Dict[str, List[Callable]] = {}
        
        # Threading
        self.running = False
        self.trading_thread = None
        
        logger.info("Live Trading Engine initialized")
    
    def _create_risk_metrics(self):
        """Create risk metrics from config"""
        from risk_management import RiskMetrics
        
        return RiskMetrics(
            max_position_size=self.config.trading.max_position_size,
            max_daily_loss=self.config.trading.max_daily_loss,
            max_drawdown=self.config.trading.max_drawdown,
            kelly_multiplier=self.config.trading.kelly_multiplier
        )
    
    def register_callback(self, event: str, callback: Callable):
        """
        Register callback for trading events.
        
        Args:
            event: Event name ('trade_executed', 'position_opened', 'alert_sent', etc.)
            callback: Callback function
        """
        if event not in self.callbacks:
            self.callbacks[event] = []
        self.callbacks[event].append(callback)
    
    def _trigger_callbacks(self, event: str, data: any):
        """Trigger all callbacks for an event"""
        if event in self.callbacks:
            for callback in self.callbacks[event]:
                try:
                    callback(data)
                except Exception as e:
                    logger.error(f"Error in callback for {event}: {e}")
    
    def update_market_data(self, symbol: str, price_data: Dict):
        """
        Update market data for a symbol.
        
        Args:
            symbol: Trading symbol
            price_data: Dictionary with OHLCV data
        """
        self.current_data[symbol] = {
            **price_data,
            'timestamp': datetime.now()
        }
        
        # Update positions with current prices
        if symbol in [pos.symbol for pos in self.risk_manager.positions.values()]:
            market_prices = {symbol: price_data.get('close', price_data.get('price', 0))}
            self.risk_manager.update_positions(market_prices)
    
    def execute_trade(self, symbol: str, action: str, quantity: float, price: float, confidence: float) -> bool:
        """
        Execute a trade.
        
        Args:
            symbol: Trading symbol
            action: Trade action ('BUY', 'SELL')
            quantity: Quantity to trade
            price: Execution price
            confidence: Model confidence
        
        Returns:
            True if trade executed successfully
        """
        start_time = time.time()
        
        try:
            # Create trade record
            trade = LiveTrade(
                timestamp=datetime.now(),
                symbol=symbol,
                action=action,
                quantity=quantity,
                price=price,
                confidence=confidence,
                reason="ML Signal",
                execution_time=0.0,
                status="PENDING"
            )
            
            # Execute through risk manager
            success = False
            if action == "BUY":
                success = self.risk_manager.open_position(
                    symbol=symbol,
                    size=quantity,
                    price=price,
                    side='long'
                )
            elif action == "SELL":
                if symbol in self.risk_manager.positions:
                    success = self.risk_manager.close_position(symbol, price, "ML Signal")
                else:
                    success = self.risk_manager.open_position(
                        symbol=symbol,
                        size=quantity,
                        price=price,
                        side='short'
                    )
            
            # Update trade record
            trade.execution_time = time.time() - start_time
            trade.status = "EXECUTED" if success else "FAILED"
            
            self.live_trades.append(trade)
            
            # Trigger callbacks
            self._trigger_callbacks('trade_executed', trade)
            
            # Send alert for large trades
            if abs(quantity * price) > self.config.trading.initial_capital * 0.05:  # > 5% of capital
                self.alert_system.send_alert(
                    f"Large trade executed: {action} {quantity} {symbol} at ${price:.2f}",
                    level="WARNING",
                    data=asdict(trade)
                )
            
            logger.info(f"Trade {'executed' if success else 'failed'}: {action} {quantity} {symbol} at ${price:.2f}")
            
            return success
            
        except Exception as e:
            logger.error(f"Error executing trade: {e}")
            return False
    
    def process_market_data(self, symbol: str) -> Optional[str]:
        """
        Process market data and generate trading signal.
        
        Args:
            symbol: Trading symbol
        
        Returns:
            Trading signal ('BUY', 'SELL', 'HOLD') or None
        """
        try:
            if symbol not in self.current_data:
                return None
            
            # For demo purposes, generate simple signals based on price movement
            # In production, this would use the full ML strategy
            
            data = self.current_data[symbol]
            current_price = data.get('close', data.get('price', 0))
            
            # Simple momentum-based signal (placeholder)
            if 'prev_price' in data:
                price_change = (current_price - data['prev_price']) / data['prev_price']
                
                if price_change > 0.01:  # 1% increase
                    return "BUY"
                elif price_change < -0.01:  # 1% decrease
                    return "SELL"
            
            # Update previous price
            self.current_data[symbol]['prev_price'] = current_price
            
            return "HOLD"
            
        except Exception as e:
            logger.error(f"Error processing market data for {symbol}: {e}")
            return None
    
    def update_system_metrics(self):
        """Update system performance metrics"""
        try:
            portfolio_summary = self.risk_manager.get_portfolio_summary()
            
            metrics = SystemMetrics(
                timestamp=datetime.now(),
                portfolio_value=portfolio_summary['portfolio_value'],
                total_pnl=portfolio_summary['total_pnl'],
                daily_pnl=0.0,  # Would calculate from today's trades
                drawdown=portfolio_summary['current_drawdown'],
                positions_count=portfolio_summary['open_positions'],
                model_confidence=0.68,  # Placeholder
                execution_latency=0.025,  # Placeholder
                system_health=0.95  # Placeholder
            )
            
            self.system_metrics.append(metrics)
            
            # Keep only last 1000 metrics
            if len(self.system_metrics) > 1000:
                self.system_metrics = self.system_metrics[-1000:]
            
            # Check for alerts
            self._check_risk_alerts(portfolio_summary)
            
        except Exception as e:
            logger.error(f"Error updating system metrics: {e}")
    
    def _check_risk_alerts(self, portfolio_summary: Dict):
        """Check for risk-based alerts"""
        # Large drawdown alert
        if portfolio_summary['current_drawdown'] > self.config.alerts.alert_on_large_drawdown:
            self.alert_system.send_alert(
                f"Large drawdown detected: {portfolio_summary['current_drawdown']:.2%}",
                level="WARNING",
                data=portfolio_summary
            )
        
        # High profit alert
        if portfolio_summary['total_return'] > self.config.alerts.alert_on_high_profit:
            self.alert_system.send_alert(
                f"High profit achieved: {portfolio_summary['total_return']:.2%}",
                level="INFO",
                data=portfolio_summary
            )
    
    def start_trading(self, symbols: List[str] = None):
        """
        Start live trading.
        
        Args:
            symbols: List of symbols to trade (default: ['SYNTH'])
        """
        if self.state != TradingState.STOPPED:
            logger.warning("Trading is already running or in transition")
            return
        
        if symbols is None:
            symbols = ['SYNTH']  # Default synthetic symbol
        
        self.state = TradingState.STARTING
        self.running = True
        
        logger.info(f"Starting live trading for symbols: {symbols}")
        
        # Start trading thread
        self.trading_thread = threading.Thread(
            target=self._trading_loop,
            args=(symbols,),
            daemon=True
        )
        self.trading_thread.start()
        
        self.state = TradingState.RUNNING
        
        self.alert_system.send_alert(
            f"MoneyPrinter trading started for {len(symbols)} symbols",
            level="INFO",
            data={'symbols': symbols}
        )
    
    def stop_trading(self):
        """Stop live trading"""
        logger.info("Stopping live trading...")
        
        self.running = False
        self.state = TradingState.STOPPED
        
        # Close all positions
        for symbol in list(self.risk_manager.positions.keys()):
            if symbol in self.current_data:
                current_price = self.current_data[symbol].get('close', 0)
                self.risk_manager.close_position(symbol, current_price, "System Shutdown")
        
        self.alert_system.send_alert(
            "MoneyPrinter trading stopped",
            level="INFO"
        )
    
    def emergency_stop(self):
        """Emergency stop - immediately halt all trading"""
        logger.critical("EMERGENCY STOP ACTIVATED!")
        
        self.state = TradingState.EMERGENCY_STOP
        self.running = False
        
        # Close all positions at market prices
        for symbol in list(self.risk_manager.positions.keys()):
            if symbol in self.current_data:
                current_price = self.current_data[symbol].get('close', 0)
                self.risk_manager.close_position(symbol, current_price, "Emergency Stop")
        
        self.alert_system.send_alert(
            "EMERGENCY STOP ACTIVATED - All positions closed",
            level="CRITICAL"
        )
    
    def _trading_loop(self, symbols: List[str]):
        """Main trading loop"""
        logger.info("Trading loop started")
        
        try:
            while self.running:
                start_time = time.time()
                
                # Simulate market data updates
                for symbol in symbols:
                    self._simulate_market_data(symbol)
                
                # Process each symbol
                for symbol in symbols:
                    signal = self.process_market_data(symbol)
                    
                    if signal and signal != "HOLD":
                        current_price = self.current_data[symbol].get('close', 100.0)
                        quantity = 100  # Placeholder quantity calculation
                        confidence = 0.7  # Placeholder confidence
                        
                        self.execute_trade(symbol, signal, quantity, current_price, confidence)
                
                # Update system metrics
                self.update_system_metrics()
                
                # Sleep to maintain loop frequency
                elapsed = time.time() - start_time
                sleep_time = max(0, 1.0 - elapsed)  # 1 second loop
                time.sleep(sleep_time)
                
        except Exception as e:
            logger.error(f"Error in trading loop: {e}")
            self.state = TradingState.ERROR
            self.alert_system.send_alert(f"Trading loop error: {e}", level="ERROR")
        
        logger.info("Trading loop ended")
    
    def _simulate_market_data(self, symbol: str):
        """Simulate market data for demo purposes"""
        if symbol not in self.current_data:
            # Initialize with base price
            self.current_data[symbol] = {
                'close': 100.0,
                'open': 100.0,
                'high': 100.0,
                'low': 100.0,
                'volume': 1000,
                'timestamp': datetime.now()
            }
        
        # Simulate price movement
        current_price = self.current_data[symbol]['close']
        price_change = np.random.normal(0, 0.001)  # 0.1% volatility
        new_price = current_price * (1 + price_change)
        
        # Update OHLCV data
        self.current_data[symbol].update({
            'open': self.current_data[symbol]['close'],  # Previous close as new open
            'close': new_price,
            'high': max(self.current_data[symbol]['close'], new_price),
            'low': min(self.current_data[symbol]['close'], new_price),
            'volume': np.random.randint(500, 2000),
            'timestamp': datetime.now()
        })
    
    def get_status(self) -> Dict:
        """Get current system status"""
        latest_metrics = self.system_metrics[-1] if self.system_metrics else None
        
        return {
            'state': self.state.value,
            'running': self.running,
            'symbols_tracked': list(self.current_data.keys()),
            'active_positions': len(self.risk_manager.positions),
            'total_trades': len(self.live_trades),
            'latest_metrics': asdict(latest_metrics) if latest_metrics else None,
            'portfolio_summary': self.risk_manager.get_portfolio_summary(),
            'recent_trades': [asdict(trade) for trade in self.live_trades[-10:]],  # Last 10 trades
            'alerts_count': len(self.alert_system.alerts_sent)
        }

def demo_live_trading():
    """Demo the live trading system"""
    print("🚀 MoneyPrinter Live Trading Demo")
    print("=" * 50)
    
    # Initialize trading engine
    engine = LiveTradingEngine()
    
    # Register some callbacks
    def on_trade_executed(trade):
        print(f"📈 Trade: {trade.action} {trade.quantity} {trade.symbol} at ${trade.price:.2f}")
    
    engine.register_callback('trade_executed', on_trade_executed)
    
    # Start trading
    engine.start_trading(['SYNTH'])
    
    try:
        # Run for 30 seconds
        for i in range(30):
            time.sleep(1)
            if i % 5 == 0:  # Print status every 5 seconds
                status = engine.get_status()
                print(f"📊 Status: {status['state']} | Positions: {status['active_positions']} | Trades: {status['total_trades']}")
        
    except KeyboardInterrupt:
        print("\n⏹️  Manual stop requested")
    
    finally:
        # Stop trading
        engine.stop_trading()
        
        # Print final status
        final_status = engine.get_status()
        print(f"\n🏁 Final Results:")
        print(f"  Total Trades: {final_status['total_trades']}")
        print(f"  Portfolio Value: ${final_status['portfolio_summary']['portfolio_value']:,.2f}")
        print(f"  Total P&L: ${final_status['portfolio_summary']['total_pnl']:,.2f}")
        print(f"  Alerts Sent: {final_status['alerts_count']}")

if __name__ == '__main__':
    demo_live_trading()