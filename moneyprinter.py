#!/usr/bin/env python3
"""
MoneyPrinter - The Ultimate AI-Powered Trading System

A comprehensive, advanced, ML-based trading strategy that combines:
- Multiple ML models in an ensemble
- Advanced feature engineering
- Risk management with Kelly criterion
- Real-time trading capabilities
- Web dashboard for monitoring
- Configuration management
- Alert system

Usage:
    python moneyprinter.py --mode backtest  # Run backtest
    python moneyprinter.py --mode live      # Start live trading
    python moneyprinter.py --mode dashboard # Launch web dashboard
    python moneyprinter.py --mode optimize  # Hyperparameter optimization
"""

import argparse
import sys
import os
import logging
from datetime import datetime
import signal
import threading

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from config_manager import ConfigManager
from moneyprinter_strategy import MoneyPrinterStrategy
from live_trading import LiveTradingEngine
from optimize_hyperparameters import objective
import optuna

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('moneyprinter.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class MoneyPrinter:
    """
    Main MoneyPrinter application class.
    """
    
    def __init__(self, config_file: str = 'config/moneyprinter_config.yaml'):
        """
        Initialize MoneyPrinter.
        
        Args:
            config_file: Configuration file path
        """
        self.config_manager = ConfigManager(config_file)
        self.config = self.config_manager.config
        
        # Initialize components
        self.strategy = None
        self.live_engine = None
        
        logger.info("💰 MoneyPrinter initialized")
        
        # Setup signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals"""
        logger.info("Shutdown signal received, cleaning up...")
        if self.live_engine:
            self.live_engine.stop_trading()
        sys.exit(0)
    
    def run_backtest(self, data_path: str = 'data/market_data.csv'):
        """
        Run backtesting mode.
        
        Args:
            data_path: Path to market data
        """
        logger.info("🎯 Running MoneyPrinter Backtest")
        
        try:
            # Initialize strategy
            self.strategy = MoneyPrinterStrategy(
                initial_capital=self.config.trading.initial_capital
            )
            
            # Run complete strategy
            results = self.strategy.run_full_strategy(data_path)
            
            # Save results if configured
            if self.config.save_results:
                self._save_results(results, 'backtest')
            
            return results
            
        except Exception as e:
            logger.error(f"Error in backtest: {e}")
            raise
    
    def run_live_trading(self, symbols: list = None):
        """
        Run live trading mode.
        
        Args:
            symbols: List of symbols to trade
        """
        logger.info("🚀 Starting MoneyPrinter Live Trading")
        
        try:
            # Initialize live trading engine
            self.live_engine = LiveTradingEngine(self.config_manager.config_file)
            
            # Start trading
            self.live_engine.start_trading(symbols or ['SYNTH'])
            
            # Keep running until interrupted
            while self.live_engine.running:
                status = self.live_engine.get_status()
                
                if status['state'] == 'error':
                    logger.error("Trading engine encountered an error, stopping...")
                    break
                
                # Print status every minute
                logger.info(f"Trading Status: {status['state']} | "
                          f"Positions: {status['active_positions']} | "
                          f"Trades: {status['total_trades']} | "
                          f"P&L: ${status['portfolio_summary']['total_pnl']:,.2f}")
                
                # Sleep for 60 seconds
                import time
                time.sleep(60)
                
        except KeyboardInterrupt:
            logger.info("Live trading interrupted by user")
        except Exception as e:
            logger.error(f"Error in live trading: {e}")
            raise
        finally:
            if self.live_engine:
                self.live_engine.stop_trading()
    
    def launch_dashboard(self, port: int = 8501):
        """
        Launch web dashboard.
        
        Args:
            port: Port to run dashboard on
        """
        logger.info(f"🌐 Launching MoneyPrinter Dashboard on port {port}")
        
        try:
            import subprocess
            subprocess.run([
                'streamlit', 'run', 'dashboard.py',
                '--server.port', str(port),
                '--server.headless', 'true'
            ])
        except Exception as e:
            logger.error(f"Error launching dashboard: {e}")
            logger.info("Make sure Streamlit is installed: pip install streamlit")
    
    def optimize_hyperparameters(self, n_trials: int = 50):
        """
        Run hyperparameter optimization.
        
        Args:
            n_trials: Number of optimization trials
        """
        logger.info(f"🎛️ Starting Hyperparameter Optimization ({n_trials} trials)")
        
        try:
            # Create optimization study
            study = optuna.create_study(direction='maximize')
            study.optimize(objective, n_trials=n_trials)
            
            # Print results
            logger.info(f"Best trial: {study.best_trial.value}")
            logger.info("Best parameters:")
            for key, value in study.best_trial.params.items():
                logger.info(f"  {key}: {value}")
            
            # Save best parameters to config
            self._update_config_with_best_params(study.best_trial.params)
            
            return study.best_trial
            
        except Exception as e:
            logger.error(f"Error in optimization: {e}")
            raise
    
    def _update_config_with_best_params(self, params: dict):
        """Update configuration with best parameters"""
        try:
            # Update relevant config parameters
            config_updates = {}
            
            # Map optimization parameters to config
            param_mapping = {
                'frac_diff_d': 'model.frac_diff_d',
                'ae_window_size': 'model.ae_window_size',
                'ae_encoding_dim': 'model.ae_encoding_dim',
                'transformer_lr': 'model.transformer_lr'
                # Add more mappings as needed
            }
            
            for param, config_key in param_mapping.items():
                if param in params:
                    config_updates[config_key] = params[param]
            
            if config_updates:
                self.config_manager.update_config(**config_updates)
                logger.info(f"Updated configuration with best parameters: {config_updates}")
            
        except Exception as e:
            logger.error(f"Error updating config with best parameters: {e}")
    
    def _save_results(self, results: dict, mode: str):
        """Save results to file"""
        try:
            os.makedirs(self.config.results_directory, exist_ok=True)
            
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"{self.config.results_directory}/moneyprinter_{mode}_{timestamp}.json"
            
            import json
            with open(filename, 'w') as f:
                # Convert any numpy types to native Python types for JSON serialization
                json_results = self._convert_to_json_serializable(results)
                json.dump(json_results, f, indent=2, default=str)
            
            logger.info(f"Results saved to {filename}")
            
        except Exception as e:
            logger.error(f"Error saving results: {e}")
    
    def _convert_to_json_serializable(self, obj):
        """Convert object to JSON serializable format"""
        import numpy as np
        
        if isinstance(obj, dict):
            return {k: self._convert_to_json_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._convert_to_json_serializable(item) for item in obj]
        elif isinstance(obj, (np.integer, np.int64)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float64)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return obj
    
    def run_full_analysis(self):
        """Run complete analysis including backtest, optimization, and reporting"""
        logger.info("🔬 Running Full MoneyPrinter Analysis")
        
        try:
            results = {}
            
            # 1. Run backtest with current config
            logger.info("Step 1/3: Running initial backtest...")
            results['initial_backtest'] = self.run_backtest()
            
            # 2. Run hyperparameter optimization
            logger.info("Step 2/3: Optimizing hyperparameters...")
            best_trial = self.optimize_hyperparameters(n_trials=20)
            results['optimization'] = {
                'best_value': best_trial.value,
                'best_params': best_trial.params
            }
            
            # 3. Run backtest with optimized parameters
            logger.info("Step 3/3: Running optimized backtest...")
            results['optimized_backtest'] = self.run_backtest()
            
            # Compare results
            initial_return = float(results['initial_backtest']['backtest_results']['Total Return'].rstrip('%'))
            optimized_return = float(results['optimized_backtest']['backtest_results']['Total Return'].rstrip('%'))
            improvement = optimized_return - initial_return
            
            logger.info(f"📊 Analysis Complete!")
            logger.info(f"Initial Return: {initial_return:.2f}%")
            logger.info(f"Optimized Return: {optimized_return:.2f}%")
            logger.info(f"Improvement: {improvement:+.2f}%")
            
            # Save complete analysis
            if self.config.save_results:
                self._save_results(results, 'full_analysis')
            
            return results
            
        except Exception as e:
            logger.error(f"Error in full analysis: {e}")
            raise

def main():
    """Main application entry point"""
    parser = argparse.ArgumentParser(
        description='MoneyPrinter - The Ultimate AI-Powered Trading System'
    )
    
    parser.add_argument(
        '--mode', 
        choices=['backtest', 'live', 'dashboard', 'optimize', 'analyze'],
        default='backtest',
        help='Operating mode'
    )
    
    parser.add_argument(
        '--config',
        default='config/moneyprinter_config.yaml',
        help='Configuration file path'
    )
    
    parser.add_argument(
        '--data',
        default='data/market_data.csv',
        help='Market data file path'
    )
    
    parser.add_argument(
        '--symbols',
        nargs='+',
        default=['SYNTH'],
        help='Trading symbols for live mode'
    )
    
    parser.add_argument(
        '--port',
        type=int,
        default=8501,
        help='Dashboard port'
    )
    
    parser.add_argument(
        '--trials',
        type=int,
        default=50,
        help='Number of optimization trials'
    )
    
    args = parser.parse_args()
    
    # Print banner
    print_banner()
    
    try:
        # Initialize MoneyPrinter
        money_printer = MoneyPrinter(args.config)
        
        # Run selected mode
        if args.mode == 'backtest':
            results = money_printer.run_backtest(args.data)
            print_results_summary(results)
            
        elif args.mode == 'live':
            money_printer.run_live_trading(args.symbols)
            
        elif args.mode == 'dashboard':
            money_printer.launch_dashboard(args.port)
            
        elif args.mode == 'optimize':
            best_trial = money_printer.optimize_hyperparameters(args.trials)
            print(f"\n🎯 Best optimization result: {best_trial.value:.4f}")
            
        elif args.mode == 'analyze':
            results = money_printer.run_full_analysis()
            print_analysis_summary(results)
        
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
    except Exception as e:
        logger.error(f"Application error: {e}")
        sys.exit(1)

def print_banner():
    """Print application banner"""
    banner = """
    ███╗   ███╗ ██████╗ ███╗   ██╗███████╗██╗   ██╗██████╗ ██████╗ ██╗███╗   ██╗████████╗███████╗██████╗ 
    ████╗ ████║██╔═══██╗████╗  ██║██╔════╝╚██╗ ██╔╝██╔══██╗██╔══██╗██║████╗  ██║╚══██╔══╝██╔════╝██╔══██╗
    ██╔████╔██║██║   ██║██╔██╗ ██║█████╗   ╚████╔╝ ██████╔╝██████╔╝██║██╔██╗ ██║   ██║   █████╗  ██████╔╝
    ██║╚██╔╝██║██║   ██║██║╚██╗██║██╔══╝    ╚██╔╝  ██╔═══╝ ██╔══██╗██║██║╚██╗██║   ██║   ██╔══╝  ██╔══██╗
    ██║ ╚═╝ ██║╚██████╔╝██║ ╚████║███████╗   ██║   ██║     ██║  ██║██║██║ ╚████║   ██║   ███████╗██║  ██║
    ╚═╝     ╚═╝ ╚═════╝ ╚═╝  ╚═══╝╚══════╝   ╚═╝   ╚═╝     ╚═╝  ╚═╝╚═╝╚═╝  ╚═══╝   ╚═╝   ╚══════╝╚═╝  ╚═╝
    
    🚀 The Ultimate AI-Powered Trading System 🚀
    💰 Advanced ML | Risk Management | Live Trading | Real-time Dashboard 💰
    """
    print(banner)

def print_results_summary(results: dict):
    """Print backtest results summary"""
    print("\n" + "="*80)
    print("🏆 MONEYPRINTER BACKTEST RESULTS 🏆")
    print("="*80)
    
    # Model performance
    print("\n📊 Model Performance:")
    for model, score in results['model_scores'].items():
        print(f"  {model.upper()}: {score:.4f}")
    
    # Key metrics
    metrics = results['backtest_results']
    print(f"\n📈 Key Metrics:")
    print(f"  Total Return: {metrics['Total Return']}")
    print(f"  Sharpe Ratio: {metrics['Sharpe Ratio']}")
    print(f"  Max Drawdown: {metrics['Max Drawdown']}")
    print(f"  Win Rate: {metrics['Win Rate']}")
    print(f"  Profit Factor: {metrics['Profit Factor']}")
    
    # Data info
    info = results['data_info']
    print(f"\n📋 Data Summary:")
    print(f"  Total Samples: {info['total_samples']:,}")
    print(f"  Features: {info['features_count']}")
    print(f"  Buy Signals: {info['positive_signals']:,}")
    print(f"  Sell Signals: {info['negative_signals']:,}")
    
    print("="*80)

def print_analysis_summary(results: dict):
    """Print full analysis summary"""
    print("\n" + "="*80)
    print("🔬 MONEYPRINTER FULL ANALYSIS RESULTS 🔬")
    print("="*80)
    
    initial_return = float(results['initial_backtest']['backtest_results']['Total Return'].rstrip('%'))
    optimized_return = float(results['optimized_backtest']['backtest_results']['Total Return'].rstrip('%'))
    improvement = optimized_return - initial_return
    
    print(f"\n📊 Performance Comparison:")
    print(f"  Initial Return: {initial_return:,.2f}%")
    print(f"  Optimized Return: {optimized_return:,.2f}%")
    print(f"  Improvement: {improvement:+,.2f}%")
    
    print(f"\n🎯 Best Parameters:")
    for param, value in results['optimization']['best_params'].items():
        print(f"  {param}: {value}")
    
    print("="*80)

if __name__ == '__main__':
    main()