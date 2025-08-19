#!/usr/bin/env python3
"""
Configuration Management System for MoneyPrinter
Handles all settings, parameters, and environment configurations.
"""

import yaml
import json
import os
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Any
import logging

@dataclass
class TradingConfig:
    """Trading strategy configuration"""
    initial_capital: float = 100000.0
    max_position_size: float = 0.1  # 10% of portfolio
    max_daily_loss: float = 0.02    # 2% daily loss limit
    max_drawdown: float = 0.1       # 10% max drawdown
    kelly_multiplier: float = 0.25  # Conservative Kelly
    risk_free_rate: float = 0.02    # 2% risk-free rate
    transaction_cost: float = 0.001 # 0.1% transaction cost
    slippage: float = 0.0005        # 0.05% slippage

@dataclass
class ModelConfig:
    """ML model configuration"""
    ensemble_models: List[str] = None
    feature_selection_threshold: float = 0.01
    validation_split: float = 0.2
    cross_validation_folds: int = 5
    hyperparameter_optimization: bool = True
    model_retrain_frequency: int = 24  # hours
    prediction_threshold: float = 0.6

    def __post_init__(self):
        if self.ensemble_models is None:
            self.ensemble_models = ['rf', 'xgb', 'lgb', 'gb', 'svm', 'lr']

@dataclass
class DataConfig:
    """Data source configuration"""
    primary_data_source: str = 'synthetic'  # 'yfinance', 'alpha_vantage', 'synthetic'
    backup_data_source: str = 'synthetic'
    data_update_frequency: int = 1  # hours
    lookback_period: int = 365  # days
    data_validation: bool = True
    min_data_quality_score: float = 0.8

@dataclass
class AlertConfig:
    """Alert and notification configuration"""
    enable_email_alerts: bool = False
    enable_webhook_alerts: bool = False
    email_recipients: List[str] = None
    webhook_url: Optional[str] = None
    alert_on_large_drawdown: float = 0.05  # 5%
    alert_on_high_profit: float = 0.1      # 10%
    alert_on_model_degradation: float = 0.1  # 10% accuracy drop

    def __post_init__(self):
        if self.email_recipients is None:
            self.email_recipients = []

@dataclass
class MoneyPrinterConfig:
    """Complete MoneyPrinter configuration"""
    trading: TradingConfig = None
    model: ModelConfig = None
    data: DataConfig = None
    alerts: AlertConfig = None
    
    # Environment settings
    environment: str = 'development'  # 'development', 'testing', 'production'
    log_level: str = 'INFO'
    save_results: bool = True
    results_directory: str = 'results'
    
    def __post_init__(self):
        if self.trading is None:
            self.trading = TradingConfig()
        if self.model is None:
            self.model = ModelConfig()
        if self.data is None:
            self.data = DataConfig()
        if self.alerts is None:
            self.alerts = AlertConfig()

class ConfigManager:
    """
    Manages configuration loading, saving, and validation.
    """
    
    def __init__(self, config_file: str = 'config/moneyprinter_config.yaml'):
        """
        Initialize config manager.
        
        Args:
            config_file: Path to configuration file
        """
        self.config_file = config_file
        self.config = MoneyPrinterConfig()
        self.logger = logging.getLogger(__name__)
        
        # Create config directory if it doesn't exist
        os.makedirs(os.path.dirname(config_file), exist_ok=True)
        
        # Load existing config or create default
        if os.path.exists(config_file):
            self.load_config()
        else:
            self.save_config()  # Save default config
    
    def load_config(self) -> MoneyPrinterConfig:
        """
        Load configuration from file.
        
        Returns:
            Loaded configuration object
        """
        try:
            with open(self.config_file, 'r') as f:
                config_dict = yaml.safe_load(f)
            
            # Convert dict to config objects
            self.config = self._dict_to_config(config_dict)
            self.logger.info(f"Configuration loaded from {self.config_file}")
            
        except Exception as e:
            self.logger.error(f"Error loading config: {e}")
            self.logger.info("Using default configuration")
            self.config = MoneyPrinterConfig()
        
        return self.config
    
    def save_config(self):
        """Save current configuration to file."""
        try:
            config_dict = self._config_to_dict()
            
            with open(self.config_file, 'w') as f:
                yaml.dump(config_dict, f, default_flow_style=False, indent=2)
            
            self.logger.info(f"Configuration saved to {self.config_file}")
            
        except Exception as e:
            self.logger.error(f"Error saving config: {e}")
    
    def update_config(self, **kwargs):
        """
        Update configuration values.
        
        Args:
            **kwargs: Configuration values to update
        """
        try:
            for key, value in kwargs.items():
                if hasattr(self.config, key):
                    setattr(self.config, key, value)
                elif '.' in key:
                    # Handle nested attributes like 'trading.initial_capital'
                    parts = key.split('.')
                    obj = self.config
                    for part in parts[:-1]:
                        obj = getattr(obj, part)
                    setattr(obj, parts[-1], value)
                else:
                    self.logger.warning(f"Unknown config key: {key}")
            
            self.save_config()
            self.logger.info(f"Configuration updated: {kwargs}")
            
        except Exception as e:
            self.logger.error(f"Error updating config: {e}")
    
    def get_environment_config(self) -> Dict[str, Any]:
        """
        Get environment-specific configuration.
        
        Returns:
            Dictionary of environment variables and settings
        """
        env_config = {
            'ENVIRONMENT': self.config.environment,
            'LOG_LEVEL': self.config.log_level,
            'ALPHA_VANTAGE_KEY': os.getenv('ALPHA_VANTAGE_KEY', ''),
            'EMAIL_USERNAME': os.getenv('EMAIL_USERNAME', ''),
            'EMAIL_PASSWORD': os.getenv('EMAIL_PASSWORD', ''),
            'WEBHOOK_SECRET': os.getenv('WEBHOOK_SECRET', ''),
        }
        
        return env_config
    
    def validate_config(self) -> List[str]:
        """
        Validate configuration and return list of issues.
        
        Returns:
            List of validation error messages
        """
        issues = []
        
        # Trading config validation
        if self.config.trading.initial_capital <= 0:
            issues.append("Initial capital must be positive")
        
        if self.config.trading.max_position_size <= 0 or self.config.trading.max_position_size > 1:
            issues.append("Max position size must be between 0 and 1")
        
        if self.config.trading.kelly_multiplier <= 0 or self.config.trading.kelly_multiplier > 1:
            issues.append("Kelly multiplier should be between 0 and 1")
        
        # Model config validation
        if not self.config.model.ensemble_models:
            issues.append("At least one model must be specified")
        
        if self.config.model.validation_split <= 0 or self.config.model.validation_split >= 1:
            issues.append("Validation split must be between 0 and 1")
        
        # Data config validation
        if self.config.data.lookback_period <= 0:
            issues.append("Lookback period must be positive")
        
        # Alert config validation
        if self.config.alerts.enable_email_alerts and not self.config.alerts.email_recipients:
            issues.append("Email alerts enabled but no recipients specified")
        
        if self.config.alerts.enable_webhook_alerts and not self.config.alerts.webhook_url:
            issues.append("Webhook alerts enabled but no URL specified")
        
        return issues
    
    def _dict_to_config(self, config_dict: Dict) -> MoneyPrinterConfig:
        """Convert dictionary to config object."""
        trading_config = TradingConfig(**config_dict.get('trading', {}))
        model_config = ModelConfig(**config_dict.get('model', {}))
        data_config = DataConfig(**config_dict.get('data', {}))
        alerts_config = AlertConfig(**config_dict.get('alerts', {}))
        
        main_config = {k: v for k, v in config_dict.items() 
                      if k not in ['trading', 'model', 'data', 'alerts']}
        
        return MoneyPrinterConfig(
            trading=trading_config,
            model=model_config,
            data=data_config,
            alerts=alerts_config,
            **main_config
        )
    
    def _config_to_dict(self) -> Dict:
        """Convert config object to dictionary."""
        return {
            'trading': asdict(self.config.trading),
            'model': asdict(self.config.model),
            'data': asdict(self.config.data),
            'alerts': asdict(self.config.alerts),
            'environment': self.config.environment,
            'log_level': self.config.log_level,
            'save_results': self.config.save_results,
            'results_directory': self.config.results_directory,
        }

def create_sample_configs():
    """Create sample configurations for different environments."""
    
    # Development config
    dev_config = MoneyPrinterConfig(
        environment='development',
        log_level='DEBUG',
        trading=TradingConfig(
            initial_capital=10000.0,
            max_position_size=0.05,  # More conservative
            kelly_multiplier=0.1     # Very conservative
        ),
        model=ModelConfig(
            hyperparameter_optimization=False,  # Faster training
            cross_validation_folds=3           # Fewer folds
        ),
        data=DataConfig(
            primary_data_source='synthetic',
            lookback_period=90  # Shorter period
        )
    )
    
    # Production config
    prod_config = MoneyPrinterConfig(
        environment='production',
        log_level='INFO',
        trading=TradingConfig(
            initial_capital=100000.0,
            max_position_size=0.1,
            kelly_multiplier=0.25
        ),
        model=ModelConfig(
            hyperparameter_optimization=True,
            cross_validation_folds=5
        ),
        data=DataConfig(
            primary_data_source='yfinance',
            backup_data_source='alpha_vantage',
            lookback_period=365
        ),
        alerts=AlertConfig(
            enable_email_alerts=True,
            enable_webhook_alerts=True,
            email_recipients=['trader@example.com']
        )
    )
    
    # Save sample configs
    os.makedirs('config', exist_ok=True)
    
    with open('config/development.yaml', 'w') as f:
        yaml.dump(asdict(dev_config), f, default_flow_style=False, indent=2)
    
    with open('config/production.yaml', 'w') as f:
        yaml.dump(asdict(prod_config), f, default_flow_style=False, indent=2)

if __name__ == '__main__':
    # Demo the configuration system
    print("🔧 MoneyPrinter Configuration System")
    print("=" * 50)
    
    # Create config manager
    config_manager = ConfigManager()
    
    # Display current config
    print("\n📋 Current Configuration:")
    print(f"Environment: {config_manager.config.environment}")
    print(f"Initial Capital: ${config_manager.config.trading.initial_capital:,.2f}")
    print(f"Max Position Size: {config_manager.config.trading.max_position_size:.1%}")
    print(f"Models: {', '.join(config_manager.config.model.ensemble_models)}")
    print(f"Data Source: {config_manager.config.data.primary_data_source}")
    
    # Validate config
    issues = config_manager.validate_config()
    if issues:
        print(f"\n⚠️ Configuration Issues:")
        for issue in issues:
            print(f"  - {issue}")
    else:
        print(f"\n✅ Configuration is valid!")
    
    # Update some settings
    print(f"\n🔄 Updating configuration...")
    config_manager.update_config(**{
        'trading.initial_capital': 50000.0,
        'environment': 'testing'
    })
    
    # Create sample configs
    create_sample_configs()
    print(f"\n📁 Sample configurations created in config/ directory")
    
    print(f"\n🎯 Configuration system ready!")