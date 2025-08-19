#!/usr/bin/env python3
"""
Advanced Portfolio Optimization and Multi-Asset Trading
Implements Modern Portfolio Theory, risk parity, and advanced optimization techniques.
"""

import numpy as np
import pandas as pd
from scipy.optimize import minimize, differential_evolution
from scipy import linalg
import logging
from typing import Dict, List, Tuple, Optional, Any
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

class AdvancedPortfolioOptimizer:
    """
    Advanced portfolio optimization using multiple methodologies.
    """
    
    def __init__(self, lookback_period=252, rebalance_frequency=20):
        self.lookback_period = lookback_period
        self.rebalance_frequency = rebalance_frequency
        self.covariance_estimators = {
            'sample': self._sample_covariance,
            'shrinkage': self._shrinkage_covariance,
            'robust': self._robust_covariance
        }
        
    def calculate_returns(self, prices: pd.DataFrame) -> pd.DataFrame:
        """Calculate returns from price data."""
        return prices.pct_change().dropna()
    
    def _sample_covariance(self, returns: pd.DataFrame) -> np.ndarray:
        """Calculate sample covariance matrix."""
        return returns.cov().values
    
    def _shrinkage_covariance(self, returns: pd.DataFrame, shrinkage=0.2) -> np.ndarray:
        """Calculate shrinkage covariance matrix (Ledoit-Wolf)."""
        sample_cov = self._sample_covariance(returns)
        
        # Target matrix (diagonal with average variance)
        avg_var = np.mean(np.diag(sample_cov))
        target = np.eye(len(sample_cov)) * avg_var
        
        # Shrinkage estimator
        shrunk_cov = (1 - shrinkage) * sample_cov + shrinkage * target
        return shrunk_cov
    
    def _robust_covariance(self, returns: pd.DataFrame) -> np.ndarray:
        """Calculate robust covariance matrix using Minimum Covariance Determinant."""
        from sklearn.covariance import MinCovDet
        
        try:
            robust_cov = MinCovDet(random_state=42).fit(returns.values)
            return robust_cov.covariance_
        except:
            # Fallback to shrinkage if robust estimation fails
            return self._shrinkage_covariance(returns)
    
    def mean_variance_optimization(self, expected_returns: np.ndarray, 
                                 cov_matrix: np.ndarray, 
                                 risk_aversion: float = 1.0,
                                 constraints: Dict = None) -> np.ndarray:
        """
        Mean-variance optimization (Markowitz).
        """
        n_assets = len(expected_returns)
        
        # Objective function: minimize -μ'w + λ/2 * w'Σw
        def objective(weights):
            portfolio_return = np.dot(weights, expected_returns)
            portfolio_variance = np.dot(weights, np.dot(cov_matrix, weights))
            return -portfolio_return + risk_aversion * portfolio_variance / 2
        
        # Default constraints
        if constraints is None:
            constraints = []
        
        # Weight sum constraint
        constraints.append({'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0})
        
        # Bounds (no short selling by default)
        bounds = [(0, 1) for _ in range(n_assets)]
        
        # Initial guess
        x0 = np.ones(n_assets) / n_assets
        
        # Optimize
        result = minimize(objective, x0, method='SLSQP', bounds=bounds, constraints=constraints)
        
        if result.success:
            return result.x
        else:
            logger.warning("Mean-variance optimization failed, using equal weights")
            return np.ones(n_assets) / n_assets
    
    def risk_parity_optimization(self, cov_matrix: np.ndarray) -> np.ndarray:
        """
        Risk parity optimization - equal risk contribution.
        """
        n_assets = cov_matrix.shape[0]
        
        def risk_budget_objective(weights):
            """
            Objective function for risk parity.
            Minimizes the sum of squared differences between actual and target risk contributions.
            """
            weights = np.maximum(weights, 1e-8)  # Avoid division by zero
            portfolio_vol = np.sqrt(np.dot(weights, np.dot(cov_matrix, weights)))
            
            # Risk contributions
            marginal_contrib = np.dot(cov_matrix, weights)
            risk_contrib = weights * marginal_contrib / portfolio_vol
            
            # Target: equal risk contribution
            target_risk = portfolio_vol / n_assets
            
            # Sum of squared deviations from target
            return np.sum((risk_contrib - target_risk) ** 2)
        
        # Constraints and bounds
        constraints = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0}]
        bounds = [(0.01, 1.0) for _ in range(n_assets)]  # Minimum 1% allocation
        
        # Initial guess
        x0 = np.ones(n_assets) / n_assets
        
        # Optimize
        result = minimize(risk_budget_objective, x0, method='SLSQP', 
                         bounds=bounds, constraints=constraints)
        
        if result.success:
            return result.x
        else:
            logger.warning("Risk parity optimization failed, using equal weights")
            return np.ones(n_assets) / n_assets
    
    def black_litterman_optimization(self, market_caps: np.ndarray, 
                                   returns: pd.DataFrame,
                                   views: Dict = None,
                                   tau: float = 0.025,
                                   risk_aversion: float = 3.0) -> np.ndarray:
        """
        Black-Litterman optimization with investor views.
        """
        # Market portfolio weights
        w_market = market_caps / np.sum(market_caps)
        
        # Sample covariance
        cov_matrix = self._sample_covariance(returns)
        
        # Implied expected returns (reverse optimization)
        pi = risk_aversion * np.dot(cov_matrix, w_market)
        
        # If no views provided, return market portfolio
        if views is None or len(views) == 0:
            return w_market
        
        # Process views
        P = []  # Picking matrix
        Q = []  # Views vector
        Omega = []  # Uncertainty matrix diagonal
        
        for view in views:
            asset_idx = view.get('asset_idx', 0)
            expected_return = view.get('expected_return', 0.0)
            confidence = view.get('confidence', 0.5)  # 0 to 1
            
            # Picking matrix row (which asset the view is about)
            p_row = np.zeros(len(w_market))
            p_row[asset_idx] = 1.0
            P.append(p_row)
            
            # Expected return from view
            Q.append(expected_return)
            
            # Uncertainty (inverse of confidence)
            omega_val = tau * np.dot(p_row, np.dot(cov_matrix, p_row)) / confidence
            Omega.append(omega_val)
        
        P = np.array(P)
        Q = np.array(Q)
        Omega = np.diag(Omega)
        
        # Black-Litterman formula
        tau_sigma = tau * cov_matrix
        M1 = linalg.inv(tau_sigma)
        M2 = np.dot(P.T, np.dot(linalg.inv(Omega), P))
        M3 = np.dot(linalg.inv(tau_sigma), pi)
        M4 = np.dot(P.T, np.dot(linalg.inv(Omega), Q))
        
        # New expected returns
        mu_bl = np.dot(linalg.inv(M1 + M2), M3 + M4)
        
        # New covariance matrix
        cov_bl = linalg.inv(M1 + M2)
        
        # Optimize with Black-Litterman inputs
        return self.mean_variance_optimization(mu_bl, cov_bl, risk_aversion)
    
    def hierarchical_risk_parity(self, returns: pd.DataFrame) -> np.ndarray:
        """
        Hierarchical Risk Parity (HRP) optimization.
        """
        # Calculate correlation matrix
        corr_matrix = returns.corr().values
        
        # Replace NaN with 0
        corr_matrix = np.nan_to_num(corr_matrix)
        
        # Calculate distance matrix
        distance_matrix = np.sqrt(0.5 * (1 - corr_matrix))
        
        # Hierarchical clustering
        from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
        from scipy.spatial.distance import squareform
        
        # Convert to condensed distance matrix
        condensed_dist = squareform(distance_matrix, checks=False)
        
        # Perform clustering
        linkage_matrix = linkage(condensed_dist, method='ward')
        
        # Get cluster assignments
        n_clusters = min(5, len(returns.columns))  # Max 5 clusters
        clusters = fcluster(linkage_matrix, n_clusters, criterion='maxclust')
        
        # Calculate weights within and between clusters
        n_assets = len(returns.columns)
        weights = np.zeros(n_assets)
        
        # Covariance matrix
        cov_matrix = self._sample_covariance(returns)
        
        for cluster_id in np.unique(clusters):
            cluster_assets = np.where(clusters == cluster_id)[0]
            cluster_size = len(cluster_assets)
            
            if cluster_size == 1:
                # Single asset cluster
                weights[cluster_assets[0]] = 1.0 / n_clusters
            else:
                # Multiple assets in cluster - use risk parity within cluster
                cluster_cov = cov_matrix[np.ix_(cluster_assets, cluster_assets)]
                cluster_weights = self.risk_parity_optimization(cluster_cov)
                
                # Scale by cluster weight
                cluster_total_weight = 1.0 / n_clusters
                for i, asset_idx in enumerate(cluster_assets):
                    weights[asset_idx] = cluster_weights[i] * cluster_total_weight
        
        # Normalize weights
        return weights / np.sum(weights)

class MultiAssetTradingSystem:
    """
    Advanced multi-asset trading system with portfolio optimization.
    """
    
    def __init__(self, assets: List[str], initial_capital: float = 100000):
        self.assets = assets
        self.initial_capital = initial_capital
        self.optimizer = AdvancedPortfolioOptimizer()
        
        # Portfolio state
        self.current_weights = np.ones(len(assets)) / len(assets)
        self.portfolio_value = initial_capital
        self.positions = {asset: 0.0 for asset in assets}
        self.cash = initial_capital
        
        # Performance tracking
        self.portfolio_history = []
        self.weight_history = []
        self.rebalance_dates = []
        
    def calculate_expected_returns(self, returns: pd.DataFrame, 
                                 method: str = 'historical') -> np.ndarray:
        """
        Calculate expected returns using various methods.
        """
        if method == 'historical':
            return returns.mean().values * 252  # Annualized
        
        elif method == 'ewma':
            # Exponentially weighted moving average
            span = 60  # ~3 months
            ewma_returns = returns.ewm(span=span).mean().iloc[-1].values * 252
            return ewma_returns
        
        elif method == 'capm':
            # Simple CAPM-based expected returns
            market_return = 0.10  # Assumed market return
            risk_free_rate = 0.02  # Assumed risk-free rate
            
            # Calculate betas (simplified)
            market_proxy = returns.mean(axis=1)  # Use average as market proxy
            betas = []
            
            for asset in returns.columns:
                asset_returns = returns[asset].dropna()
                if len(asset_returns) > 30:
                    covariance = np.cov(asset_returns, market_proxy)[0, 1]
                    market_variance = np.var(market_proxy)
                    beta = covariance / market_variance if market_variance > 0 else 1.0
                else:
                    beta = 1.0
                betas.append(beta)
            
            betas = np.array(betas)
            expected_returns = risk_free_rate + betas * (market_return - risk_free_rate)
            
            return expected_returns
        
        else:
            # Default to historical
            return self.calculate_expected_returns(returns, 'historical')
    
    def optimize_portfolio(self, price_data: Dict[str, pd.Series], 
                          method: str = 'mean_variance',
                          **kwargs) -> np.ndarray:
        """
        Optimize portfolio weights using specified method.
        """
        # Prepare returns data
        returns_data = {}
        for asset, prices in price_data.items():
            if asset in self.assets:
                returns_data[asset] = prices.pct_change().dropna()
        
        returns_df = pd.DataFrame(returns_data)
        
        if len(returns_df) < 30:  # Not enough data
            logger.warning("Insufficient data for optimization, using equal weights")
            return np.ones(len(self.assets)) / len(self.assets)
        
        try:
            if method == 'mean_variance':
                expected_returns = self.calculate_expected_returns(returns_df)
                cov_matrix = self.optimizer._sample_covariance(returns_df)
                risk_aversion = kwargs.get('risk_aversion', 1.0)
                
                return self.optimizer.mean_variance_optimization(
                    expected_returns, cov_matrix, risk_aversion
                )
            
            elif method == 'risk_parity':
                cov_matrix = self.optimizer._sample_covariance(returns_df)
                return self.optimizer.risk_parity_optimization(cov_matrix)
            
            elif method == 'hierarchical_risk_parity':
                return self.optimizer.hierarchical_risk_parity(returns_df)
            
            elif method == 'black_litterman':
                # Use market caps (simplified as equal for now)
                market_caps = np.ones(len(self.assets))
                views = kwargs.get('views', [])
                
                return self.optimizer.black_litterman_optimization(
                    market_caps, returns_df, views
                )
            
            else:
                logger.warning(f"Unknown optimization method: {method}")
                return np.ones(len(self.assets)) / len(self.assets)
                
        except Exception as e:
            logger.error(f"Portfolio optimization failed: {e}")
            return np.ones(len(self.assets)) / len(self.assets)
    
    def rebalance_portfolio(self, current_prices: Dict[str, float],
                          target_weights: np.ndarray,
                          transaction_cost: float = 0.001) -> Dict[str, Any]:
        """
        Rebalance portfolio to target weights.
        """
        rebalance_info = {
            'trades': {},
            'total_cost': 0.0,
            'weight_changes': {}
        }
        
        # Calculate current portfolio value
        portfolio_value = self.cash
        for asset in self.assets:
            if asset in current_prices:
                portfolio_value += self.positions[asset] * current_prices[asset]
        
        self.portfolio_value = portfolio_value
        
        # Calculate target positions
        target_positions = {}
        for i, asset in enumerate(self.assets):
            if asset in current_prices:
                target_value = portfolio_value * target_weights[i]
                target_positions[asset] = target_value / current_prices[asset]
            else:
                target_positions[asset] = 0
        
        # Execute trades
        total_transaction_cost = 0
        
        for asset in self.assets:
            current_position = self.positions[asset]
            target_position = target_positions[asset]
            trade_quantity = target_position - current_position
            
            if abs(trade_quantity) > 0.001:  # Minimum trade threshold
                if asset in current_prices:
                    trade_value = abs(trade_quantity * current_prices[asset])
                    cost = trade_value * transaction_cost
                    
                    # Update positions
                    self.positions[asset] = target_position
                    self.cash -= trade_quantity * current_prices[asset] + cost
                    
                    total_transaction_cost += cost
                    
                    rebalance_info['trades'][asset] = {
                        'quantity': trade_quantity,
                        'value': trade_quantity * current_prices[asset],
                        'cost': cost
                    }
        
        # Update current weights
        self.current_weights = target_weights.copy()
        rebalance_info['total_cost'] = total_transaction_cost
        
        # Track weight changes
        for i, asset in enumerate(self.assets):
            old_weight = self.current_weights[i] if hasattr(self, 'current_weights') else 1.0/len(self.assets)
            rebalance_info['weight_changes'][asset] = target_weights[i] - old_weight
        
        return rebalance_info
    
    def generate_multi_asset_signals(self, price_data: Dict[str, pd.Series],
                                   optimization_method: str = 'risk_parity',
                                   rebalance_frequency: int = 20) -> pd.DataFrame:
        """
        Generate multi-asset trading signals based on portfolio optimization.
        """
        # Align all price series
        aligned_data = pd.DataFrame(price_data)
        aligned_data = aligned_data.dropna()
        
        if len(aligned_data) < 50:
            logger.warning("Insufficient data for multi-asset signals")
            return pd.DataFrame()
        
        # Generate signals
        signals_data = []
        
        for i in range(50, len(aligned_data), rebalance_frequency):
            current_date = aligned_data.index[i]
            
            # Historical data window
            hist_data = aligned_data.iloc[max(0, i-252):i]  # Up to 1 year
            
            # Convert to price series dict
            hist_prices = {asset: hist_data[asset] for asset in self.assets if asset in hist_data.columns}
            
            if len(hist_prices) == 0:
                continue
            
            # Optimize portfolio
            optimal_weights = self.optimize_portfolio(
                hist_prices, 
                method=optimization_method
            )
            
            # Current prices
            current_prices = {asset: aligned_data.loc[current_date, asset] 
                            for asset in self.assets if asset in aligned_data.columns}
            
            # Generate rebalancing signals
            signal_row = {
                'date': current_date,
                'optimization_method': optimization_method
            }
            
            for j, asset in enumerate(self.assets):
                if j < len(optimal_weights):
                    signal_row[f'{asset}_weight'] = optimal_weights[j]
                    signal_row[f'{asset}_price'] = current_prices.get(asset, np.nan)
                    
                    # Generate buy/sell signal based on weight change
                    old_weight = self.current_weights[j] if hasattr(self, 'current_weights') else 1.0/len(self.assets)
                    weight_change = optimal_weights[j] - old_weight
                    
                    if weight_change > 0.05:  # Increase allocation significantly
                        signal_row[f'{asset}_signal'] = 1  # Buy
                    elif weight_change < -0.05:  # Decrease allocation significantly  
                        signal_row[f'{asset}_signal'] = 2  # Sell
                    else:
                        signal_row[f'{asset}_signal'] = 0  # Hold
                    
                    signal_row[f'{asset}_weight_change'] = weight_change
            
            signals_data.append(signal_row)
            
            # Update current weights for next iteration
            self.current_weights = optimal_weights.copy()
        
        return pd.DataFrame(signals_data)
    
    def calculate_portfolio_metrics(self, returns: pd.Series) -> Dict[str, float]:
        """
        Calculate comprehensive portfolio performance metrics.
        """
        if len(returns) == 0:
            return {}
        
        # Basic metrics
        total_return = (1 + returns).prod() - 1
        annualized_return = (1 + total_return) ** (252 / len(returns)) - 1
        volatility = returns.std() * np.sqrt(252)
        
        # Risk metrics
        downside_returns = returns[returns < 0]
        downside_vol = downside_returns.std() * np.sqrt(252) if len(downside_returns) > 0 else 0
        
        # Sharpe ratio
        risk_free_rate = 0.02  # 2% risk-free rate
        sharpe_ratio = (annualized_return - risk_free_rate) / volatility if volatility > 0 else 0
        
        # Sortino ratio
        sortino_ratio = (annualized_return - risk_free_rate) / downside_vol if downside_vol > 0 else 0
        
        # Maximum drawdown
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdowns = (cumulative - running_max) / running_max
        max_drawdown = drawdowns.min()
        
        # Calmar ratio
        calmar_ratio = annualized_return / abs(max_drawdown) if max_drawdown != 0 else 0
        
        # Win rate
        win_rate = (returns > 0).mean()
        
        return {
            'total_return': total_return,
            'annualized_return': annualized_return,
            'volatility': volatility,
            'sharpe_ratio': sharpe_ratio,
            'sortino_ratio': sortino_ratio,
            'max_drawdown': max_drawdown,
            'calmar_ratio': calmar_ratio,
            'win_rate': win_rate,
            'downside_volatility': downside_vol
        }

if __name__ == '__main__':
    # Test the advanced portfolio optimization system
    logger.info("Testing Advanced Portfolio Optimization System...")
    
    # Create sample multi-asset data
    np.random.seed(42)
    dates = pd.date_range('2022-01-01', '2023-12-31', freq='D')
    assets = ['STOCK_A', 'STOCK_B', 'BOND_C', 'COMMODITY_D']
    
    # Generate correlated price series
    n_assets = len(assets)
    n_days = len(dates)
    
    # Random correlation matrix
    correlation = np.random.rand(n_assets, n_assets)
    correlation = (correlation + correlation.T) / 2
    np.fill_diagonal(correlation, 1.0)
    
    # Generate returns
    returns = np.random.multivariate_normal(
        mean=[0.0005, 0.0003, 0.0001, 0.0008],  # Different expected returns
        cov=correlation * 0.02,  # Scale for realistic volatility
        size=n_days
    )
    
    # Convert to prices
    price_data = {}
    for i, asset in enumerate(assets):
        price_series = 100 * np.cumprod(1 + returns[:, i])
        price_data[asset] = pd.Series(price_series, index=dates)
    
    # Test multi-asset trading system
    trading_system = MultiAssetTradingSystem(assets)
    
    # Test different optimization methods
    methods = ['mean_variance', 'risk_parity', 'hierarchical_risk_parity']
    
    for method in methods:
        logger.info(f"Testing {method} optimization...")
        
        optimal_weights = trading_system.optimize_portfolio(price_data, method=method)
        
        print(f"\n{method.upper()} Optimal Weights:")
        for i, asset in enumerate(assets):
            print(f"  {asset}: {optimal_weights[i]:.1%}")
        
        # Generate signals
        signals = trading_system.generate_multi_asset_signals(
            price_data, 
            optimization_method=method,
            rebalance_frequency=30
        )
        
        print(f"Generated {len(signals)} rebalancing signals")
    
    logger.info("Advanced Portfolio Optimization System test completed!")