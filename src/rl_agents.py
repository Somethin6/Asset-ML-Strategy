#!/usr/bin/env python3
"""
Advanced Reinforcement Learning Trading Agents
Implements multiple RL algorithms for autonomous trading decisions.
"""

import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import deque
import random
import logging
from typing import Dict, List, Tuple, Optional, Any
from stable_baselines3 import PPO, A2C, SAC, TD3, DDPG
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.callbacks import BaseCallback
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

class AdvancedTradingEnvironment(gym.Env):
    """
    Advanced trading environment with multiple assets and complex reward structure.
    """
    
    def __init__(self, data: pd.DataFrame, initial_balance=100000, lookback_window=60, 
                 transaction_cost=0.001, max_position=1.0, features_columns=None):
        super().__init__()
        
        self.data = data.copy()
        self.initial_balance = initial_balance
        self.lookback_window = lookback_window
        self.transaction_cost = transaction_cost
        self.max_position = max_position
        
        # Feature columns (if not provided, use all except OHLCV)
        if features_columns is None:
            excluded_cols = ['open', 'high', 'low', 'close', 'volume', 'target', 'future_return']
            self.features_columns = [col for col in data.columns if col not in excluded_cols]
        else:
            self.features_columns = features_columns
        
        self.n_features = len(self.features_columns)
        
        # Action space: [position_change, confidence]
        # position_change: -1 (sell all) to 1 (buy all), 0 (hold)
        # confidence: 0 to 1 (how confident in the decision)
        self.action_space = spaces.Box(
            low=np.array([-1.0, 0.0]), 
            high=np.array([1.0, 1.0]), 
            dtype=np.float32
        )
        
        # Observation space: features + portfolio state
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.lookback_window * self.n_features + 10,),  # +10 for portfolio state
            dtype=np.float32
        )
        
        # State variables
        self.reset()
    
    def reset(self, seed=None):
        """Reset environment to initial state."""
        super().reset(seed=seed)
        
        self.current_step = self.lookback_window
        self.balance = self.initial_balance
        self.position = 0.0  # -1 to 1 (short to long)
        self.shares = 0.0
        self.portfolio_value = self.initial_balance
        self.total_trades = 0
        self.winning_trades = 0
        self.total_fees = 0.0
        
        # Performance tracking
        self.max_portfolio_value = self.initial_balance
        self.drawdown = 0.0
        self.episode_returns = []
        
        # Market state
        self.last_price = self.data.iloc[self.current_step]['close']
        
        return self._get_observation(), {}
    
    def step(self, action):
        """Execute one trading step."""
        if self.current_step >= len(self.data) - 1:
            return self._get_observation(), 0, True, True, {}
        
        # Parse action
        position_change, confidence = action
        position_change = np.clip(position_change, -1, 1)
        confidence = np.clip(confidence, 0, 1)
        
        # Current market data
        current_price = self.data.iloc[self.current_step]['close']
        next_price = self.data.iloc[self.current_step + 1]['close']
        
        # Calculate position change
        old_position = self.position
        target_position = np.clip(position_change * confidence, -self.max_position, self.max_position)
        position_delta = target_position - old_position
        
        # Execute trade if significant position change
        trade_executed = False
        if abs(position_delta) > 0.01:  # Minimum position change threshold
            self._execute_trade(position_delta, current_price)
            trade_executed = True
        
        # Move to next step
        self.current_step += 1
        
        # Calculate returns
        price_return = (next_price - current_price) / current_price
        portfolio_return = self.position * price_return
        
        # Update portfolio value
        old_portfolio_value = self.portfolio_value
        self.portfolio_value = self.balance + self.shares * next_price
        
        # Calculate reward with advanced components
        reward = self._calculate_advanced_reward(
            portfolio_return, old_portfolio_value, trade_executed, confidence
        )
        
        # Update performance metrics
        self._update_performance_metrics()
        
        # Check if episode is done
        done = (
            self.current_step >= len(self.data) - 1 or
            self.portfolio_value <= self.initial_balance * 0.1  # Stop loss at 90% loss
        )
        
        info = {
            'portfolio_value': self.portfolio_value,
            'position': self.position,
            'total_trades': self.total_trades,
            'win_rate': self.winning_trades / max(1, self.total_trades),
            'drawdown': self.drawdown,
            'price_return': price_return,
            'portfolio_return': portfolio_return
        }
        
        return self._get_observation(), reward, done, False, info
    
    def _execute_trade(self, position_delta: float, price: float):
        """Execute a trade with transaction costs."""
        # Calculate shares to trade
        max_shares = self.portfolio_value // price
        target_shares = self.position * max_shares
        new_shares = (self.position + position_delta) * max_shares
        shares_delta = new_shares - self.shares
        
        # Execute trade
        if shares_delta != 0:
            trade_value = abs(shares_delta * price)
            fees = trade_value * self.transaction_cost
            
            if shares_delta > 0:  # Buying
                required_cash = shares_delta * price + fees
                if required_cash <= self.balance:
                    self.balance -= required_cash
                    self.shares += shares_delta
                    self.position += position_delta
                    self.total_fees += fees
                    self.total_trades += 1
            else:  # Selling
                self.balance += abs(shares_delta) * price - fees
                self.shares += shares_delta  # shares_delta is negative
                self.position += position_delta
                self.total_fees += fees
                self.total_trades += 1
                
                # Check if winning trade
                if shares_delta < 0 and position_delta < 0:  # Selling a long position
                    self.winning_trades += 1
    
    def _calculate_advanced_reward(self, portfolio_return: float, old_portfolio_value: float,
                                 trade_executed: bool, confidence: float) -> float:
        """Calculate advanced reward with multiple components."""
        # Base return reward
        return_reward = portfolio_return * 100
        
        # Risk-adjusted reward (Sharpe ratio component)
        if len(self.episode_returns) > 10:
            returns_std = np.std(self.episode_returns[-10:])
            if returns_std > 0:
                risk_adjusted_reward = portfolio_return / returns_std
            else:
                risk_adjusted_reward = portfolio_return
        else:
            risk_adjusted_reward = portfolio_return
        
        # Drawdown penalty
        drawdown_penalty = -self.drawdown * 10
        
        # Transaction cost penalty
        transaction_penalty = -0.1 if trade_executed else 0
        
        # Confidence reward (reward confident correct decisions)
        if portfolio_return > 0:
            confidence_reward = confidence * 0.5
        else:
            confidence_reward = -(confidence * 0.5)
        
        # Position management reward
        position_reward = 0
        if abs(self.position) < 0.1:  # Reward for staying out when uncertain
            position_reward = 0.1
        elif abs(self.position) > 0.8 and portfolio_return > 0:  # Reward for strong positions when right
            position_reward = 0.2
        
        # Total reward
        total_reward = (
            return_reward * 0.4 +
            risk_adjusted_reward * 0.3 +
            drawdown_penalty * 0.1 +
            transaction_penalty * 0.05 +
            confidence_reward * 0.1 +
            position_reward * 0.05
        )
        
        # Track returns for risk calculation
        self.episode_returns.append(portfolio_return)
        if len(self.episode_returns) > 100:
            self.episode_returns.pop(0)
        
        return total_reward
    
    def _update_performance_metrics(self):
        """Update performance tracking metrics."""
        # Update max portfolio value and drawdown
        if self.portfolio_value > self.max_portfolio_value:
            self.max_portfolio_value = self.portfolio_value
            self.drawdown = 0.0
        else:
            self.drawdown = (self.max_portfolio_value - self.portfolio_value) / self.max_portfolio_value
    
    def _get_observation(self):
        """Get current observation state."""
        # Market features (lookback window)
        start_idx = max(0, self.current_step - self.lookback_window)
        end_idx = self.current_step
        
        market_data = self.data[self.features_columns].iloc[start_idx:end_idx].values
        
        # Pad if not enough historical data
        if len(market_data) < self.lookback_window:
            padding = np.zeros((self.lookback_window - len(market_data), self.n_features))
            market_data = np.vstack([padding, market_data])
        
        # Flatten market data
        market_features = market_data.flatten()
        
        # Portfolio state
        current_price = self.data.iloc[self.current_step]['close']
        portfolio_state = np.array([
            self.position,  # Current position
            self.balance / self.initial_balance,  # Normalized balance
            self.portfolio_value / self.initial_balance,  # Normalized portfolio value
            self.drawdown,  # Current drawdown
            self.total_trades / 100,  # Normalized trade count
            self.winning_trades / max(1, self.total_trades),  # Win rate
            current_price / 1000,  # Normalized price
            len(self.episode_returns) / 100,  # Time in episode
            np.mean(self.episode_returns[-10:]) if self.episode_returns else 0,  # Recent return average
            np.std(self.episode_returns[-10:]) if len(self.episode_returns) > 1 else 0  # Recent volatility
        ])
        
        # Combine features
        observation = np.concatenate([market_features, portfolio_state]).astype(np.float32)
        
        return observation

class MultiAgentRLSystem:
    """
    Multi-agent reinforcement learning system with different algorithms.
    """
    
    def __init__(self, env_class=AdvancedTradingEnvironment):
        self.env_class = env_class
        self.agents = {}
        self.trained = False
        
    def create_environment(self, data: pd.DataFrame, **env_kwargs):
        """Create trading environment."""
        return self.env_class(data, **env_kwargs)
    
    def initialize_agents(self, env, total_timesteps_per_agent=10000):
        """Initialize different RL agents."""
        logger.info("Initializing RL agents...")
        
        # PPO Agent (Policy Gradient)
        self.agents['ppo'] = PPO(
            'MlpPolicy',
            env,
            learning_rate=3e-4,
            n_steps=2048,
            batch_size=64,
            n_epochs=10,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            verbose=0
        )
        
        # A2C Agent (Actor-Critic)
        self.agents['a2c'] = A2C(
            'MlpPolicy',
            env,
            learning_rate=7e-4,
            n_steps=5,
            gamma=0.99,
            gae_lambda=1.0,
            verbose=0
        )
        
        # SAC Agent (for continuous action spaces) 
        try:
            self.agents['sac'] = SAC(
                'MlpPolicy',
                env,
                learning_rate=3e-4,
                buffer_size=100000,
                batch_size=256,
                gamma=0.99,
                tau=0.005,
                verbose=0
            )
        except Exception as e:
            logger.warning(f"Could not initialize SAC agent: {e}")
        
        self.total_timesteps_per_agent = total_timesteps_per_agent
        
        logger.info(f"Initialized {len(self.agents)} RL agents")
    
    def train_agents(self, env, timesteps_per_agent=None):
        """Train all RL agents."""
        if timesteps_per_agent is None:
            timesteps_per_agent = self.total_timesteps_per_agent
        
        training_results = {}
        
        for agent_name, agent in self.agents.items():
            logger.info(f"Training {agent_name.upper()} agent...")
            
            try:
                # Train agent
                agent.learn(total_timesteps=timesteps_per_agent)
                
                # Evaluate performance
                obs, _ = env.reset()
                total_reward = 0
                steps = 0
                done = False
                
                while not done and steps < 1000:
                    action, _ = agent.predict(obs, deterministic=True)
                    obs, reward, done, _, info = env.step(action)
                    total_reward += reward
                    steps += 1
                
                training_results[agent_name] = {
                    'total_reward': total_reward,
                    'steps': steps,
                    'final_portfolio_value': info.get('portfolio_value', 0)
                }
                
                logger.info(f"{agent_name.upper()} training completed. "
                          f"Reward: {total_reward:.2f}, "
                          f"Portfolio: ${info.get('portfolio_value', 0):,.2f}")
                
            except Exception as e:
                logger.error(f"Error training {agent_name} agent: {e}")
                training_results[agent_name] = {
                    'total_reward': 0,
                    'steps': 0,
                    'final_portfolio_value': 0
                }
        
        self.trained = True
        return training_results
    
    def predict_ensemble(self, env, num_episodes=1) -> Dict[str, Any]:
        """Make ensemble predictions from all RL agents."""
        if not self.trained:
            raise ValueError("Agents not trained yet. Call train_agents first.")
        
        ensemble_results = {}
        
        for agent_name, agent in self.agents.items():
            try:
                obs, _ = env.reset()
                episode_actions = []
                episode_rewards = []
                done = False
                steps = 0
                
                while not done and steps < len(env.data) - env.lookback_window - 1:
                    action, _ = agent.predict(obs, deterministic=True)
                    obs, reward, done, _, info = env.step(action)
                    
                    episode_actions.append(action)
                    episode_rewards.append(reward)
                    steps += 1
                
                ensemble_results[agent_name] = {
                    'actions': np.array(episode_actions),
                    'rewards': np.array(episode_rewards),
                    'final_portfolio_value': info.get('portfolio_value', env.initial_balance),
                    'total_trades': info.get('total_trades', 0),
                    'win_rate': info.get('win_rate', 0)
                }
                
            except Exception as e:
                logger.error(f"Error predicting with {agent_name} agent: {e}")
                ensemble_results[agent_name] = {
                    'actions': np.array([]),
                    'rewards': np.array([]),
                    'final_portfolio_value': env.initial_balance,
                    'total_trades': 0,
                    'win_rate': 0
                }
        
        return ensemble_results
    
    def get_ensemble_signals(self, env) -> Tuple[np.ndarray, np.ndarray]:
        """Get ensemble trading signals from all RL agents."""
        ensemble_results = self.predict_ensemble(env)
        
        if not ensemble_results:
            return np.zeros(len(env.data)), np.full(len(env.data), 0.5)
        
        # Combine signals from all agents
        all_actions = []
        agent_weights = []
        
        for agent_name, results in ensemble_results.items():
            if len(results['actions']) > 0:
                # Convert actions to signals (position changes to buy/sell/hold)
                actions = results['actions'][:, 0] if len(results['actions'].shape) > 1 else results['actions']
                
                # Convert to signals: -1 (sell), 0 (hold), 1 (buy)
                signals = np.where(actions > 0.3, 1, np.where(actions < -0.3, -1, 0))
                
                all_actions.append(signals)
                
                # Weight by performance (portfolio value and win rate)
                performance_weight = (
                    results['final_portfolio_value'] / env.initial_balance * 0.7 +
                    results['win_rate'] * 0.3
                )
                agent_weights.append(max(0.1, performance_weight))  # Minimum weight of 0.1
        
        if not all_actions:
            return np.zeros(len(env.data)), np.full(len(env.data), 0.5)
        
        # Pad actions to match data length
        max_len = len(env.data) - env.lookback_window
        padded_actions = []
        
        for actions in all_actions:
            if len(actions) < max_len:
                # Pad with zeros (hold signals)
                padded = np.zeros(max_len)
                padded[:len(actions)] = actions
            else:
                padded = actions[:max_len]
            padded_actions.append(padded)
        
        # Weighted ensemble voting
        agent_weights = np.array(agent_weights)
        agent_weights = agent_weights / agent_weights.sum()  # Normalize weights
        
        ensemble_signals = np.zeros(max_len)
        for actions, weight in zip(padded_actions, agent_weights):
            ensemble_signals += actions * weight
        
        # Convert to discrete signals and probabilities
        final_signals = np.where(
            ensemble_signals > 0.3, 1,
            np.where(ensemble_signals < -0.3, 2, 0)
        )
        
        # Calculate probabilities based on signal strength
        signal_probs = np.abs(ensemble_signals)
        signal_probs = np.clip(signal_probs, 0.5, 1.0)  # Minimum 50% confidence
        
        # Pad to match original data length
        final_signals = np.concatenate([
            np.zeros(env.lookback_window), final_signals
        ])[:len(env.data)]
        
        signal_probs = np.concatenate([
            np.full(env.lookback_window, 0.5), signal_probs
        ])[:len(env.data)]
        
        return final_signals, signal_probs

class QuantumInspiredRLAgent:
    """
    Quantum-inspired reinforcement learning agent using superposition and entanglement concepts.
    """
    
    def __init__(self, state_size, action_size, quantum_dim=16):
        self.state_size = state_size
        self.action_size = action_size
        self.quantum_dim = quantum_dim
        
        # Quantum-inspired neural network
        self.q_network = self._build_quantum_network()
        self.target_network = self._build_quantum_network()
        
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=1e-3)
        
        # Experience replay
        self.memory = deque(maxlen=10000)
        self.batch_size = 32
        self.gamma = 0.99
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        
        # Update target network
        self._update_target_network()
    
    def _build_quantum_network(self):
        """Build quantum-inspired neural network."""
        return nn.Sequential(
            nn.Linear(self.state_size, 128),
            nn.ReLU(),
            
            # Quantum superposition layer
            nn.Linear(128, self.quantum_dim * 2),  # Real and imaginary parts
            nn.Tanh(),
            
            # Quantum entanglement layer
            nn.Linear(self.quantum_dim * 2, self.quantum_dim),
            nn.ReLU(),
            
            # Classical output layer
            nn.Linear(self.quantum_dim, 64),
            nn.ReLU(),
            nn.Linear(64, self.action_size)
        )
    
    def _update_target_network(self):
        """Update target network with current network weights."""
        self.target_network.load_state_dict(self.q_network.state_dict())
    
    def remember(self, state, action, reward, next_state, done):
        """Store experience in replay buffer."""
        self.memory.append((state, action, reward, next_state, done))
    
    def act(self, state):
        """Choose action using epsilon-greedy policy with quantum enhancement."""
        if np.random.random() <= self.epsilon:
            # Quantum-inspired random exploration
            quantum_noise = np.random.normal(0, 0.1, self.action_size)
            return np.clip(np.random.randn(self.action_size) + quantum_noise, -1, 1)
        
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        q_values = self.q_network(state_tensor)
        
        # Apply quantum superposition to action selection
        action = torch.tanh(q_values).squeeze().detach().numpy()
        
        return action
    
    def replay(self):
        """Train the network on a batch of experiences."""
        if len(self.memory) < self.batch_size:
            return
        
        batch = random.sample(self.memory, self.batch_size)
        states = torch.FloatTensor([e[0] for e in batch])
        actions = torch.FloatTensor([e[1] for e in batch])
        rewards = torch.FloatTensor([e[2] for e in batch])
        next_states = torch.FloatTensor([e[3] for e in batch])
        dones = torch.BoolTensor([e[4] for e in batch])
        
        current_q_values = self.q_network(states)
        next_q_values = self.target_network(next_states).detach()
        
        target_q_values = rewards + (self.gamma * torch.max(next_q_values, 1)[0] * ~dones)
        
        # Quantum-inspired loss function
        loss = F.mse_loss(torch.max(current_q_values, 1)[0], target_q_values)
        
        # Add quantum regularization
        quantum_reg = 0.01 * torch.sum(torch.square(current_q_values))
        total_loss = loss + quantum_reg
        
        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()
        
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
    
    def train(self, env, episodes=1000):
        """Train the quantum-inspired RL agent."""
        scores = []
        
        for episode in range(episodes):
            state, _ = env.reset()
            total_reward = 0
            done = False
            steps = 0
            
            while not done and steps < 1000:
                action = self.act(state)
                next_state, reward, done, _, info = env.step(action)
                
                self.remember(state, action, reward, next_state, done)
                state = next_state
                total_reward += reward
                steps += 1
                
                if len(self.memory) > self.batch_size:
                    self.replay()
            
            scores.append(total_reward)
            
            # Update target network periodically
            if episode % 100 == 0:
                self._update_target_network()
            
            if episode % 100 == 0:
                logger.info(f"Episode {episode}, Average Score: {np.mean(scores[-100:]):.2f}")
        
        return scores

if __name__ == '__main__':
    # Test the RL system
    logger.info("Testing Advanced RL System...")
    
    # Create sample data
    dates = pd.date_range('2023-01-01', '2023-12-31', freq='H')
    n_samples = len(dates)
    
    sample_data = pd.DataFrame({
        'date': dates,
        'open': 100 + np.random.randn(n_samples).cumsum(),
        'high': 105 + np.random.randn(n_samples).cumsum(),
        'low': 95 + np.random.randn(n_samples).cumsum(),
        'close': 100 + np.random.randn(n_samples).cumsum(),
        'volume': np.random.randint(1000, 10000, n_samples)
    })
    
    # Add some basic features
    for i in range(10):
        sample_data[f'feature_{i}'] = np.random.randn(n_samples)
    
    # Create RL system
    rl_system = MultiAgentRLSystem()
    env = rl_system.create_environment(sample_data)
    
    # Initialize and train agents
    rl_system.initialize_agents(env, total_timesteps_per_agent=1000)
    results = rl_system.train_agents(env, timesteps_per_agent=1000)
    
    print("RL Training Results:", results)
    
    # Get ensemble signals
    signals, probabilities = rl_system.get_ensemble_signals(env)
    print(f"Generated {len(signals)} signals with {np.sum(signals > 0)} buy signals")
    
    logger.info("Advanced RL System test completed!")