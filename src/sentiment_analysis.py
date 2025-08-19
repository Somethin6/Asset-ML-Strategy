#!/usr/bin/env python3
"""
Advanced Sentiment Analysis for Trading
Incorporates news sentiment, social media analysis, and market psychology indicators.
"""

import numpy as np
import pandas as pd
import requests
import re
import logging
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime, timedelta
import time
import warnings
warnings.filterwarnings('ignore')

# For sentiment analysis
try:
    from textblob import TextBlob
    TEXTBLOB_AVAILABLE = True
except ImportError:
    TEXTBLOB_AVAILABLE = False
    
try:
    import nltk
    from nltk.sentiment import SentimentIntensityAnalyzer
    NLTK_AVAILABLE = True
except ImportError:
    NLTK_AVAILABLE = False

logger = logging.getLogger(__name__)

class MarketSentimentAnalyzer:
    """
    Advanced market sentiment analyzer using multiple data sources.
    """
    
    def __init__(self):
        self.sentiment_cache = {}
        self.fear_greed_cache = {}
        
        # Initialize NLTK sentiment analyzer if available
        if NLTK_AVAILABLE:
            try:
                self.sia = SentimentIntensityAnalyzer()
            except:
                try:
                    nltk.download('vader_lexicon', quiet=True)
                    self.sia = SentimentIntensityAnalyzer()
                except:
                    self.sia = None
        else:
            self.sia = None
    
    def analyze_text_sentiment(self, text: str) -> Dict[str, float]:
        """
        Analyze sentiment of text using multiple methods.
        """
        if not text or not isinstance(text, str):
            return {'compound': 0.0, 'positive': 0.0, 'negative': 0.0, 'neutral': 1.0}
        
        results = {'compound': 0.0, 'positive': 0.0, 'negative': 0.0, 'neutral': 1.0}
        
        # Method 1: NLTK VADER (if available)
        if self.sia:
            try:
                scores = self.sia.polarity_scores(text)
                results.update(scores)
            except Exception as e:
                logger.debug(f"NLTK sentiment analysis failed: {e}")
        
        # Method 2: TextBlob (if available)
        if TEXTBLOB_AVAILABLE:
            try:
                blob = TextBlob(text)
                polarity = blob.sentiment.polarity
                subjectivity = blob.sentiment.subjectivity
                
                # Convert TextBlob scores to VADER-like format
                if polarity > 0:
                    results['positive'] = max(results['positive'], polarity)
                elif polarity < 0:
                    results['negative'] = max(results['negative'], abs(polarity))
                else:
                    results['neutral'] = max(results['neutral'], 1 - subjectivity)
                
                results['compound'] = polarity
                
            except Exception as e:
                logger.debug(f"TextBlob sentiment analysis failed: {e}")
        
        # Method 3: Simple keyword-based analysis (fallback)
        if results['compound'] == 0.0:
            results = self._simple_keyword_sentiment(text)
        
        return results
    
    def _simple_keyword_sentiment(self, text: str) -> Dict[str, float]:
        """
        Simple keyword-based sentiment analysis as fallback.
        """
        text = text.lower()
        
        # Positive financial keywords
        positive_words = [
            'bull', 'bullish', 'growth', 'profit', 'gain', 'rise', 'surge', 'rally',
            'strong', 'positive', 'optimistic', 'breakthrough', 'success', 'boom',
            'uptrend', 'momentum', 'beat', 'exceed', 'outperform', 'upgrade'
        ]
        
        # Negative financial keywords
        negative_words = [
            'bear', 'bearish', 'loss', 'decline', 'fall', 'crash', 'drop', 'weak',
            'negative', 'pessimistic', 'recession', 'crisis', 'downgrade', 'miss',
            'underperform', 'risk', 'concern', 'worry', 'fear', 'volatile'
        ]
        
        positive_count = sum(1 for word in positive_words if word in text)
        negative_count = sum(1 for word in negative_words if word in text)
        total_words = len(text.split())
        
        if total_words == 0:
            return {'compound': 0.0, 'positive': 0.0, 'negative': 0.0, 'neutral': 1.0}
        
        pos_score = positive_count / total_words
        neg_score = negative_count / total_words
        compound = pos_score - neg_score
        neutral = max(0.0, 1.0 - pos_score - neg_score)
        
        return {
            'compound': compound,
            'positive': pos_score,
            'negative': neg_score,
            'neutral': neutral
        }
    
    def get_fear_greed_index(self) -> float:
        """
        Simulate Fear & Greed Index (CNN's index).
        Returns value between 0 (extreme fear) and 100 (extreme greed).
        """
        # In a real implementation, you would fetch from CNN or Alternative.me
        # For now, we'll simulate based on market conditions
        
        current_time = datetime.now()
        cache_key = current_time.strftime('%Y-%m-%d')
        
        if cache_key in self.fear_greed_cache:
            return self.fear_greed_cache[cache_key]
        
        # Simulate fear & greed index
        base_index = 50  # Neutral
        
        # Add some randomness to simulate market conditions
        market_volatility = np.random.normal(0, 15)
        fear_greed_value = max(0, min(100, base_index + market_volatility))
        
        self.fear_greed_cache[cache_key] = fear_greed_value
        return fear_greed_value
    
    def analyze_market_news_sentiment(self, symbol: str = 'MARKET') -> Dict[str, float]:
        """
        Analyze overall market sentiment from simulated news.
        In production, this would connect to real news APIs.
        """
        # Simulate news sentiment analysis
        # In production: fetch from NewsAPI, Alpha Vantage, etc.
        
        cache_key = f"{symbol}_{datetime.now().strftime('%Y-%m-%d-%H')}"
        
        if cache_key in self.sentiment_cache:
            return self.sentiment_cache[cache_key]
        
        # Simulate various market sentiment scenarios
        sentiment_scenarios = [
            "Market shows strong bullish momentum with positive earnings reports",
            "Investors remain optimistic about economic growth prospects",
            "Strong buying pressure evident across major sectors",
            "Market sentiment turns bearish amid economic concerns", 
            "Volatility increases as investors show growing uncertainty",
            "Mixed signals from market participants create cautious sentiment",
            "Technical indicators suggest continued upward momentum",
            "Risk-off sentiment dominates as geopolitical tensions rise",
            "Breakthrough in technology sector drives positive sentiment",
            "Federal reserve policy creates market uncertainty"
        ]
        
        # Randomly select and analyze sentiment scenario
        selected_news = np.random.choice(sentiment_scenarios)
        sentiment = self.analyze_text_sentiment(selected_news)
        
        self.sentiment_cache[cache_key] = sentiment
        return sentiment
    
    def get_social_media_sentiment(self, symbol: str = 'MARKET') -> Dict[str, Any]:
        """
        Simulate social media sentiment analysis.
        In production: integrate with Twitter API, Reddit API, etc.
        """
        # Simulate social media posts
        social_posts = [
            f"{symbol} looking strong! #bullish #trading",
            f"Concerned about {symbol} direction, might be time to sell",
            f"{symbol} breaking resistance, could be good entry point",
            f"Market sentiment on {symbol} seems mixed today",
            f"Technical analysis suggests {symbol} could rally further",
            f"Risk management important with {symbol} volatility",
            f"{symbol} fundamentals look solid for long-term hold",
            f"Short-term bearish on {symbol}, watching key levels"
        ]
        
        # Analyze sentiment of multiple posts
        sentiments = []
        for post in np.random.choice(social_posts, size=5, replace=True):
            sentiment = self.analyze_text_sentiment(post)
            sentiments.append(sentiment)
        
        # Aggregate sentiments
        avg_sentiment = {
            'compound': np.mean([s['compound'] for s in sentiments]),
            'positive': np.mean([s['positive'] for s in sentiments]),
            'negative': np.mean([s['negative'] for s in sentiments]),
            'neutral': np.mean([s['neutral'] for s in sentiments])
        }
        
        return {
            'sentiment': avg_sentiment,
            'post_count': len(sentiments),
            'engagement_score': np.random.uniform(0.5, 1.0)  # Simulated engagement
        }
    
    def calculate_market_psychology_indicators(self) -> Dict[str, float]:
        """
        Calculate various market psychology indicators.
        """
        # Fear & Greed Index
        fear_greed = self.get_fear_greed_index()
        
        # News sentiment
        news_sentiment = self.analyze_market_news_sentiment()
        
        # Social media sentiment
        social_sentiment = self.get_social_media_sentiment()
        
        # VIX simulation (volatility index)
        vix_level = max(10, min(80, 20 + np.random.normal(0, 10)))
        
        # Market psychology composite score
        psychology_score = (
            fear_greed * 0.3 +
            (news_sentiment['compound'] + 1) * 50 * 0.25 +  # Convert to 0-100 scale
            (social_sentiment['sentiment']['compound'] + 1) * 50 * 0.25 +
            (100 - min(100, vix_level * 1.25)) * 0.2  # Inverse of VIX
        )
        
        return {
            'fear_greed_index': fear_greed,
            'news_sentiment_compound': news_sentiment['compound'],
            'social_sentiment_compound': social_sentiment['sentiment']['compound'],
            'vix_level': vix_level,
            'psychology_composite': psychology_score,
            'market_regime': self._classify_market_regime(psychology_score)
        }
    
    def _classify_market_regime(self, psychology_score: float) -> str:
        """
        Classify market regime based on psychology score.
        """
        if psychology_score >= 75:
            return 'extreme_greed'
        elif psychology_score >= 60:
            return 'greed'
        elif psychology_score >= 40:
            return 'neutral'
        elif psychology_score >= 25:
            return 'fear'
        else:
            return 'extreme_fear'

class SentimentFeatureEngineer:
    """
    Engineer sentiment-based features for trading models.
    """
    
    def __init__(self):
        self.sentiment_analyzer = MarketSentimentAnalyzer()
        self.sentiment_history = []
        
    def add_sentiment_features(self, df: pd.DataFrame, symbol: str = 'MARKET') -> pd.DataFrame:
        """
        Add sentiment-based features to market data.
        """
        logger.info("Adding sentiment analysis features...")
        
        sentiment_features = []
        
        for idx, row in df.iterrows():
            # Get market psychology indicators
            psychology = self.sentiment_analyzer.calculate_market_psychology_indicators()
            
            # Store in history for momentum calculation
            self.sentiment_history.append(psychology)
            if len(self.sentiment_history) > 20:  # Keep last 20 periods
                self.sentiment_history.pop(0)
            
            # Calculate sentiment momentum
            if len(self.sentiment_history) >= 5:
                recent_sentiment = np.mean([h['psychology_composite'] for h in self.sentiment_history[-5:]])
                older_sentiment = np.mean([h['psychology_composite'] for h in self.sentiment_history[-10:-5]]) \
                    if len(self.sentiment_history) >= 10 else recent_sentiment
                sentiment_momentum = recent_sentiment - older_sentiment
            else:
                sentiment_momentum = 0.0
            
            # Calculate sentiment volatility
            if len(self.sentiment_history) >= 10:
                sentiment_values = [h['psychology_composite'] for h in self.sentiment_history[-10:]]
                sentiment_volatility = np.std(sentiment_values)
            else:
                sentiment_volatility = 0.0
            
            # Create feature vector
            features = {
                'sentiment_fear_greed': psychology['fear_greed_index'],
                'sentiment_news': psychology['news_sentiment_compound'],
                'sentiment_social': psychology['social_sentiment_compound'],
                'sentiment_vix': psychology['vix_level'],
                'sentiment_composite': psychology['psychology_composite'],
                'sentiment_regime_extreme_fear': 1.0 if psychology['market_regime'] == 'extreme_fear' else 0.0,
                'sentiment_regime_fear': 1.0 if psychology['market_regime'] == 'fear' else 0.0,
                'sentiment_regime_neutral': 1.0 if psychology['market_regime'] == 'neutral' else 0.0,
                'sentiment_regime_greed': 1.0 if psychology['market_regime'] == 'greed' else 0.0,
                'sentiment_regime_extreme_greed': 1.0 if psychology['market_regime'] == 'extreme_greed' else 0.0,
                'sentiment_momentum': sentiment_momentum,
                'sentiment_volatility': sentiment_volatility,
                'sentiment_contrarian_signal': self._calculate_contrarian_signal(psychology),
                'sentiment_trend_strength': self._calculate_trend_strength(psychology)
            }
            
            sentiment_features.append(features)
        
        # Convert to DataFrame and add to original data
        sentiment_df = pd.DataFrame(sentiment_features)
        
        # Combine with original data
        result_df = pd.concat([df.reset_index(drop=True), sentiment_df], axis=1)
        
        logger.info(f"Added {len(sentiment_features[0])} sentiment features")
        
        return result_df
    
    def _calculate_contrarian_signal(self, psychology: Dict[str, Any]) -> float:
        """
        Calculate contrarian trading signal based on sentiment extremes.
        """
        composite_score = psychology['psychology_composite']
        
        # Contrarian signals at extremes
        if composite_score >= 80:  # Extreme greed - bearish signal
            return -1.0
        elif composite_score <= 20:  # Extreme fear - bullish signal
            return 1.0
        elif composite_score >= 70:  # Moderate greed - weak bearish
            return -0.5
        elif composite_score <= 30:  # Moderate fear - weak bullish
            return 0.5
        else:
            return 0.0  # Neutral
    
    def _calculate_trend_strength(self, psychology: Dict[str, Any]) -> float:
        """
        Calculate trend strength based on sentiment alignment.
        """
        # Strong trend when multiple sentiment indicators align
        sentiment_scores = [
            (psychology['fear_greed_index'] - 50) / 50,  # Normalize to -1 to 1
            psychology['news_sentiment_compound'],
            psychology['social_sentiment_compound']
        ]
        
        # Measure alignment (how much sentiment indicators agree)
        mean_sentiment = np.mean(sentiment_scores)
        std_sentiment = np.std(sentiment_scores)
        
        # Strong trend when low disagreement and clear direction
        alignment = 1.0 - std_sentiment  # Lower std = higher alignment
        direction_strength = abs(mean_sentiment)
        
        trend_strength = alignment * direction_strength
        
        return min(1.0, trend_strength)

class EconomicIndicatorAnalyzer:
    """
    Analyze economic indicators and their sentiment impact.
    """
    
    def __init__(self):
        self.economic_calendar = {}
        
    def get_economic_calendar_sentiment(self, date: str) -> Dict[str, float]:
        """
        Get sentiment based on economic calendar events.
        Simulates important economic releases and their impact.
        """
        # Simulate major economic events and their typical market impact
        major_events = [
            {'name': 'Fed Rate Decision', 'impact': 'high', 'sentiment_range': (-0.5, 0.5)},
            {'name': 'Non-Farm Payrolls', 'impact': 'high', 'sentiment_range': (-0.3, 0.3)},
            {'name': 'GDP Growth', 'impact': 'medium', 'sentiment_range': (-0.2, 0.2)},
            {'name': 'Inflation Data', 'impact': 'high', 'sentiment_range': (-0.4, 0.4)},
            {'name': 'Earnings Release', 'impact': 'medium', 'sentiment_range': (-0.3, 0.3)}
        ]
        
        # Randomly determine if there's an event today
        if np.random.random() < 0.1:  # 10% chance of major event
            event = np.random.choice(major_events)
            sentiment_impact = np.random.uniform(event['sentiment_range'][0], event['sentiment_range'][1])
            
            return {
                'has_event': True,
                'event_name': event['name'],
                'event_impact': event['impact'],
                'sentiment_impact': sentiment_impact,
                'market_volatility_expected': abs(sentiment_impact) > 0.2
            }
        
        return {
            'has_event': False,
            'event_name': None,
            'event_impact': 'none',
            'sentiment_impact': 0.0,
            'market_volatility_expected': False
        }

class AdvancedSentimentSystem:
    """
    Complete advanced sentiment analysis system for trading.
    """
    
    def __init__(self):
        self.sentiment_analyzer = MarketSentimentAnalyzer()
        self.feature_engineer = SentimentFeatureEngineer()
        self.economic_analyzer = EconomicIndicatorAnalyzer()
        
    def analyze_complete_sentiment(self, df: pd.DataFrame, symbol: str = 'MARKET') -> pd.DataFrame:
        """
        Perform complete sentiment analysis and feature engineering.
        """
        logger.info("Starting complete sentiment analysis...")
        
        # Add basic sentiment features
        df_with_sentiment = self.feature_engineer.add_sentiment_features(df, symbol)
        
        # Add economic calendar features
        economic_features = []
        
        for idx, row in df.iterrows():
            date_str = row.get('date', datetime.now()).strftime('%Y-%m-%d') if 'date' in row else datetime.now().strftime('%Y-%m-%d')
            economic_data = self.economic_analyzer.get_economic_calendar_sentiment(date_str)
            
            econ_features = {
                'econ_has_event': 1.0 if economic_data['has_event'] else 0.0,
                'econ_sentiment_impact': economic_data['sentiment_impact'],
                'econ_volatility_expected': 1.0 if economic_data['market_volatility_expected'] else 0.0,
                'econ_impact_high': 1.0 if economic_data['event_impact'] == 'high' else 0.0,
                'econ_impact_medium': 1.0 if economic_data['event_impact'] == 'medium' else 0.0
            }
            
            economic_features.append(econ_features)
        
        # Add economic features
        economic_df = pd.DataFrame(economic_features)
        df_complete = pd.concat([df_with_sentiment.reset_index(drop=True), economic_df], axis=1)
        
        # Calculate composite sentiment score
        sentiment_columns = [col for col in df_complete.columns if col.startswith('sentiment_')]
        if sentiment_columns:
            df_complete['sentiment_composite_score'] = df_complete[sentiment_columns].mean(axis=1)
        
        logger.info(f"Complete sentiment analysis finished. Added {len(economic_features[0]) + len(sentiment_columns)} features")
        
        return df_complete
    
    def get_sentiment_trading_signals(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate trading signals based on sentiment analysis.
        """
        if 'sentiment_composite_score' not in df.columns:
            df = self.analyze_complete_sentiment(df)
        
        signals = np.zeros(len(df))
        confidences = np.full(len(df), 0.5)
        
        for i in range(len(df)):
            row = df.iloc[i]
            
            # Composite sentiment signal
            composite_score = row.get('sentiment_composite_score', 0)
            contrarian_signal = row.get('sentiment_contrarian_signal', 0)
            trend_strength = row.get('sentiment_trend_strength', 0)
            
            # Generate signal based on multiple factors
            if contrarian_signal > 0.5:  # Strong contrarian buy signal
                signals[i] = 1
                confidences[i] = min(0.9, 0.6 + trend_strength * 0.3)
            elif contrarian_signal < -0.5:  # Strong contrarian sell signal
                signals[i] = 2
                confidences[i] = min(0.9, 0.6 + trend_strength * 0.3)
            elif composite_score > 0.3 and trend_strength > 0.6:  # Trend following buy
                signals[i] = 1
                confidences[i] = min(0.8, 0.5 + trend_strength * 0.3)
            elif composite_score < -0.3 and trend_strength > 0.6:  # Trend following sell
                signals[i] = 2
                confidences[i] = min(0.8, 0.5 + trend_strength * 0.3)
            else:
                signals[i] = 0  # Hold
                confidences[i] = 0.5
        
        return signals, confidences

if __name__ == '__main__':
    # Test the sentiment analysis system
    logger.info("Testing Advanced Sentiment Analysis System...")
    
    # Create sample data
    dates = pd.date_range('2023-01-01', '2023-01-31', freq='D')
    sample_data = pd.DataFrame({
        'date': dates,
        'open': 100 + np.random.randn(len(dates)).cumsum(),
        'high': 105 + np.random.randn(len(dates)).cumsum(),
        'low': 95 + np.random.randn(len(dates)).cumsum(),
        'close': 100 + np.random.randn(len(dates)).cumsum(),
        'volume': np.random.randint(1000, 10000, len(dates))
    })
    
    # Initialize system
    sentiment_system = AdvancedSentimentSystem()
    
    # Analyze sentiment
    df_with_sentiment = sentiment_system.analyze_complete_sentiment(sample_data, 'TEST')
    
    print(f"Original features: {len(sample_data.columns)}")
    print(f"With sentiment features: {len(df_with_sentiment.columns)}")
    print(f"Added {len(df_with_sentiment.columns) - len(sample_data.columns)} sentiment features")
    
    # Generate signals
    signals, confidences = sentiment_system.get_sentiment_trading_signals(df_with_sentiment)
    print(f"Generated signals: {np.sum(signals == 1)} buy, {np.sum(signals == 2)} sell, {np.sum(signals == 0)} hold")
    
    logger.info("Advanced Sentiment Analysis System test completed!")