#!/usr/bin/env python3
"""
Simplified API for local testing and development
"""

import os
from datetime import datetime
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn

app = FastAPI(
    title="Asset-ML-Strategy API - Dev",
    description="Development version of production API",
    version="2.0.0-dev",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS middleware for frontend communication
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic models
class HealthResponse(BaseModel):
    status: str
    timestamp: str
    version: str

class TradingSignal(BaseModel):
    symbol: str
    action: str
    confidence: float
    price: float
    timestamp: str

# Health check endpoint
@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    return HealthResponse(
        status="healthy",
        timestamp=datetime.now().isoformat(),
        version="2.0.0-dev"
    )

# Mock trading signals
@app.get("/trading/signals/{symbol}", response_model=TradingSignal)
async def get_trading_signals(symbol: str):
    """Get trading signals for a symbol"""
    return TradingSignal(
        symbol=symbol,
        action="BUY",
        confidence=0.85,
        price=150.25,
        timestamp=datetime.now().isoformat()
    )

# Mock portfolio data
@app.get("/trading/portfolio")
async def get_portfolio():
    """Get current portfolio status"""
    return {
        "total_value": 125000.50,
        "available_cash": 25000.00,
        "daily_pnl": 2500.00,
        "total_pnl": 25000.50,
        "positions": [
            {
                "symbol": "AAPL",
                "quantity": 100,
                "avg_price": 150.00,
                "current_price": 155.25,
                "unrealized_pnl": 525.00
            },
            {
                "symbol": "GOOGL", 
                "quantity": 50,
                "avg_price": 2800.00,
                "current_price": 2850.50,
                "unrealized_pnl": 2525.00
            }
        ]
    }

# Mock market data
@app.get("/market/data/{symbol}")
async def get_market_data(symbol: str, timeframe: str = "1D", limit: int = 100):
    """Get market data for a symbol"""
    import random
    
    # Generate mock OHLCV data
    data_points = []
    base_price = 150.0
    
    for i in range(limit):
        open_price = base_price + random.uniform(-5, 5)
        close_price = open_price + random.uniform(-3, 3)
        high_price = max(open_price, close_price) + random.uniform(0, 2)
        low_price = min(open_price, close_price) - random.uniform(0, 2)
        volume = random.uniform(1000000, 5000000)
        
        data_points.append({
            "timestamp": f"2024-01-{i+1:02d}T00:00:00Z",
            "open": round(open_price, 2),
            "high": round(high_price, 2),
            "low": round(low_price, 2),
            "close": round(close_price, 2),
            "volume": int(volume)
        })
        
        base_price = close_price
    
    return {
        "symbol": symbol,
        "timeframe": timeframe,
        "data": data_points
    }

# Mock analytics
@app.get("/analytics/performance")
async def get_performance_analytics():
    """Get comprehensive performance analytics"""
    return {
        "summary": {
            "total_return": 25.50,
            "sharpe_ratio": 2.1,
            "max_drawdown": -5.2,
            "calmar_ratio": 4.9,
            "win_rate": 0.65
        },
        "monthly_returns": [2.1, 3.5, -1.2, 4.8, 2.9, 1.8, -0.5, 3.2, 2.6, 1.4, 4.1, 2.8],
        "risk_metrics": {
            "var_95": -2.5,
            "var_99": -4.2,
            "expected_shortfall": -3.1,
            "volatility": 15.2
        }
    }

# Mock strategies
@app.get("/strategies")
async def get_strategies():
    """Get available trading strategies"""
    return [
        {
            "id": "ml_ensemble",
            "name": "ML Ensemble Strategy",
            "description": "Advanced machine learning ensemble with 6 models",
            "status": "active",
            "performance": {
                "total_return": 645.715,
                "sharpe_ratio": 6.71,
                "max_drawdown": -1.82,
                "win_rate": 0.4725
            }
        },
        {
            "id": "momentum",
            "name": "Momentum Strategy", 
            "description": "Price momentum based strategy",
            "status": "active",
            "performance": {
                "total_return": 123.45,
                "sharpe_ratio": 1.85,
                "max_drawdown": -8.5,
                "win_rate": 0.58
            }
        }
    ]

# Startup message
@app.on_event("startup")
async def startup_event():
    print("🚀 Asset-ML-Strategy Development API started")
    print("📊 Available endpoints:")
    print("   - http://localhost:8000/docs (API Documentation)")
    print("   - http://localhost:8000/health (Health Check)")
    print("   - http://localhost:8000/trading/portfolio (Portfolio)")
    print("   - http://localhost:8000/trading/signals/AAPL (Trading Signals)")

if __name__ == "__main__":
    uvicorn.run(
        "api_dev:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )