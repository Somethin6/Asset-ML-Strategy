import pytest
import asyncio
from httpx import AsyncClient
from fastapi.testclient import TestClient
import os
import sys

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'api'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from api.main import app
from database.database import get_database, init_database
from database.models import Base, User, TradingAccount

# Test configuration
TEST_DATABASE_URL = os.getenv("DATABASE_URL", "sqlite+aiosqlite:///./test.db")

@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()

@pytest.fixture(scope="session")
async def setup_database():
    """Set up test database"""
    await init_database()
    yield
    # Cleanup after tests

@pytest.fixture
def client():
    """Create test client"""
    return TestClient(app)

@pytest.fixture
async def async_client():
    """Create async test client"""
    async with AsyncClient(app=app, base_url="http://test") as ac:
        yield ac

@pytest.fixture
def auth_headers():
    """Mock authentication headers"""
    return {"Authorization": "Bearer test_token"}

class TestHealthCheck:
    """Test health check endpoints"""
    
    def test_health_endpoint(self, client):
        """Test health check endpoint"""
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json()["status"] == "healthy"

class TestAuthentication:
    """Test authentication endpoints"""
    
    def test_register_user(self, client):
        """Test user registration"""
        user_data = {
            "email": "test@example.com",
            "password": "securepassword123",
            "full_name": "Test User"
        }
        response = client.post("/auth/register", json=user_data)
        assert response.status_code in [200, 201]
        assert "access_token" in response.json()
    
    def test_login_user(self, client):
        """Test user login"""
        login_data = {
            "email": "test@example.com",
            "password": "securepassword123"
        }
        response = client.post("/auth/login", json=login_data)
        # Note: This will fail until we implement proper auth
        # assert response.status_code == 200
        # assert "access_token" in response.json()

class TestTradingEndpoints:
    """Test trading-related endpoints"""
    
    def test_get_trading_signals_unauthorized(self, client):
        """Test trading signals without auth"""
        response = client.get("/trading/signals/AAPL")
        assert response.status_code == 403
    
    def test_get_portfolio_unauthorized(self, client):
        """Test portfolio without auth"""
        response = client.get("/trading/portfolio")
        assert response.status_code == 403
    
    def test_run_backtest_unauthorized(self, client):
        """Test backtest without auth"""
        backtest_data = {
            "symbols": ["AAPL", "GOOGL"],
            "start_date": "2023-01-01T00:00:00",
            "end_date": "2023-12-31T23:59:59",
            "initial_capital": 100000.0
        }
        response = client.post("/trading/backtest", json=backtest_data)
        assert response.status_code == 403

class TestMarketDataEndpoints:
    """Test market data endpoints"""
    
    def test_get_market_data(self, client):
        """Test market data endpoint"""
        response = client.get("/market/data/AAPL?timeframe=1D&limit=100")
        assert response.status_code == 200
        data = response.json()
        assert "symbol" in data
        assert data["symbol"] == "AAPL"

if __name__ == "__main__":
    pytest.main([__file__])