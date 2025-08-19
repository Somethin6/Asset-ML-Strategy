import { createSlice, createAsyncThunk } from '@reduxjs/toolkit'

interface Position {
  symbol: string
  quantity: number
  avgPrice: number
  currentPrice: number
  unrealizedPnl: number
  marketValue: number
  percentage: number
}

interface PortfolioState {
  totalValue: number
  availableCash: number
  totalPnl: number
  dailyPnl: number
  dailyReturn: number
  positions: Position[]
  performance: {
    totalReturn: number
    sharpeRatio: number
    maxDrawdown: number
    volatility: number
    winRate: number
  }
  isLoading: boolean
  error: string | null
  lastUpdated: string | null
}

const initialState: PortfolioState = {
  totalValue: 0,
  availableCash: 0,
  totalPnl: 0,
  dailyPnl: 0,
  dailyReturn: 0,
  positions: [],
  performance: {
    totalReturn: 0,
    sharpeRatio: 0,
    maxDrawdown: 0,
    volatility: 0,
    winRate: 0,
  },
  isLoading: false,
  error: null,
  lastUpdated: null,
}

export const fetchPortfolio = createAsyncThunk(
  'portfolio/fetch',
  async () => {
    const response = await fetch('/api/trading/portfolio')
    
    if (!response.ok) {
      throw new Error('Failed to fetch portfolio')
    }
    
    const data = await response.json()
    return data
  }
)

export const fetchPerformanceAnalytics = createAsyncThunk(
  'portfolio/fetchAnalytics',
  async () => {
    const response = await fetch('/api/analytics/performance')
    
    if (!response.ok) {
      throw new Error('Failed to fetch performance analytics')
    }
    
    return await response.json()
  }
)

const portfolioSlice = createSlice({
  name: 'portfolio',
  initialState,
  reducers: {
    clearError: (state) => {
      state.error = null
    },
    updatePosition: (state, action) => {
      const { symbol, currentPrice } = action.payload
      const position = state.positions.find(p => p.symbol === symbol)
      if (position) {
        position.currentPrice = currentPrice
        position.unrealizedPnl = (currentPrice - position.avgPrice) * position.quantity
        position.marketValue = currentPrice * position.quantity
      }
    },
  },
  extraReducers: (builder) => {
    builder
      .addCase(fetchPortfolio.pending, (state) => {
        state.isLoading = true
        state.error = null
      })
      .addCase(fetchPortfolio.fulfilled, (state, action) => {
        state.isLoading = false
        state.totalValue = action.payload.total_value
        state.availableCash = action.payload.available_cash
        state.totalPnl = action.payload.performance?.total_pnl || 0
        state.dailyPnl = action.payload.performance?.daily_pnl || 0
        state.dailyReturn = action.payload.performance?.daily_return || 0
        
        // Process positions
        state.positions = action.payload.positions?.map((pos: any) => ({
          symbol: pos.symbol,
          quantity: pos.quantity,
          avgPrice: pos.avg_price,
          currentPrice: pos.current_price,
          unrealizedPnl: pos.unrealized_pnl,
          marketValue: pos.current_price * pos.quantity,
          percentage: (pos.current_price * pos.quantity) / state.totalValue * 100,
        })) || []
        
        state.lastUpdated = new Date().toISOString()
      })
      .addCase(fetchPortfolio.rejected, (state, action) => {
        state.isLoading = false
        state.error = action.error.message || 'Failed to fetch portfolio'
      })
      .addCase(fetchPerformanceAnalytics.fulfilled, (state, action) => {
        state.performance = {
          totalReturn: action.payload.summary.total_return,
          sharpeRatio: action.payload.summary.sharpe_ratio,
          maxDrawdown: action.payload.summary.max_drawdown,
          volatility: action.payload.risk_metrics?.volatility || 0,
          winRate: action.payload.summary.win_rate || 0,
        }
      })
  },
})

export const { clearError, updatePosition } = portfolioSlice.actions
export default portfolioSlice.reducer