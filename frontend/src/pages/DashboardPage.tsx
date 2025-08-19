import React, { useEffect } from 'react'
import { useDispatch, useSelector } from 'react-redux'
import { 
  Grid, 
  Card, 
  CardContent, 
  Typography, 
  Box, 
  Chip,
  useTheme,
  LinearProgress
} from '@mui/material'
import { 
  TrendingUp, 
  TrendingDown, 
  AccountBalance, 
  Assessment,
  Timer,
  ShowChart
} from '@mui/icons-material'

import { RootState } from '../store/store'
import { fetchPortfolio, fetchPerformanceAnalytics } from '../store/slices/portfolioSlice'
import { PortfolioChart } from '../components/charts/PortfolioChart'
import { PositionsTable } from '../components/portfolio/PositionsTable'
import { PerformanceMetrics } from '../components/analytics/PerformanceMetrics'
import { TradingSignalsWidget } from '../components/trading/TradingSignalsWidget'

const DashboardPage: React.FC = () => {
  const theme = useTheme()
  const dispatch = useDispatch()
  
  const { 
    totalValue, 
    availableCash, 
    dailyPnl, 
    dailyReturn, 
    positions,
    performance,
    isLoading 
  } = useSelector((state: RootState) => state.portfolio)

  const { systemStatus } = useSelector((state: RootState) => state.app)

  useEffect(() => {
    dispatch(fetchPortfolio())
    dispatch(fetchPerformanceAnalytics())
    
    // Set up real-time updates
    const interval = setInterval(() => {
      dispatch(fetchPortfolio())
    }, 30000) // Update every 30 seconds

    return () => clearInterval(interval)
  }, [dispatch])

  const formatCurrency = (value: number) => 
    new Intl.NumberFormat('en-US', { 
      style: 'currency', 
      currency: 'USD' 
    }).format(value)

  const formatPercentage = (value: number) => 
    `${value > 0 ? '+' : ''}${value.toFixed(2)}%`

  if (isLoading && positions.length === 0) {
    return (
      <Box sx={{ width: '100%', mt: 2 }}>
        <LinearProgress />
        <Typography variant="body2" sx={{ mt: 1, textAlign: 'center' }}>
          Loading portfolio data...
        </Typography>
      </Box>
    )
  }

  return (
    <Box sx={{ flexGrow: 1, p: 3 }}>
      {/* Header */}
      <Box sx={{ mb: 4 }}>
        <Typography variant="h4" component="h1" gutterBottom>
          Trading Dashboard
        </Typography>
        <Box sx={{ display: 'flex', gap: 1, alignItems: 'center' }}>
          <Chip 
            icon={<Timer />}
            label={`Last updated: ${new Date().toLocaleTimeString()}`}
            size="small"
            color="primary"
          />
          <Chip
            label={`API: ${systemStatus.api}`}
            size="small"
            color={systemStatus.api === 'healthy' ? 'success' : 'error'}
          />
        </Box>
      </Box>

      {/* Key Metrics Row */}
      <Grid container spacing={3} sx={{ mb: 4 }}>
        <Grid item xs={12} sm={6} md={3}>
          <Card>
            <CardContent>
              <Box sx={{ display: 'flex', alignItems: 'center', mb: 1 }}>
                <AccountBalance color="primary" sx={{ mr: 1 }} />
                <Typography variant="subtitle2" color="textSecondary">
                  Total Value
                </Typography>
              </Box>
              <Typography variant="h5" component="div">
                {formatCurrency(totalValue)}
              </Typography>
              <Typography variant="body2" color="textSecondary">
                Cash: {formatCurrency(availableCash)}
              </Typography>
            </CardContent>
          </Card>
        </Grid>

        <Grid item xs={12} sm={6} md={3}>
          <Card>
            <CardContent>
              <Box sx={{ display: 'flex', alignItems: 'center', mb: 1 }}>
                {dailyPnl >= 0 ? 
                  <TrendingUp color="success" sx={{ mr: 1 }} /> : 
                  <TrendingDown color="error" sx={{ mr: 1 }} />
                }
                <Typography variant="subtitle2" color="textSecondary">
                  Daily P&L
                </Typography>
              </Box>
              <Typography 
                variant="h5" 
                component="div"
                color={dailyPnl >= 0 ? 'success.main' : 'error.main'}
              >
                {formatCurrency(dailyPnl)}
              </Typography>
              <Typography 
                variant="body2" 
                color={dailyReturn >= 0 ? 'success.main' : 'error.main'}
              >
                {formatPercentage(dailyReturn)}
              </Typography>
            </CardContent>
          </Card>
        </Grid>

        <Grid item xs={12} sm={6} md={3}>
          <Card>
            <CardContent>
              <Box sx={{ display: 'flex', alignItems: 'center', mb: 1 }}>
                <Assessment color="primary" sx={{ mr: 1 }} />
                <Typography variant="subtitle2" color="textSecondary">
                  Sharpe Ratio
                </Typography>
              </Box>
              <Typography variant="h5" component="div">
                {performance.sharpeRatio.toFixed(2)}
              </Typography>
              <Typography variant="body2" color="textSecondary">
                Total Return: {formatPercentage(performance.totalReturn)}
              </Typography>
            </CardContent>
          </Card>
        </Grid>

        <Grid item xs={12} sm={6} md={3}>
          <Card>
            <CardContent>
              <Box sx={{ display: 'flex', alignItems: 'center', mb: 1 }}>
                <ShowChart color="primary" sx={{ mr: 1 }} />
                <Typography variant="subtitle2" color="textSecondary">
                  Win Rate
                </Typography>
              </Box>
              <Typography variant="h5" component="div">
                {formatPercentage(performance.winRate * 100)}
              </Typography>
              <Typography variant="body2" color="textSecondary">
                Max DD: {formatPercentage(performance.maxDrawdown)}
              </Typography>
            </CardContent>
          </Card>
        </Grid>
      </Grid>

      {/* Main Content Row */}
      <Grid container spacing={3}>
        {/* Portfolio Chart */}
        <Grid item xs={12} lg={8}>
          <Card sx={{ height: 400 }}>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                Portfolio Performance
              </Typography>
              <PortfolioChart />
            </CardContent>
          </Card>
        </Grid>

        {/* Trading Signals */}
        <Grid item xs={12} lg={4}>
          <Card sx={{ height: 400 }}>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                Trading Signals
              </Typography>
              <TradingSignalsWidget />
            </CardContent>
          </Card>
        </Grid>

        {/* Performance Metrics */}
        <Grid item xs={12} md={6}>
          <Card>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                Performance Analytics
              </Typography>
              <PerformanceMetrics />
            </CardContent>
          </Card>
        </Grid>

        {/* Current Positions */}
        <Grid item xs={12} md={6}>
          <Card>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                Current Positions
              </Typography>
              <PositionsTable positions={positions} />
            </CardContent>
          </Card>
        </Grid>
      </Grid>
    </Box>
  )
}

export default DashboardPage