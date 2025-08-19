import React from 'react'
import {
  Box,
  Grid,
  Typography,
  LinearProgress,
  Chip,
} from '@mui/material'
import { useSelector } from 'react-redux'
import { RootState } from '../../store/store'

export const PerformanceMetrics: React.FC = () => {
  const { performance } = useSelector((state: RootState) => state.portfolio)

  const formatPercentage = (value: number) =>
    `${value > 0 ? '+' : ''}${value.toFixed(2)}%`

  const metrics = [
    {
      label: 'Total Return',
      value: performance.totalReturn,
      format: formatPercentage,
      color: performance.totalReturn >= 0 ? 'success' : 'error',
    },
    {
      label: 'Sharpe Ratio',
      value: performance.sharpeRatio,
      format: (val: number) => val.toFixed(2),
      color: performance.sharpeRatio >= 1 ? 'success' : performance.sharpeRatio >= 0 ? 'warning' : 'error',
    },
    {
      label: 'Max Drawdown',
      value: Math.abs(performance.maxDrawdown),
      format: (val: number) => `-${val.toFixed(2)}%`,
      color: performance.maxDrawdown > -10 ? 'success' : performance.maxDrawdown > -20 ? 'warning' : 'error',
    },
    {
      label: 'Win Rate',
      value: performance.winRate * 100,
      format: (val: number) => `${val.toFixed(1)}%`,
      color: performance.winRate >= 0.6 ? 'success' : performance.winRate >= 0.4 ? 'warning' : 'error',
    },
  ]

  return (
    <Box>
      <Grid container spacing={3}>
        {metrics.map((metric, index) => (
          <Grid item xs={12} sm={6} key={index}>
            <Box sx={{ mb: 2 }}>
              <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 1 }}>
                <Typography variant="body2" color="textSecondary">
                  {metric.label}
                </Typography>
                <Chip
                  label={metric.format(metric.value)}
                  size="small"
                  color={metric.color as any}
                  variant="outlined"
                />
              </Box>
              
              {/* Progress bar for visual representation */}
              {metric.label === 'Win Rate' && (
                <LinearProgress
                  variant="determinate"
                  value={metric.value}
                  color={metric.color as any}
                  sx={{ height: 6, borderRadius: 3 }}
                />
              )}
              
              {metric.label === 'Sharpe Ratio' && (
                <LinearProgress
                  variant="determinate"
                  value={Math.min((Math.max(metric.value, 0) / 4) * 100, 100)}
                  color={metric.color as any}
                  sx={{ height: 6, borderRadius: 3 }}
                />
              )}
            </Box>
          </Grid>
        ))}
      </Grid>

      {/* Additional Risk Metrics */}
      <Box sx={{ mt: 3 }}>
        <Typography variant="subtitle2" gutterBottom>
          Risk Assessment
        </Typography>
        <Box sx={{ display: 'flex', gap: 1, flexWrap: 'wrap' }}>
          <Chip
            label="Low Risk"
            size="small"
            color={performance.maxDrawdown > -5 ? 'success' : 'default'}
            variant={performance.maxDrawdown > -5 ? 'filled' : 'outlined'}
          />
          <Chip
            label="Medium Risk"
            size="small"
            color={performance.maxDrawdown <= -5 && performance.maxDrawdown > -15 ? 'warning' : 'default'}
            variant={performance.maxDrawdown <= -5 && performance.maxDrawdown > -15 ? 'filled' : 'outlined'}
          />
          <Chip
            label="High Risk"
            size="small"
            color={performance.maxDrawdown <= -15 ? 'error' : 'default'}
            variant={performance.maxDrawdown <= -15 ? 'filled' : 'outlined'}
          />
        </Box>
      </Box>
    </Box>
  )
}