// Placeholder page exports
import React from 'react'
import { Typography, Box } from '@mui/material'

const createPlaceholderPage = (title: string) => {
  return () => (
    <Box sx={{ p: 3 }}>
      <Typography variant="h4" gutterBottom>
        {title}
      </Typography>
      <Typography variant="body1" color="textSecondary">
        This page is under construction. Full implementation coming soon.
      </Typography>
    </Box>
  )
}

export const RegisterPage = createPlaceholderPage('Register')
export const TradingPage = createPlaceholderPage('Trading')  
export const PortfolioPage = createPlaceholderPage('Portfolio')
export const BacktestPage = createPlaceholderPage('Backtest')
export const StrategiesPage = createPlaceholderPage('Strategies')
export const AnalyticsPage = createPlaceholderPage('Analytics')
export const SettingsPage = createPlaceholderPage('Settings')
export const AdminPage = createPlaceholderPage('Admin Panel')