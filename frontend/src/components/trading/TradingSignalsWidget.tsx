import React, { useState, useEffect } from 'react'
import {
  Box,
  List,
  ListItem,
  ListItemText,
  Chip,
  Typography,
  CircularProgress,
  Alert,
  Divider
} from '@mui/material'
import { TrendingUp, TrendingDown, Remove } from '@mui/icons-material'

interface TradingSignal {
  symbol: string
  action: 'BUY' | 'SELL' | 'HOLD'
  confidence: number
  price: number
  timestamp: string
}

export const TradingSignalsWidget: React.FC = () => {
  const [signals, setSignals] = useState<TradingSignal[]>([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)

  const fetchSignals = async () => {
    try {
      setLoading(true)
      
      // Fetch signals for popular symbols
      const symbols = ['AAPL', 'GOOGL', 'MSFT', 'TSLA', 'AMZN']
      const signalPromises = symbols.map(async (symbol) => {
        const response = await fetch(`/api/trading/signals/${symbol}`)
        if (!response.ok) throw new Error(`Failed to fetch signal for ${symbol}`)
        return await response.json()
      })

      const signalResponses = await Promise.all(signalPromises)
      const fetchedSignals = signalResponses.map(response => ({
        symbol: response.symbol,
        action: response.action,
        confidence: response.confidence,
        price: response.price,
        timestamp: response.timestamp
      }))

      setSignals(fetchedSignals)
      setError(null)
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to fetch signals')
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => {
    fetchSignals()
    
    // Refresh signals every 60 seconds
    const interval = setInterval(fetchSignals, 60000)
    return () => clearInterval(interval)
  }, [])

  const getSignalIcon = (action: string) => {
    switch (action) {
      case 'BUY':
        return <TrendingUp color="success" />
      case 'SELL':
        return <TrendingDown color="error" />
      default:
        return <Remove color="disabled" />
    }
  }

  const getSignalColor = (action: string) => {
    switch (action) {
      case 'BUY':
        return 'success'
      case 'SELL':
        return 'error'
      default:
        return 'default'
    }
  }

  const formatTime = (timestamp: string) => {
    return new Date(timestamp).toLocaleTimeString()
  }

  if (loading) {
    return (
      <Box sx={{ display: 'flex', justifyContent: 'center', alignItems: 'center', height: 200 }}>
        <CircularProgress />
      </Box>
    )
  }

  if (error) {
    return (
      <Alert severity="error">
        {error}
      </Alert>
    )
  }

  return (
    <Box sx={{ height: '300px', overflow: 'auto' }}>
      <List dense>
        {signals.map((signal, index) => (
          <React.Fragment key={`${signal.symbol}-${index}`}>
            <ListItem
              sx={{
                borderRadius: 1,
                mb: 1,
                backgroundColor: 'rgba(255,255,255,0.02)',
                '&:hover': {
                  backgroundColor: 'rgba(255,255,255,0.05)',
                },
              }}
            >
              <Box sx={{ display: 'flex', alignItems: 'center', mr: 2 }}>
                {getSignalIcon(signal.action)}
              </Box>
              
              <ListItemText
                primary={
                  <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                    <Typography variant="subtitle2" component="span">
                      {signal.symbol}
                    </Typography>
                    <Chip
                      label={signal.action}
                      size="small"
                      color={getSignalColor(signal.action) as any}
                      variant="outlined"
                    />
                  </Box>
                }
                secondary={
                  <Box sx={{ display: 'flex', flexDirection: 'column', gap: 0.5, mt: 0.5 }}>
                    <Typography variant="body2" color="textSecondary">
                      Price: ${signal.price.toFixed(2)}
                    </Typography>
                    <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                      <Typography variant="caption" color="textSecondary">
                        Confidence: {(signal.confidence * 100).toFixed(1)}%
                      </Typography>
                      <Typography variant="caption" color="textSecondary">
                        {formatTime(signal.timestamp)}
                      </Typography>
                    </Box>
                  </Box>
                }
              />
            </ListItem>
            {index < signals.length - 1 && <Divider />}
          </React.Fragment>
        ))}
      </List>

      {signals.length === 0 && (
        <Box sx={{ textAlign: 'center', mt: 4 }}>
          <Typography variant="body2" color="textSecondary">
            No trading signals available
          </Typography>
        </Box>
      )}
    </Box>
  )
}