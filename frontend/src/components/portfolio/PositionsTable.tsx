import React from 'react'
import {
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  Chip,
  Typography,
  Box
} from '@mui/material'

interface Position {
  symbol: string
  quantity: number
  avgPrice: number
  currentPrice: number
  unrealizedPnl: number
  marketValue: number
  percentage: number
}

interface PositionsTableProps {
  positions: Position[]
}

export const PositionsTable: React.FC<PositionsTableProps> = ({ positions }) => {
  const formatCurrency = (value: number) =>
    new Intl.NumberFormat('en-US', { 
      style: 'currency', 
      currency: 'USD' 
    }).format(value)

  const formatPercentage = (value: number) =>
    `${value > 0 ? '+' : ''}${value.toFixed(2)}%`

  if (positions.length === 0) {
    return (
      <Box sx={{ textAlign: 'center', py: 4 }}>
        <Typography variant="body2" color="textSecondary">
          No positions currently held
        </Typography>
      </Box>
    )
  }

  return (
    <TableContainer sx={{ maxHeight: 300 }}>
      <Table size="small" stickyHeader>
        <TableHead>
          <TableRow>
            <TableCell>Symbol</TableCell>
            <TableCell align="right">Qty</TableCell>
            <TableCell align="right">Avg Price</TableCell>
            <TableCell align="right">Current</TableCell>
            <TableCell align="right">P&L</TableCell>
          </TableRow>
        </TableHead>
        <TableBody>
          {positions.map((position) => (
            <TableRow 
              key={position.symbol}
              sx={{ '&:hover': { backgroundColor: 'rgba(255,255,255,0.02)' } }}
            >
              <TableCell>
                <Typography variant="subtitle2" component="div">
                  {position.symbol}
                </Typography>
                <Typography variant="caption" color="textSecondary">
                  {formatPercentage(position.percentage)} of portfolio
                </Typography>
              </TableCell>
              <TableCell align="right">
                <Typography variant="body2">
                  {position.quantity.toLocaleString()}
                </Typography>
              </TableCell>
              <TableCell align="right">
                <Typography variant="body2">
                  {formatCurrency(position.avgPrice)}
                </Typography>
              </TableCell>
              <TableCell align="right">
                <Typography variant="body2">
                  {formatCurrency(position.currentPrice)}
                </Typography>
              </TableCell>
              <TableCell align="right">
                <Chip
                  label={formatCurrency(position.unrealizedPnl)}
                  size="small"
                  color={position.unrealizedPnl >= 0 ? 'success' : 'error'}
                  variant="outlined"
                />
              </TableCell>
            </TableRow>
          ))}
        </TableBody>
      </Table>
    </TableContainer>
  )
}