import React, { useEffect, useRef } from 'react'
import { Box, useTheme } from '@mui/material'
import { useSelector } from 'react-redux'
import { Line } from 'react-chartjs-2'
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend,
  Filler,
} from 'chart.js'

import { RootState } from '../../store/store'

ChartJS.register(
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend,
  Filler
)

export const PortfolioChart: React.FC = () => {
  const theme = useTheme()
  const chartRef = useRef<ChartJS<'line'>>(null)
  const { totalValue, dailyPnl } = useSelector((state: RootState) => state.portfolio)

  // Mock historical data - in real app, this would come from API
  const generateMockData = () => {
    const data = []
    const labels = []
    let baseValue = 100000
    
    for (let i = 30; i >= 0; i--) {
      const date = new Date()
      date.setDate(date.getDate() - i)
      labels.push(date.toLocaleDateString())
      
      // Simulate portfolio growth with some volatility
      baseValue += (Math.random() - 0.45) * 2000 + 100
      data.push(baseValue)
    }
    
    return { labels, data }
  }

  const { labels, data: portfolioData } = generateMockData()

  const chartData = {
    labels,
    datasets: [
      {
        label: 'Portfolio Value',
        data: portfolioData,
        borderColor: theme.palette.primary.main,
        backgroundColor: `${theme.palette.primary.main}20`,
        fill: true,
        tension: 0.4,
        pointRadius: 0,
        pointHoverRadius: 6,
        borderWidth: 2,
      },
    ],
  }

  const options = {
    responsive: true,
    maintainAspectRatio: false,
    plugins: {
      legend: {
        display: false,
      },
      tooltip: {
        mode: 'index' as const,
        intersect: false,
        backgroundColor: theme.palette.background.paper,
        titleColor: theme.palette.text.primary,
        bodyColor: theme.palette.text.primary,
        borderColor: theme.palette.divider,
        borderWidth: 1,
        cornerRadius: 8,
        displayColors: false,
        callbacks: {
          label: (context: any) => {
            const value = context.parsed.y
            return `Portfolio Value: $${value.toLocaleString()}`
          },
        },
      },
    },
    scales: {
      x: {
        display: true,
        grid: {
          display: false,
        },
        ticks: {
          color: theme.palette.text.secondary,
          maxTicksLimit: 7,
        },
      },
      y: {
        display: true,
        position: 'right' as const,
        grid: {
          color: `${theme.palette.text.secondary}10`,
        },
        ticks: {
          color: theme.palette.text.secondary,
          callback: (value: any) => `$${(value / 1000).toFixed(0)}k`,
        },
      },
    },
    elements: {
      point: {
        hoverBackgroundColor: theme.palette.primary.main,
        hoverBorderColor: theme.palette.common.white,
        hoverBorderWidth: 2,
      },
    },
    interaction: {
      mode: 'nearest' as const,
      axis: 'x' as const,
      intersect: false,
    },
  }

  return (
    <Box sx={{ height: '300px', width: '100%' }}>
      <Line ref={chartRef} data={chartData} options={options} />
    </Box>
  )
}