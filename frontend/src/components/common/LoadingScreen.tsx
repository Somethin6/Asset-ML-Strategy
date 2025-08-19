import React from 'react'
import { 
  Box, 
  CircularProgress, 
  Typography, 
  Paper,
  useTheme
} from '@mui/material'

export const LoadingScreen: React.FC = () => {
  const theme = useTheme()

  return (
    <Box
      sx={{
        position: 'fixed',
        top: 0,
        left: 0,
        right: 0,
        bottom: 0,
        display: 'flex',
        flexDirection: 'column',
        alignItems: 'center',
        justifyContent: 'center',
        backgroundColor: theme.palette.background.default,
        zIndex: 9999,
      }}
    >
      <Paper
        elevation={0}
        sx={{
          p: 4,
          display: 'flex',
          flexDirection: 'column',
          alignItems: 'center',
          gap: 3,
          backgroundColor: 'transparent',
        }}
      >
        <Box sx={{ position: 'relative' }}>
          <CircularProgress
            size={60}
            thickness={4}
            sx={{
              color: theme.palette.primary.main,
            }}
          />
          <Typography
            variant="h6"
            sx={{
              position: 'absolute',
              top: '50%',
              left: '50%',
              transform: 'translate(-50%, -50%)',
              color: theme.palette.primary.main,
              fontSize: '12px',
              fontWeight: 'bold',
            }}
          >
            💰
          </Typography>
        </Box>
        
        <Box sx={{ textAlign: 'center' }}>
          <Typography variant="h6" color="primary" gutterBottom>
            Asset-ML-Strategy
          </Typography>
          <Typography variant="body2" color="textSecondary">
            Loading your trading platform...
          </Typography>
        </Box>
      </Paper>
    </Box>
  )
}