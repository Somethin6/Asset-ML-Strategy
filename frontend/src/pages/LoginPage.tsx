import React from 'react'
import { Typography, Box } from '@mui/material'

const LoginPage: React.FC = () => {
  return (
    <Box sx={{ p: 3, textAlign: 'center', mt: 8 }}>
      <Typography variant="h4" gutterBottom>
        Login Page
      </Typography>
      <Typography variant="body1" color="textSecondary">
        Authentication form will be implemented here
      </Typography>
    </Box>
  )
}

export { LoginPage }