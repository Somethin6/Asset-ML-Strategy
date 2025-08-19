import React, { useEffect } from 'react'
import { Routes, Route, Navigate } from 'react-router-dom'
import { useDispatch, useSelector } from 'react-redux'
import { Box } from '@mui/material'

import { RootState } from './store/store'
import { initializeApp } from './store/slices/appSlice'
import { ProtectedRoute } from './components/auth/ProtectedRoute'
import { Layout } from './components/layout/Layout'
import { LoadingScreen } from './components/common/LoadingScreen'

// Pages
import { LoginPage } from './pages/LoginPage'
import DashboardPage from './pages/DashboardPage'
import { 
  RegisterPage,
  TradingPage,
  PortfolioPage,
  BacktestPage,
  StrategiesPage,
  AnalyticsPage,
  SettingsPage,
  AdminPage
} from './pages/index'

const App: React.FC = () => {
  const dispatch = useDispatch()
  const { isInitialized, isLoading } = useSelector((state: RootState) => state.app)
  const { isAuthenticated } = useSelector((state: RootState) => state.auth)

  useEffect(() => {
    dispatch(initializeApp())
  }, [dispatch])

  if (!isInitialized || isLoading) {
    return <LoadingScreen />
  }

  return (
    <Box sx={{ display: 'flex', flexDirection: 'column', minHeight: '100vh' }}>
      <Routes>
        {/* Public Routes */}
        <Route
          path="/login"
          element={
            isAuthenticated ? <Navigate to="/dashboard" replace /> : <LoginPage />
          }
        />
        <Route
          path="/register"
          element={
            isAuthenticated ? <Navigate to="/dashboard" replace /> : <RegisterPage />
          }
        />

        {/* Protected Routes */}
        <Route
          path="/*"
          element={
            <ProtectedRoute>
              <Layout>
                <Routes>
                  <Route path="/" element={<Navigate to="/dashboard" replace />} />
                  <Route path="/dashboard" element={<DashboardPage />} />
                  <Route path="/trading" element={<TradingPage />} />
                  <Route path="/portfolio" element={<PortfolioPage />} />
                  <Route path="/backtest" element={<BacktestPage />} />
                  <Route path="/strategies" element={<StrategiesPage />} />
                  <Route path="/analytics" element={<AnalyticsPage />} />
                  <Route path="/settings" element={<SettingsPage />} />
                  <Route
                    path="/admin"
                    element={
                      <ProtectedRoute requiredRole="admin">
                        <AdminPage />
                      </ProtectedRoute>
                    }
                  />
                </Routes>
              </Layout>
            </ProtectedRoute>
          }
        />
      </Routes>
    </Box>
  )
}

export default App