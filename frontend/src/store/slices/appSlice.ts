import { createSlice, createAsyncThunk, PayloadAction } from '@reduxjs/toolkit'

interface AppState {
  isInitialized: boolean
  isLoading: boolean
  theme: 'dark' | 'light'
  notifications: Array<{
    id: string
    type: 'success' | 'error' | 'warning' | 'info'
    message: string
    timestamp: number
  }>
  systemStatus: {
    api: 'healthy' | 'unhealthy' | 'unknown'
    database: 'healthy' | 'unhealthy' | 'unknown'
    trading: 'active' | 'inactive' | 'unknown'
  }
}

const initialState: AppState = {
  isInitialized: false,
  isLoading: false,
  theme: 'dark',
  notifications: [],
  systemStatus: {
    api: 'unknown',
    database: 'unknown',
    trading: 'unknown',
  },
}

// Async thunks
export const initializeApp = createAsyncThunk(
  'app/initialize',
  async () => {
    try {
      // Check system health
      const response = await fetch('/api/health')
      const healthData = await response.json()
      
      return {
        systemHealth: healthData,
      }
    } catch (error) {
      throw new Error('Failed to initialize application')
    }
  }
)

export const checkSystemHealth = createAsyncThunk(
  'app/checkHealth',
  async () => {
    const response = await fetch('/api/health')
    return await response.json()
  }
)

const appSlice = createSlice({
  name: 'app',
  initialState,
  reducers: {
    setLoading: (state, action: PayloadAction<boolean>) => {
      state.isLoading = action.payload
    },
    setTheme: (state, action: PayloadAction<'dark' | 'light'>) => {
      state.theme = action.payload
    },
    addNotification: (state, action: PayloadAction<Omit<AppState['notifications'][0], 'id' | 'timestamp'>>) => {
      state.notifications.push({
        ...action.payload,
        id: Math.random().toString(36).substr(2, 9),
        timestamp: Date.now(),
      })
    },
    removeNotification: (state, action: PayloadAction<string>) => {
      state.notifications = state.notifications.filter(n => n.id !== action.payload)
    },
    clearNotifications: (state) => {
      state.notifications = []
    },
  },
  extraReducers: (builder) => {
    builder
      .addCase(initializeApp.pending, (state) => {
        state.isLoading = true
      })
      .addCase(initializeApp.fulfilled, (state, action) => {
        state.isInitialized = true
        state.isLoading = false
        state.systemStatus.api = 'healthy'
      })
      .addCase(initializeApp.rejected, (state) => {
        state.isInitialized = true
        state.isLoading = false
        state.systemStatus.api = 'unhealthy'
      })
      .addCase(checkSystemHealth.fulfilled, (state, action) => {
        state.systemStatus.api = action.payload.status === 'healthy' ? 'healthy' : 'unhealthy'
      })
  },
})

export const {
  setLoading,
  setTheme,
  addNotification,
  removeNotification,
  clearNotifications,
} = appSlice.actions

export default appSlice.reducer