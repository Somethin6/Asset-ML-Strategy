import { createSlice, createAsyncThunk, PayloadAction } from '@reduxjs/toolkit'

interface User {
  id: string
  email: string
  fullName: string
  role: 'admin' | 'trader' | 'viewer'
  subscriptionTier: string
  isVerified: boolean
}

interface AuthState {
  isAuthenticated: boolean
  user: User | null
  token: string | null
  isLoading: boolean
  error: string | null
}

const initialState: AuthState = {
  isAuthenticated: false,
  user: null,
  token: localStorage.getItem('token'),
  isLoading: false,
  error: null,
}

// Check if token exists on app start
if (initialState.token) {
  initialState.isAuthenticated = true
  // In a real app, you'd validate the token with the server
}

export const login = createAsyncThunk(
  'auth/login',
  async ({ email, password }: { email: string; password: string }) => {
    const response = await fetch('/api/auth/login', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ email, password }),
    })
    
    if (!response.ok) {
      throw new Error('Login failed')
    }
    
    const data = await response.json()
    
    // Store token
    localStorage.setItem('token', data.access_token)
    
    return data
  }
)

export const register = createAsyncThunk(
  'auth/register',
  async ({ email, password, fullName }: { email: string; password: string; fullName: string }) => {
    const response = await fetch('/api/auth/register', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ email, password, full_name: fullName }),
    })
    
    if (!response.ok) {
      throw new Error('Registration failed')
    }
    
    const data = await response.json()
    
    // Store token
    localStorage.setItem('token', data.access_token)
    
    return data
  }
)

export const logout = createAsyncThunk('auth/logout', async () => {
  localStorage.removeItem('token')
  return true
})

export const fetchUserProfile = createAsyncThunk(
  'auth/fetchProfile',
  async (_, { getState }) => {
    const state = getState() as { auth: AuthState }
    const token = state.auth.token
    
    const response = await fetch('/api/auth/profile', {
      headers: {
        Authorization: `Bearer ${token}`,
      },
    })
    
    if (!response.ok) {
      throw new Error('Failed to fetch profile')
    }
    
    return await response.json()
  }
)

const authSlice = createSlice({
  name: 'auth',
  initialState,
  reducers: {
    clearError: (state) => {
      state.error = null
    },
    setUser: (state, action: PayloadAction<User>) => {
      state.user = action.payload
    },
  },
  extraReducers: (builder) => {
    builder
      // Login
      .addCase(login.pending, (state) => {
        state.isLoading = true
        state.error = null
      })
      .addCase(login.fulfilled, (state, action) => {
        state.isLoading = false
        state.isAuthenticated = true
        state.token = action.payload.access_token
        // Mock user data for development
        state.user = {
          id: '1',
          email: 'user@example.com',
          fullName: 'Demo User',
          role: 'trader',
          subscriptionTier: 'premium',
          isVerified: true,
        }
      })
      .addCase(login.rejected, (state, action) => {
        state.isLoading = false
        state.error = action.error.message || 'Login failed'
      })
      // Register
      .addCase(register.pending, (state) => {
        state.isLoading = true
        state.error = null
      })
      .addCase(register.fulfilled, (state, action) => {
        state.isLoading = false
        state.isAuthenticated = true
        state.token = action.payload.access_token
        // Mock user data for development
        state.user = {
          id: '1',
          email: 'user@example.com',
          fullName: 'New User',
          role: 'trader',
          subscriptionTier: 'basic',
          isVerified: false,
        }
      })
      .addCase(register.rejected, (state, action) => {
        state.isLoading = false
        state.error = action.error.message || 'Registration failed'
      })
      // Logout
      .addCase(logout.fulfilled, (state) => {
        state.isAuthenticated = false
        state.user = null
        state.token = null
        state.error = null
      })
      // Fetch profile
      .addCase(fetchUserProfile.fulfilled, (state, action) => {
        state.user = action.payload
      })
  },
})

export const { clearError, setUser } = authSlice.actions
export default authSlice.reducer