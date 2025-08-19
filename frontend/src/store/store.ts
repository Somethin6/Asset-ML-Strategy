import { configureStore } from '@reduxjs/toolkit'
import authSlice from './slices/authSlice'
import appSlice from './slices/appSlice'
import portfolioSlice from './slices/portfolioSlice'
import tradingSlice from './slices/tradingSlice'
import marketDataSlice from './slices/marketDataSlice'

export const store = configureStore({
  reducer: {
    auth: authSlice,
    app: appSlice,
    portfolio: portfolioSlice,
    trading: tradingSlice,
    marketData: marketDataSlice,
  },
  middleware: (getDefaultMiddleware) =>
    getDefaultMiddleware({
      serializableCheck: {
        ignoredActions: ['persist/PERSIST', 'persist/REHYDRATE'],
      },
    }),
  devTools: process.env.NODE_ENV !== 'production',
})

export type RootState = ReturnType<typeof store.getState>
export type AppDispatch = typeof store.dispatch