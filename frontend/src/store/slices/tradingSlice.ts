import { createSlice } from '@reduxjs/toolkit'

// This is a placeholder slice for trading state
const tradingSlice = createSlice({
  name: 'trading',
  initialState: {
    activeStrategies: [],
    signals: [],
    isTrading: false,
  },
  reducers: {
    setTradingStatus: (state, action) => {
      state.isTrading = action.payload
    },
  },
})

export const { setTradingStatus } = tradingSlice.actions
export default tradingSlice.reducer