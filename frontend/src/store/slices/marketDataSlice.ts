import { createSlice } from '@reduxjs/toolkit'

// This is a placeholder slice for market data
const marketDataSlice = createSlice({
  name: 'marketData',
  initialState: {
    symbols: [],
    prices: {},
    isLoading: false,
  },
  reducers: {
    updatePrice: (state, action) => {
      const { symbol, price } = action.payload
      state.prices[symbol] = price
    },
  },
})

export const { updatePrice } = marketDataSlice.actions
export default marketDataSlice.reducer