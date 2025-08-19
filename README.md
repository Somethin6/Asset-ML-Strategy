# Asset ML Strategy

A **free, local** machine learning application for financial Excel data analysis with a perfect GUI interface.

## Features

- ✅ **Excel File Support**: Process Excel files with financial data columns
- ✅ **Required Columns**: Date, Open, High, Low, Close, Adj Close, Volume
- ✅ **Machine Learning**: Random Forest regression for price prediction
- ✅ **Perfect GUI**: Modern tkinter interface with tabs and charts
- ✅ **Completely Free**: No paid services or cloud dependencies
- ✅ **100% Local**: All processing happens on your machine
- ✅ **Visualizations**: Price charts, volume analysis, correlation matrices
- ✅ **Technical Indicators**: SMA, volatility, price changes

## Requirements

- Python 3.7+
- Required packages (install via `pip install -r requirements.txt`):
  - pandas
  - numpy
  - scikit-learn
  - openpyxl
  - matplotlib
  - seaborn
  - tkinter (usually comes with Python)

## Installation

1. Clone or download this repository
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the application:
   ```bash
   python asset_ml_strategy.py
   ```

## Usage

### GUI Application
1. **Launch**: Run `python asset_ml_strategy.py`
2. **Load Data**: Click "Select Excel File" and choose your Excel file
3. **Analyze**: Use the ML Analysis tab to train models and make predictions
4. **Visualize**: View charts in the Visualization tab

### Excel File Format
Your Excel file must contain these columns:
```
Date | Open | High | Low | Close | Adj Close | Volume
```

Example:
```
2023-01-01,150.25,155.80,149.50,154.30,154.30,1250000
2023-01-02,154.50,158.20,153.10,157.85,157.85,1180000
```

### Sample Data
A sample Excel file (`sample_data.xlsx`) is included for testing.

## Features in Detail

### Data Loading Tab
- Select Excel files with financial data
- Automatic validation of required columns
- Data preview with scrollable table
- Statistical information display

### ML Analysis Tab
- Configure target variable (Close, High, Low, etc.)
- Adjustable train/test split ratio
- Random Forest model training
- Performance metrics (R², MSE)
- Feature importance analysis
- Technical indicators (SMA, volatility)

### Visualization Tab
- **Price Chart**: OHLC candlestick-style visualization
- **Volume Chart**: Trading volume analysis
- **Correlation Matrix**: Relationships between variables
- **Predictions vs Actual**: Model performance visualization

## Technical Details

### Machine Learning
- **Algorithm**: Random Forest Regression
- **Features**: OHLC prices, volume, technical indicators
- **Indicators**: Simple Moving Averages (5, 10, 20 periods), Volatility, Price Changes
- **Validation**: Train/test split with customizable ratio

### GUI Framework
- **Framework**: tkinter (included with Python)
- **Layout**: Tabbed interface with professional styling
- **Charts**: Matplotlib integration for interactive charts
- **Responsiveness**: Resizable windows and scrollable content

## Testing

Run the test script to verify functionality:
```bash
python test_functionality.py
```

This will test:
- Excel data loading
- ML model training
- Visualization generation
- GUI module availability

## Troubleshooting

### Common Issues
1. **"No module named 'tkinter'"**: Install python3-tk package
   ```bash
   sudo apt-get install python3-tk  # Ubuntu/Debian
   brew install python-tk           # macOS
   ```

2. **Excel file not loading**: Ensure your file has the required columns exactly as specified

3. **Charts not displaying**: Make sure matplotlib is properly installed

### System Requirements
- **OS**: Windows, macOS, Linux
- **Memory**: 512MB RAM minimum
- **Storage**: 100MB free space
- **Display**: GUI requires display server

## License

This project is completely free and open source. Use it for any purpose without restrictions.

## Contributing

Feel free to submit issues, feature requests, or pull requests to improve the application.

---

**Key Benefits:**
- 🆓 **Completely Free** - No subscriptions or paid features
- 💻 **100% Local** - Your data never leaves your computer
- 🎯 **Perfect GUI** - Professional interface designed for ease of use
- 📊 **Comprehensive Analysis** - ML predictions with detailed visualizations
- 🔒 **Privacy First** - All processing happens locally on your machine