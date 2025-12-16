# Crypto Analytics Platform

Real-time cryptocurrency analytics platform with WebSocket data ingestion, statistical analysis, and professional dark UI.

## Features

### Real-Time Data Ingestion
- **WebSocket Integration**: Live price feeds from Binance for BTC/USDT and ETH/USDT
- **Automatic Resampling**: OHLCV data aggregation at multiple timeframes (1s, 1m, 5m, 15m, 30m, 1h)
- **SQLite Storage**: Efficient tick-by-tick data storage with indexed queries

### Analytics Capabilities
- **Price Charts**: Real-time candlestick charts with volume and trade count
- **Pair Analytics**: Hedge ratio calculation (OLS, Huber, Kalman Filter)
- **Spread Trading**: Cointegration analysis and z-score calculations
- **Rolling Statistics**: Volatility, correlation, and momentum indicators
- **Stationarity Testing**: Augmented Dickey-Fuller (ADF) test
- **Alert System**: Z-score and price change threshold alerts

### Professional Features
- **Clean Dark UI**: Professional fintech-style interface
- **OHLCV Upload**: Manual CSV data upload support
- **Data Export**: Download analytics and statistics
- **Auto-refresh**: Configurable dashboard refresh intervals

## Quick Start

### Prerequisites
```bash
Python 3.8+
pip (Python package manager)
```

### Installation

1. **Clone/Download the repository**
```bash
cd crypto_analysis_platform
```

2. **Create virtual environment (recommended)**
```bash
python -m venv venv
# Windows
venv\Scripts\activate
# Linux/Mac
source venv/bin/activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Run the application**
```bash
python app.py
```

The application will:
- Start WebSocket data collector in the background
- Launch the Streamlit dashboard
- Automatically open your browser to http://localhost:8501

## Project Structure

```
crypto_analysis_platform/
├── app.py                 # Main launcher
├── src/
│   ├── data_collector.py  # WebSocket data ingestion
│   ├── storage.py         # SQLite database layer
│   ├── analytics.py       # Statistical analysis engine
│   ├── dashboard.py       # Streamlit frontend
│   └── config.py          # Configuration settings
├── data/                  # SQLite database storage
├── export/                # Exported data files
├── tests/                 # Verification scripts
└── requirements.txt       # Python dependencies
```

## Methodology

### Data Pipeline

1. **Ingestion**: WebSocket connection to Binance streams real-time trade data
2. **Storage**: Tick data stored in SQLite with timestamp indexing
3. **Aggregation**: Real-time OHLCV calculation using pandas resampling
4. **Analytics**: Statistical computations on resampled data
5. **Visualization**: Plotly charts rendered in Streamlit dashboard
6. **Alerts**: Threshold monitoring with real-time notifications

### Analytics Explanation

#### Hedge Ratio Calculation
Three methods available for pairs trading:

| Method | Description | Use Case |
|--------|-------------|----------|
| **OLS** | Ordinary Least Squares regression | Static beta estimation |
| **Huber** | Robust regression resistant to outliers | Noisy data |
| **Kalman** | Adaptive filter tracking time-varying relationships | Dynamic hedging |

#### Spread & Z-Score
```python
spread = price1 - (beta * price2)
z_score = (spread - mean(spread)) / std(spread)
```
Trading signals generated when z-score exceeds thresholds (typically ±2σ).

#### Stationarity Testing
Augmented Dickey-Fuller test validates mean-reversion assumption for spread trading.

### Alert System

| Alert Type | Condition | Action |
|------------|-----------|--------|
| Z-Score | abs(z_score) > threshold | Display warning banner |
| Price Change | change_pct > threshold | Real-time notification |

## Configuration

Edit `src/config.py` to customize:
- Trading symbols
- Timeframe intervals
- Window sizes for rolling calculations
- Alert thresholds
- Chart display limits

## Data Upload

Upload historical OHLCV data via the sidebar:
1. Prepare CSV with columns: `timestamp, open, high, low, close, volume`
2. Click "Upload OHLCV CSV"
3. Enter custom symbol name
4. Click "Process Upload"

**Note**: Application works fully without uploads using live WebSocket data.

## Technical Stack

- **Frontend**: Streamlit, Plotly
- **Backend**: Python 3.10+
- **Database**: SQLite3
- **WebSocket**: websockets library
- **Analytics**: pandas, numpy, scipy, statsmodels, pykalman

## Dependencies

See `requirements.txt` for complete list:
- streamlit - Dashboard framework
- plotly - Interactive charts
- pandas, numpy - Data manipulation
- websockets - Real-time data feeds
- statsmodels - Statistical tests
- pykalman - Kalman filtering
- scikit-learn - Robust regression

## Troubleshooting

**Issue**: WebSocket connection fails
**Solution**: Check internet connection and Binance API availability

**Issue**: Charts not rendering
**Solution**: Ensure sufficient data collected (wait 1-2 minutes after start)

**Issue**: Import errors
**Solution**: Run `pip install -r requirements.txt` again

## License

MIT License - Free to use and modify

## Author

Created for GEMSCAP assignment submission
