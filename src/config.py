"""
Configuration Module
Centralized configuration for the application
"""

class Config:
    # Default symbols to track
    DEFAULT_SYMBOLS = ['btcusdt', 'ethusdt']
    
    # Database
    DB_PATH = "data/market_data.db"
    
    # Resampling timeframes
    TIMEFRAMES = {
        '1s': '1S',
        '1m': '1T',
        '5m': '5T',
        '15m': '15T',
        '1h': '1H'
    }
    
    # Analytics defaults
    DEFAULT_WINDOW = 20
    DEFAULT_ZSCORE_THRESHOLD = 2.0
    DEFAULT_HEDGE_METHOD = 'ols'
    
    # Alert thresholds
    ALERT_ZSCORE_HIGH = 2.5
    ALERT_ZSCORE_LOW = -2.5
    ALERT_PRICE_CHANGE = 5.0  # percentage
    
    # UI refresh rates (seconds)
    TICK_UPDATE_INTERVAL = 0.5
    ANALYTICS_UPDATE_INTERVAL = 5
    CHART_UPDATE_INTERVAL = 2
    
    # Data limits
    MAX_TICKS_DISPLAY = 10000
    MAX_CHART_POINTS = 1000
    
    # Export formats
    EXPORT_FORMATS = ['csv', 'json', 'parquet']