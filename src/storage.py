"""
Data Storage Module
Handles SQLite database operations for tick data and resampled data
"""

import sqlite3
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
import threading

class DataStorage:
    def __init__(self, db_path="data/market_data.db"):
        self.db_path = db_path
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self.lock = threading.Lock()
        self._init_db()
    
    def _init_db(self):
        """Initialize database schema"""
        with self.lock:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Enable WAL mode for concurrency
            cursor.execute("PRAGMA journal_mode=WAL;")
            
            # Tick data table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS ticks (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    symbol TEXT NOT NULL,
                    price REAL NOT NULL,
                    size REAL NOT NULL
                )
            """)
            
            # Create index for fast queries
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_ticks_symbol_time 
                ON ticks(symbol, timestamp)
            """)
            
            # Resampled OHLCV data table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS ohlcv (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    symbol TEXT NOT NULL,
                    timeframe TEXT NOT NULL,
                    open REAL NOT NULL,
                    high REAL NOT NULL,
                    low REAL NOT NULL,
                    close REAL NOT NULL,
                    volume REAL NOT NULL,
                    trades INTEGER NOT NULL,
                    UNIQUE(timestamp, symbol, timeframe)
                )
            """)
            
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_ohlcv_symbol_time 
                ON ohlcv(symbol, timestamp, timeframe)
            """)
            
            conn.commit()
            conn.close()
    
    def insert_tick(self, tick):
        """Insert a single tick into database"""
        with self.lock:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute(
                "INSERT INTO ticks (timestamp, symbol, price, size) VALUES (?, ?, ?, ?)",
                (tick['timestamp'], tick['symbol'], tick['price'], tick['size'])
            )
            conn.commit()
            conn.close()
    
    def get_ticks(self, symbol=None, start_time=None, limit=None):
        """Retrieve ticks from database"""
        conn = sqlite3.connect(self.db_path)
        
        query = "SELECT timestamp, symbol, price, size FROM ticks"
        conditions = []
        params = []
        
        if symbol:
            conditions.append("symbol = ?")
            params.append(symbol)
        
        if start_time:
            conditions.append("timestamp >= ?")
            params.append(start_time)
        
        if conditions:
            query += " WHERE " + " AND ".join(conditions)
        
        query += " ORDER BY timestamp DESC"
        
        if limit:
            query += f" LIMIT {limit}"
        
        df = pd.read_sql_query(query, conn, params=params)
        conn.close()
        
        if not df.empty:
            # Fixed: Use format='mixed' and handle errors
            try:
                df['timestamp'] = pd.to_datetime(df['timestamp'], format='mixed', errors='coerce')
            except:
                # Fallback: try without format specification
                df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
            
            # Remove any rows where timestamp conversion failed
            df = df.dropna(subset=['timestamp'])
            df = df.sort_values('timestamp')
        
        return df
    
    def get_symbols(self):
        """Get list of all symbols in database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT DISTINCT symbol FROM ticks")
        symbols = [row[0] for row in cursor.fetchall()]
        conn.close()
        return symbols
    
    def resample_and_store(self, symbol, timeframe='1T'):
        """Resample tick data to OHLCV and store"""
        df = self.get_ticks(symbol)
        
        if df.empty:
            return
        
        df.set_index('timestamp', inplace=True)
        
        ohlcv = df.resample(timeframe).agg({
            'price': ['first', 'max', 'min', 'last', 'count'],
            'size': 'sum'
        })
        
        ohlcv.columns = ['open', 'high', 'low', 'close', 'trades', 'volume']
        ohlcv = ohlcv.dropna()
        
        # Store in database
        with self.lock:
            conn = sqlite3.connect(self.db_path)
            for idx, row in ohlcv.iterrows():
                conn.execute("""
                    INSERT OR REPLACE INTO ohlcv 
                    (timestamp, symbol, timeframe, open, high, low, close, volume, trades)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    idx.isoformat(), symbol, timeframe,
                    row['open'], row['high'], row['low'], row['close'],
                    row['volume'], int(row['trades'])
                ))
            conn.commit()
            conn.close()
    
    def get_ohlcv(self, symbol, timeframe='1T', limit=None):
        """Retrieve OHLCV data"""
        conn = sqlite3.connect(self.db_path)
        
        query = """
            SELECT timestamp, open, high, low, close, volume, trades
            FROM ohlcv
            WHERE symbol = ? AND timeframe = ?
            ORDER BY timestamp DESC
        """
        
        if limit:
            query += f" LIMIT {limit}"
        
        df = pd.read_sql_query(query, conn, params=(symbol, timeframe))
        conn.close()
        
        if not df.empty:
            # Fixed: Use format='mixed' and handle errors
            try:
                df['timestamp'] = pd.to_datetime(df['timestamp'], format='mixed', errors='coerce')
            except:
                # Fallback: try without format specification
                df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
            
            # Remove any rows where timestamp conversion failed
            df = df.dropna(subset=['timestamp'])
            df = df.sort_values('timestamp')
        
        return df
    
    def upload_ohlcv_data(self, df, symbol):
        """Upload external OHLCV data (CSV upload feature)"""
        required_cols = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
        
        if not all(col in df.columns for col in required_cols):
            raise ValueError(f"DataFrame must contain columns: {required_cols}")
        
        # Fixed: Handle timestamp conversion with errors='coerce'
        try:
            df['timestamp'] = pd.to_datetime(df['timestamp'], format='mixed', errors='coerce')
        except:
            df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
        
        # Remove rows with invalid timestamps
        df = df.dropna(subset=['timestamp'])
        
        df['symbol'] = symbol
        df['timeframe'] = 'uploaded'
        
        if 'trades' not in df.columns:
            df['trades'] = 0
        
        with self.lock:
            conn = sqlite3.connect(self.db_path)
            for _, row in df.iterrows():
                conn.execute("""
                    INSERT OR REPLACE INTO ohlcv 
                    (timestamp, symbol, timeframe, open, high, low, close, volume, trades)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    row['timestamp'].isoformat(), symbol, 'uploaded',
                    row['open'], row['high'], row['low'], row['close'],
                    row['volume'], int(row['trades'])
                ))
            conn.commit()
            conn.close()
    
    def get_stats(self):
        """Get database statistics"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("SELECT COUNT(*) FROM ticks")
        tick_count = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(DISTINCT symbol) FROM ticks")
        symbol_count = cursor.fetchone()[0]
        
        cursor.execute("SELECT MIN(timestamp), MAX(timestamp) FROM ticks")
        time_range = cursor.fetchone()
        
        conn.close()
        
        return {
            'total_ticks': tick_count,
            'symbols': symbol_count,
            'earliest': time_range[0],
            'latest': time_range[1]
        }