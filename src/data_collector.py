"""
WebSocket Data Collector
Ingests real-time tick data from Binance Futures and stores in SQLite
"""

import json
import time
import threading
from datetime import datetime
from websocket import WebSocketApp
from storage import DataStorage
from analytics import AnalyticsEngine
from config import Config

class BinanceCollector:
    def __init__(self, symbols):
        self.symbols = [s.lower().strip() for s in symbols]
        self.storage = DataStorage()
        self.ws_connections = []
        self.running = False
        self.tick_count = 0
        self.last_log = time.time()
        
    def normalize_tick(self, data):
        """Normalize Binance tick data to standard format"""
        return {
            'timestamp': datetime.fromtimestamp(data['T'] / 1000).isoformat(),
            'symbol': data['s'],
            'price': float(data['p']),
            'size': float(data['q'])
        }
    
    def on_message(self, ws, message):
        """Handle incoming WebSocket message"""
        try:
            data = json.loads(message)
            if data.get('e') == 'trade':
                tick = self.normalize_tick(data)
                self.storage.insert_tick(tick)
                self.tick_count += 1
                
                # Log stats every 5 seconds
                if time.time() - self.last_log > 5:
                    print(f"[Collector] Ticks received: {self.tick_count}")
                    self.last_log = time.time()
        except Exception as e:
            print(f"Error processing message: {e}")
    
    def on_error(self, ws, error):
        print(f"WebSocket error: {error}")
    
    def on_close(self, ws, close_status_code, close_msg):
        print(f"WebSocket closed: {close_status_code}")
    
    def on_open(self, ws):
        symbol = ws.url.split('/')[-1].split('@')[0]
        print(f"WebSocket connected: {symbol}")
    
    def start(self):
        """Start collecting data for all symbols"""
        self.running = True
        print(f"Starting data collection for: {', '.join(self.symbols)}")
        
        for symbol in self.symbols:
            url = f"wss://fstream.binance.com/ws/{symbol}@trade"
            ws = WebSocketApp(
                url,
                on_message=self.on_message,
                on_error=self.on_error,
                on_close=self.on_close,
                on_open=self.on_open
            )
            
            thread = threading.Thread(target=ws.run_forever, daemon=True)
            thread.start()
            self.ws_connections.append(ws)
        
        # Keep main thread alive
        try:
            while self.running:
                time.sleep(1)
        except KeyboardInterrupt:
            self.stop()
    
    def stop(self):
        """Stop all WebSocket connections"""
        self.running = False
        for ws in self.ws_connections:
            ws.close()
        print("Data collector stopped")

if __name__ == "__main__":
    collector = BinanceCollector(Config.DEFAULT_SYMBOLS)
    collector.start()