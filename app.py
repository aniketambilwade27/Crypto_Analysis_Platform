"""
Crypto Analytics Platform - Main Launcher
Single-command execution: python app.py
"""

import subprocess
import sys
import time
import os
from pathlib import Path

def main():
    """Launch data collector and dashboard simultaneously"""
    print("=" * 60)
    print("🚀 Crypto Analytics Platform")
    print("=" * 60)
    print()
    
    # Get project root
    project_root = Path(__file__).parent
    
    print("📊 Starting components...")
    print()
    
    # Start data collector in background
    print("✅ Starting WebSocket Data Collector...")
    collector_process = subprocess.Popen(
        [sys.executable, "src/data_collector.py"],
        cwd=project_root,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )
    
    # Give collector time to connect
    time.sleep(2)
    
    # Start Streamlit dashboard
    print("✅ Starting Streamlit Dashboard...")
    print()
    print("-" * 60)
    print("📈 Dashboard will open in your browser automatically")
    print("🔗 URL: http://localhost:8501")
    print("-" * 60)
    print()
    
    dashboard_process = subprocess.Popen(
        [sys.executable, "-m", "streamlit", "run", "src/dashboard.py"],
        cwd=project_root
    )
    
    try:
        # Wait for dashboard process
        dashboard_process.wait()
    except KeyboardInterrupt:
        print("\n\n⏹️  Shutting down...")
        collector_process.terminate()
        dashboard_process.terminate()
        print("✅ Services stopped")
        sys.exit(0)

if __name__ == "__main__":
    main()