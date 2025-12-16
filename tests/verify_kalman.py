import sys
import os
import pandas as pd
import numpy as np

# Add src to path
sys.path.append(os.path.join(os.getcwd(), 'src'))

# Mock missing dependencies
import sys
from unittest.mock import MagicMock
sys.modules['statsmodels'] = MagicMock()
sys.modules['statsmodels.tsa'] = MagicMock()
sys.modules['statsmodels.tsa.stattools'] = MagicMock()
sys.modules['sklearn'] = MagicMock()
sys.modules['sklearn.linear_model'] = MagicMock()
sys.modules['sklearn.covariance'] = MagicMock()

from analytics import AnalyticsEngine

def test_kalman():
    print("Testing Kalman Filter...")
    # Create synthetic cointegrated data
    np.random.seed(42)
    n = 100
    x = np.random.normal(0, 1, n).cumsum()
    # Time-varying beta
    beta = np.linspace(1, 2, n)
    y = beta * x + np.random.normal(0, 0.1, n)
    
    x_series = pd.Series(x)
    y_series = pd.Series(y)
    
    # Run Kalman
    result = AnalyticsEngine.compute_hedge_ratio_kalman(y_series, x_series)
    
    if result is None:
        print("FAILED: Result is None")
        return
        
    betas = result['beta']
    print(f"Result shape: {betas.shape}")
    print(f"First 5 betas: {betas.head().values}")
    print(f"Last 5 betas: {betas.tail().values}")
    
    # Check if betas tracked the true beta (1 -> 2)
    start_beta = betas.iloc[10] # Skip warm up
    end_beta = betas.iloc[-1]
    
    print(f"Estimated start beta: {start_beta:.4f} (True: ~1.0)")
    print(f"Estimated end beta: {end_beta:.4f} (True: ~2.0)")
    
    if 0.8 < start_beta < 1.2 and 1.8 < end_beta < 2.2:
        print("SUCCESS: Kalman Filter tracked the changing beta correctly.")
    else:
        print("WARNING: Kalman Filter might need tuning.")

if __name__ == "__main__":
    test_kalman()
