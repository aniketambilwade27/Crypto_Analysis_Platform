"""
Analytics Engine
Computes statistical metrics, hedge ratios, spreads, and other trading analytics
"""

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.tsa.stattools import adfuller
from sklearn.linear_model import LinearRegression, HuberRegressor
from sklearn.covariance import EmpiricalCovariance
import warnings
warnings.filterwarnings('ignore')

class AnalyticsEngine:
    
    @staticmethod
    def compute_returns(prices):
        """Compute log returns"""
        return np.log(prices / prices.shift(1)).dropna()
    
    @staticmethod
    def compute_volatility(returns, window=20):
        """Compute rolling volatility (annualized)"""
        return returns.rolling(window).std() * np.sqrt(252 * 24 * 60)  # Crypto trades 24/7
    
    @staticmethod
    def compute_hedge_ratio(y, x, method='ols', lookback=None):
        """
        Compute hedge ratio via regression
        
        Args:
            y: dependent variable (pd.Series)
            x: independent variable (pd.Series)
            method: 'ols', 'huber', or 'rolling'
            lookback: window size for rolling regression
        
        Returns:
            dict with beta, alpha, r_squared, residuals
        """
        # Align series
        df = pd.DataFrame({'y': y, 'x': x}).dropna()
        
        if len(df) < 10:
            return None
        
        X = df['x'].values.reshape(-1, 1)
        Y = df['y'].values
        
        if method == 'ols':
            model = LinearRegression()
            model.fit(X, Y)
            y_pred = model.predict(X)
            
            return {
                'beta': model.coef_[0],
                'alpha': model.intercept_,
                'r_squared': model.score(X, Y),
                'residuals': Y - y_pred,
                'timestamps': df.index
            }
        
        elif method == 'huber':
            model = HuberRegressor()
            model.fit(X, Y)
            y_pred = model.predict(X)
            
            ss_res = np.sum((Y - y_pred) ** 2)
            ss_tot = np.sum((Y - np.mean(Y)) ** 2)
            r_squared = 1 - (ss_res / ss_tot)
            
            return {
                'beta': model.coef_[0],
                'alpha': model.intercept_,
                'r_squared': r_squared,
                'residuals': Y - y_pred,
                'timestamps': df.index
            }
        
        elif method == 'rolling' and lookback:
            betas = []
            alphas = []
            
            for i in range(lookback, len(df)):
                X_window = df['x'].iloc[i-lookback:i].values.reshape(-1, 1)
                Y_window = df['y'].iloc[i-lookback:i].values
                
                model = LinearRegression()
                model.fit(X_window, Y_window)
                
                betas.append(model.coef_[0])
                alphas.append(model.intercept_)
            
            return {
                'beta': pd.Series(betas, index=df.index[lookback:]),
                'alpha': pd.Series(alphas, index=df.index[lookback:]),
                'timestamps': df.index[lookback:]
            }


        elif method == 'kalman':
            return AnalyticsEngine.compute_hedge_ratio_kalman(y, x)
    
    @staticmethod
    def compute_hedge_ratio_kalman(y, x, delta=1e-5):
        """
        Compute dynamic hedge ratio using Kalman Filter
        
        Args:
            y: dependent variable (pd.Series)
            x: independent variable (pd.Series)
            delta: covariance smoothing parameter
            
        Returns:
            dict with beta (Series), alpha (Series), timestamps
        """
        # Align series
        df = pd.DataFrame({'y': y, 'x': x}).dropna()
        if len(df) < 10:
            return None
            
        # Initialize
        n_obs = len(df)
        x_vals = df['x'].values
        y_vals = df['y'].values
        
        # State: [beta, alpha]
        # Transition: beta_t = beta_{t-1} + noise
        # Observation: y_t = beta_t * x_t + alpha_t + noise
        
        # State covariance
        P = np.eye(2)
        # Process noise covariance
        Q = delta / (1 - delta) * np.eye(2)
        # Measurement noise variance
        R = 0.001
        
        # Initial state guess [0, 0]
        state = np.zeros(2)
        
        betas = np.zeros(n_obs)
        alphas = np.zeros(n_obs)
        
        for i in range(n_obs):
            xi = x_vals[i]
            yi = y_vals[i]
            
            # Observation matrix H = [x, 1]
            H = np.array([xi, 1.0])
            
            # Prediction step (Random Walk: state mean doesn't change, covariance increases)
            # x_{t|t-1} = x_{t-1|t-1}
            # P_{t|t-1} = P_{t-1|t-1} + Q
            P = P + Q
            
            # Measurement residual
            # y_pred = H @ state
            y_pred = np.dot(H, state)
            error = yi - y_pred
            
            # Residual covariance
            # S = H @ P @ H.T + R
            S = np.dot(H, np.dot(P, H.T)) + R
            
            # Kalman Gain
            # K = P @ H.T / S
            K = np.dot(P, H.T) / S
            
            # Update step
            # x_{t|t} = x_{t|t-1} + K * error
            state = state + K * error
            
            # P_{t|t} = (I - K @ H) @ P_{t|t-1}
            # optimized: P = P - K @ S @ K.T (Joseph form is better but this is simple)
            P = P - np.outer(K, H).dot(P)
            
            betas[i] = state[0]
            alphas[i] = state[1]
            
        return {
            'beta': pd.Series(betas, index=df.index),
            'alpha': pd.Series(alphas, index=df.index),
            'timestamps': df.index
        }

    @staticmethod
    def compute_spread(y, x, beta):
        """Compute spread: y - beta * x"""
        return y - beta * x
    
    @staticmethod
    def compute_zscore(series, window=20):
        """Compute rolling z-score"""
        mean = series.rolling(window).mean()
        std = series.rolling(window).std()
        return (series - mean) / std
    
    @staticmethod
    def adf_test(series):
        """
        Augmented Dickey-Fuller test for stationarity
        
        Returns:
            dict with test_statistic, p_value, critical_values, is_stationary
        """
        try:
            result = adfuller(series.dropna(), autolag='AIC')
            
            return {
                'test_statistic': result[0],
                'p_value': result[1],
                'critical_values': result[4],
                'is_stationary': result[1] < 0.05
            }
        except Exception as e:
            return {'error': str(e)}
    
    @staticmethod
    def rolling_correlation(s1, s2, window=20):
        """Compute rolling correlation"""
        return s1.rolling(window).corr(s2)
    
    @staticmethod
    def compute_correlation_matrix(price_dict, window=None):
        """
        Compute correlation matrix for multiple price series
        
        Args:
            price_dict: Dict of {symbol: price_series}
            window: Optional rolling window (None = full period)
            
        Returns:
            DataFrame with correlation coefficients
        """
        if len(price_dict) < 2:
            return None
            
        # Align all series by timestamp
        df = pd.DataFrame(price_dict)
        df = df.dropna()
        
        if len(df) < 10:
            return None
        
        # Compute returns for correlation
        returns = df.pct_change().dropna()
        
        if window:
            # Return most recent correlation window
            if len(returns) < window:
                return None
            returns = returns.tail(window)
        
        # Compute correlation matrix
        corr_matrix = returns.corr()
        
        return corr_matrix
    
    
    @staticmethod
    def compute_drawdown(prices):
        """Compute drawdown from peak"""
        cummax = prices.cummax()
        drawdown = (prices - cummax) / cummax
        return drawdown
    
    @staticmethod
    def compute_sharpe_ratio(returns, window=20, risk_free=0):
        """Compute rolling Sharpe ratio"""
        mean_ret = returns.rolling(window).mean()
        std_ret = returns.rolling(window).std()
        sharpe = (mean_ret - risk_free) / std_ret * np.sqrt(252 * 24 * 60)
        return sharpe
    
    @staticmethod
    def detect_mean_reversion_signals(zscore, entry_threshold=2, exit_threshold=0):
        """
        Generate mean reversion trading signals
        
        Returns:
            DataFrame with signals: 1 (long), -1 (short), 0 (neutral)
        """
        signals = pd.Series(0, index=zscore.index)
        position = 0
        
        for i in range(1, len(zscore)):
            z = zscore.iloc[i]
            
            if position == 0:
                if z > entry_threshold:
                    position = -1  # Short when z-score is high
                    signals.iloc[i] = -1
                elif z < -entry_threshold:
                    position = 1  # Long when z-score is low
                    signals.iloc[i] = 1
            else:
                if (position == 1 and z > exit_threshold) or \
                   (position == -1 and z < -exit_threshold):
                    position = 0
                    signals.iloc[i] = 0
                else:
                    signals.iloc[i] = position
        
        return signals
    
    @staticmethod
    def compute_price_momentum(prices, window=20):
        """Compute price momentum (rate of change)"""
        return (prices / prices.shift(window) - 1) * 100
    
    @staticmethod
    def compute_volume_profile(df, price_bins=50):
        """
        Compute volume profile (volume at price levels)
        
        Args:
            df: DataFrame with 'price' and 'size' columns
            price_bins: number of price bins
        
        Returns:
            DataFrame with price levels and volumes
        """
        if df.empty or 'price' not in df.columns or 'size' not in df.columns:
            return pd.DataFrame()
        
        price_min = df['price'].min()
        price_max = df['price'].max()
        
        bins = np.linspace(price_min, price_max, price_bins)
        df['price_bin'] = pd.cut(df['price'], bins=bins)
        
        volume_profile = df.groupby('price_bin')['size'].sum().reset_index()
        volume_profile['price_level'] = volume_profile['price_bin'].apply(lambda x: x.mid)
        
        return volume_profile[['price_level', 'size']].rename(columns={'size': 'volume'})
    
    @staticmethod
    def compute_liquidity_metrics(df, window=20):
        """
        Compute liquidity metrics
        
        Returns:
            dict with various liquidity measures
        """
        if len(df) < window:
            return {}
        
        # Bid-ask spread proxy (using price volatility)
        returns = AnalyticsEngine.compute_returns(df['price'])
        spread_proxy = returns.rolling(window).std()
        
        # Trade intensity
        trade_intensity = df['size'].rolling(window).sum()
        
        # Price impact (volume-weighted)
        df['impact'] = abs(df['price'].diff()) / df['size']
        price_impact = df['impact'].rolling(window).mean()
        
        return {
            'spread_proxy': spread_proxy,
            'trade_intensity': trade_intensity,
            'price_impact': price_impact
        }
    
    @staticmethod
    def compute_order_flow_imbalance(df, window=20):
        """
        Compute order flow imbalance
        Positive = more buying pressure, Negative = more selling pressure
        """
        # Simple imbalance based on price direction and volume
        df = df.copy()
        df['price_change'] = df['price'].diff()
        df['direction'] = np.sign(df['price_change'])
        df['signed_volume'] = df['direction'] * df['size']
        
        imbalance = df['signed_volume'].rolling(window).sum()
        return imbalance
    
    @staticmethod
    def backtest_mean_reversion(prices, spread, zscore, entry=2, exit=0):
        """
        Simple backtest for mean reversion strategy
        
        Returns:
            dict with performance metrics
        """
        signals = AnalyticsEngine.detect_mean_reversion_signals(zscore, entry, exit)
        
        # Compute returns
        returns = prices.pct_change()
        strategy_returns = signals.shift(1) * returns
        
        # Performance metrics
        total_return = (1 + strategy_returns).prod() - 1
        sharpe = strategy_returns.mean() / strategy_returns.std() * np.sqrt(252 * 24 * 60)
        
        cumulative_returns = (1 + strategy_returns).cumprod()
        max_drawdown = (cumulative_returns / cumulative_returns.cummax() - 1).min()
        
        num_trades = (signals.diff() != 0).sum()
        
        return {
            'total_return': total_return * 100,
            'sharpe_ratio': sharpe,
            'max_drawdown': max_drawdown * 100,
            'num_trades': num_trades,
            'cumulative_returns': cumulative_returns
        }