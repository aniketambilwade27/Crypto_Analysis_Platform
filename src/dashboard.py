"""
Streamlit Dashboard - Real-Time Analytics Frontend
Interactive visualization and control interface
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import time
from datetime import datetime, timedelta
import json

from storage import DataStorage
from analytics import AnalyticsEngine
from config import Config

# Page configuration
st.set_page_config(
    page_title="Crypto Analytics Platform",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS - Professional Dark Theme
st.markdown("""
<style>
    /* Import Professional Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500;600&display=swap');
    
    /* Global Styles */
    * {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    
    /* Main Container - Professional Dark */
    .main {
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 50%, #0f172a 100%);
    }
    
    /* Hide Streamlit Branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    /* Header Styles - Clean & Professional */
    .main-header {
        font-size: 2.75rem;
        font-weight: 700;
        color: #e2e8f0;
        text-align: center;
        margin-bottom: 2rem;
        animation: fadeInDown 0.6s ease-out;
        letter-spacing: -0.02em;
    }
    
    /* Cards - Professional Dark Glass */
    .glass-card {
        background: rgba(30, 41, 59, 0.5);
        backdrop-filter: blur(10px);
        border-radius: 16px;
        border: 1px solid rgba(148, 163, 184, 0.1);
        padding: 1.5rem;
        box-shadow: 0 4px 16px rgba(0, 0, 0, 0.2);
        transition: all 0.3s ease;
    }
    
    .glass-card:hover {
        border-color: rgba(148, 163, 184, 0.2);
        box-shadow: 0 6px 24px rgba(0, 0, 0, 0.3);
    }
    
    /* Metric Cards - Sophisticated */
    .stMetric {
        background: linear-gradient(135deg, rgba(30, 41, 59, 0.8) 0%, rgba(15, 23, 42, 0.8) 100%);
        backdrop-filter: blur(10px);
        padding: 1.75rem;
        border-radius: 12px;
        border: 1px solid rgba(100, 116, 139, 0.2);
        transition: all 0.3s ease;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.15);
    }
    
    .stMetric:hover {
        transform: translateY(-2px);
        border-color: rgba(59, 130, 246, 0.4);
        box-shadow: 0 4px 16px rgba(0, 0, 0, 0.25);
    }
    
    .stMetric label {
        color: #94a3b8 !important;
        font-weight: 600;
        font-size: 0.813rem;
        text-transform: uppercase;
        letter-spacing: 0.06em;
    }
    
    .stMetric [data-testid="stMetricValue"] {
        font-size: 2.25rem !important;
        font-weight: 700;
        color: #e2e8f0 !important;
    }
    
    /* Tabs Styling - Clean Professional */
    .stTabs [data-baseweb="tab-list"] {
        gap: 4px;
        background: rgba(30, 41, 59, 0.4);
        padding: 0.5rem;
        border-radius: 10px;
        border: 1px solid rgba(100, 116, 139, 0.2);
    }
    
    .stTabs [data-baseweb="tab"] {
        background: transparent;
        border-radius: 8px;
        color: #94a3b8;
        font-weight: 500;
        padding: 0.625rem 1.25rem;
        transition: all 0.2s ease;
        border: 1px solid transparent;
    }
    
    .stTabs [data-baseweb="tab"]:hover {
        color: #cbd5e1;
        background: rgba(51, 65, 85, 0.3);
    }
    
    .stTabs [aria-selected="true"] {
        background: rgba(59, 130, 246, 0.15);
        color: #60a5fa;
        border: 1px solid rgba(59, 130, 246, 0.3);
    }
    
    /* Buttons - Professional Gradient */
    .stButton>button {
        width: 100%;
        background: linear-gradient(135deg, #3b82f6 0%, #6366f1 100%);
        color: white;
        border: none;
        border-radius: 10px;
        padding: 0.75rem 1.5rem;
        font-weight: 600;
        font-size: 0.938rem;
        transition: all 0.3s ease;
        box-shadow: 0 2px 8px rgba(59, 130, 246, 0.25);
    }
    
    .stButton>button:hover {
        transform: translateY(-1px);
        box-shadow: 0 4px 12px rgba(59, 130, 246, 0.35);
        background: linear-gradient(135deg, #2563eb 0%, #4f46e5 100%);
    }
    
    /* Sidebar Styling - Professional Dark */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, rgba(15, 23, 42, 0.95) 0%, rgba(30, 41, 59, 0.95) 100%);
        backdrop-filter: blur(10px);
        border-right: 1px solid rgba(100, 116, 139, 0.15);
    }
    
    [data-testid="stSidebar"] .stSelectbox label,
    [data-testid="stSidebar"] .stSlider label,
    [data-testid="stSidebar"] .stNumberInput label,
    [data-testid="stSidebar"] .stCheckbox label {
        color: #cbd5e1 !important;
        font-weight: 500;
        font-size: 0.875rem;
    }
    
    /* Alert Boxes - Professional */
    .alert-box {
        padding: 1rem 1.5rem;
        border-radius: 12px;
        margin: 0.75rem 0;
        border-left: 3px solid;
        animation: slideInLeft 0.4s ease-out;
        backdrop-filter: blur(10px);
    }
    
    .alert-warning {
        background: rgba(251, 191, 36, 0.1);
        border-left-color: #fbbf24;
        color: #fde047;
    }
    
    .alert-danger {
        background: rgba(239, 68, 68, 0.1);
        border-left-color: #ef4444;
        color: #fca5a5;
    }
    
    .alert-success {
        background: rgba(34, 197, 94, 0.1);
        border-left-color: #22c55e;
        color: #86efac;
    }
    
    /* Live Status Indicator - Subtle */
    .status-indicator {
        display: inline-block;
        width: 8px;
        height: 8px;
        border-radius: 50%;
        background: #22c55e;
        animation: subtlePulse 2s ease-in-out infinite;
        margin-right: 8px;
    }
    
    /* Animations */
    @keyframes fadeInDown {
        from {
            opacity: 0;
            transform: translateY(-20px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
    
    @keyframes slideInLeft {
        from {
            opacity: 0;
            transform: translateX(-15px);
        }
        to {
            opacity: 1;
            transform: translateX(0);
        }
    }
    
    @keyframes subtlePulse {
        0%, 100% {
            opacity: 1;
        }
        50% {
            opacity: 0.6;
        }
    }
    
    /* Data Tables - Professional */
    .dataframe {
        border-radius: 12px;
        overflow: hidden;
        background: rgba(30, 41, 59, 0.4);
        border: 1px solid rgba(100, 116, 139, 0.2);
    }
    
    /* Plotly Charts - Clean Border */
    .js-plotly-plot {
        border-radius: 12px;
        overflow: hidden;
        border: 1px solid rgba(100, 116, 139, 0.15);
    }
    
    /* File Uploader - Professional */
    .stFileUploader {
        background: rgba(30, 41, 59, 0.3);
        border-radius: 12px;
        padding: 1.25rem;
        border: 1.5px dashed rgba(100, 116, 139, 0.3);
        transition: all 0.3s ease;
    }
    
    .stFileUploader:hover {
        border-color: rgba(59, 130, 246, 0.5);
        background: rgba(30, 41, 59, 0.5);
    }
    
    /* Footer - Clean */
    .custom-footer {
        text-align: center;
        padding: 2rem 0 1rem 0;
        color: #64748b;
        font-size: 0.875rem;
        border-top: 1px solid rgba(100, 116, 139, 0.15);
        margin-top: 2.5rem;
    }
    
    .custom-footer .status-badge {
        display: inline-block;
        background: rgba(34, 197, 94, 0.15);
        color: #86efac;
        padding: 0.25rem 0.75rem;
        border-radius: 8px;
        font-weight: 500;
        margin-left: 0.5rem;
        border: 1px solid rgba(34, 197, 94, 0.3);
    }
    
    /* Typography - Professional */
    h1, h2, h3 {
        color: #e2e8f0 !important;
        font-weight: 600;
        letter-spacing: -0.01em;
    }
    
    /* Input Fields - Dark Professional */
    .stTextInput input, .stNumberInput input {
        background: rgba(30, 41, 59, 0.6) !important;
        border: 1px solid rgba(100, 116, 139, 0.3) !important;
        border-radius: 10px !important;
        color: #e2e8f0 !important;
        font-weight: 400;
        padding: 0.625rem !important;
    }
    
    .stTextInput input:focus, .stNumberInput input:focus {
        border-color: rgba(59, 130, 246, 0.5) !important;
        background: rgba(30, 41, 59, 0.8) !important;
        box-shadow: 0 0 0 3px rgba(59, 130, 246, 0.1) !important;
    }
    
    /* Select boxes - Professional */
    .stSelectbox div[data-baseweb="select"] {
        background: rgba(30, 41, 59, 0.6);
        border: 1px solid rgba(100, 116, 139, 0.3);
        border-radius: 10px;
    }
    
    /* Slider - Subtle Accent */
    .stSlider [data-baseweb="slider"] [role="slider"] {
        background: linear-gradient(135deg, #3b82f6 0%, #6366f1 100%);
    }
    
    /* General text color - Soft */
    p, span, div {
        color: #cbd5e1;
    }
    
    /* Scrollbar - Professional */
    ::-webkit-scrollbar {
        width: 8px;
    }
    
    ::-webkit-scrollbar-track {
        background: rgba(15, 23, 42, 0.5);
        border-radius: 10px;
    }
    
    ::-webkit-scrollbar-thumb {
        background: linear-gradient(135deg, #3b82f6 0%, #6366f1 100%);
        border-radius: 10px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: linear-gradient(135deg, #2563eb 0%, #4f46e5 100%);
    }
    
    /* Remove all excessive glow effects */
    * {
        text-shadow: none !important;
    }
    
    /* Subtle focus states only */
    button:focus, input:focus, select:focus {
        outline: 2px solid rgba(59, 130, 246, 0.3);
        outline-offset: 2px;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'storage' not in st.session_state:
    st.session_state.storage = DataStorage()
    st.session_state.analytics = AnalyticsEngine()
    st.session_state.alerts = []
    st.session_state.last_update = time.time()

# Sidebar Configuration
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/000000/bitcoin.png", width=80)
    st.title("⚙️ Configuration")
    
    # Symbol selection
    st.subheader("Symbols")
    available_symbols = st.session_state.storage.get_symbols()
    
    if not available_symbols:
        st.warning("No data available yet. WebSocket collector is gathering data...")
        available_symbols = Config.DEFAULT_SYMBOLS
    
    symbol1 = st.selectbox("Primary Symbol", available_symbols, index=0)
    
    if len(available_symbols) > 1:
        symbol2 = st.selectbox("Secondary Symbol (for pairs)", available_symbols, 
                               index=1 if len(available_symbols) > 1 else 0)
    else:
        symbol2 = symbol1
    
    # Timeframe selection
    st.subheader("Timeframe")
    timeframe = st.selectbox(
        "Resampling Period",
        options=list(Config.TIMEFRAMES.keys()),
        index=1  # Default to 1m
    )
    
    # Analytics parameters
    st.subheader("Analytics Parameters")
    window_size = st.slider("Rolling Window", 10, 100, Config.DEFAULT_WINDOW, 5)
    
    hedge_method = st.selectbox(
        "Hedge Ratio Method",
        options=['ols', 'huber', 'kalman'],
        index=0
    )
    
    # Alert settings
    st.subheader("Alert Thresholds")
    zscore_threshold = st.number_input(
        "Z-Score Threshold",
        min_value=1.0,
        max_value=5.0,
        value=Config.ALERT_ZSCORE_HIGH,
        step=0.5
    )
    
    price_change_threshold = st.number_input(
        "Price Change % Alert",
        min_value=1.0,
        max_value=20.0,
        value=Config.ALERT_PRICE_CHANGE,
        step=1.0
    )
    
    # Data upload
    st.subheader("Data Upload")
    uploaded_file = st.file_uploader("Upload OHLCV CSV", type=['csv'])
    
    if uploaded_file:
        try:
            df_upload = pd.read_csv(uploaded_file)
            upload_symbol = st.text_input("Symbol for uploaded data", "CUSTOM")
            
            if st.button("Process Upload"):
                st.session_state.storage.upload_ohlcv_data(df_upload, upload_symbol)
                st.success(f"Uploaded {len(df_upload)} rows for {upload_symbol}")
        except Exception as e:
            st.error(f"Upload error: {e}")
    
    # Auto-refresh
    st.subheader("Display Options")
    auto_refresh = st.checkbox("Auto-refresh", value=True)
    refresh_interval = st.slider("Refresh Interval (s)", 1, 30, 5)

# Main Dashboard
st.markdown('''
<div class="main-header">
    <span class="status-indicator"></span>
    📊 Real-Time Crypto Analytics Platform
</div>
''', unsafe_allow_html=True)

# Database stats
stats = st.session_state.storage.get_stats()
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Total Ticks", f"{stats['total_ticks']:,}")
with col2:
    st.metric("Active Symbols", stats['symbols'])
with col3:
    if stats['earliest']:
        earliest = pd.to_datetime(stats['earliest'])
        st.metric("Data Start", earliest.strftime("%H:%M:%S"))
    else:
        st.metric("Data Start", "N/A")
with col4:
    if stats['latest']:
        latest = pd.to_datetime(stats['latest'])
        st.metric("Latest Update", latest.strftime("%H:%M:%S"))
    else:
        st.metric("Latest Update", "N/A")

# Tabs for different views
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "📈 Price Charts", 
    "🔀 Pair Analytics", 
    "📊 Statistics",
    "🎯 Signals & Backtest",
    "🔔 Alerts",
    "💾 Data Export"
])

# Tab 1: Price Charts
with tab1:
    st.subheader(f"Real-Time Price: {symbol1.upper()}")
    
    # Get tick data
    df_ticks = st.session_state.storage.get_ticks(
        symbol=symbol1, 
        limit=Config.MAX_CHART_POINTS
    )
    
    if not df_ticks.empty:
        # Resample data
        df_resampled = df_ticks.set_index('timestamp').resample(
            Config.TIMEFRAMES[timeframe]
        ).agg({
            'price': ['first', 'max', 'min', 'last', 'count'],
            'size': 'sum'
        })
        
        df_resampled.columns = ['open', 'high', 'low', 'close', 'trades', 'volume']
        df_resampled = df_resampled.dropna()

        # Aggressive filtering for invalid prices
        df_resampled = df_resampled[
            (df_resampled['open'] > 0) & 
            (df_resampled['high'] > 0) & 
            (df_resampled['low'] > 0) & 
            (df_resampled['close'] > 0) &
            (df_resampled['volume'] > 0)
        ]
        
        # Require at least 3 candles for a meaningful chart
        if len(df_resampled) >= 3:
            # Create candlestick chart
            fig = make_subplots(
                rows=3, cols=1,
                shared_xaxes=True,
                vertical_spacing=0.05,
                row_heights=[0.5, 0.25, 0.25],
                subplot_titles=('Price', 'Volume', 'Trades')
            )
            
            # Candlestick
            fig.add_trace(
                go.Candlestick(
                    x=df_resampled.index,
                    open=df_resampled['open'],
                    high=df_resampled['high'],
                    low=df_resampled['low'],
                    close=df_resampled['close'],
                    name='Price'
                ),
                row=1, col=1
            )
            
            # Volume bars
            colors = ['red' if close < open else 'green' 
                     for close, open in zip(df_resampled['close'], df_resampled['open'])]
            
            fig.add_trace(
                go.Bar(
                    x=df_resampled.index,
                    y=df_resampled['volume'],
                    name='Volume',
                    marker_color=colors
                ),
                row=2, col=1
            )
            
            # Trade count
            fig.add_trace(
                go.Scatter(
                    x=df_resampled.index,
                    y=df_resampled['trades'],
                    mode='lines',
                    name='Trades',
                    line=dict(color='orange')
                ),
                row=3, col=1
            )
            
            fig.update_layout(
                height=800,
                showlegend=True,
                xaxis_rangeslider_visible=False,
                template='plotly_dark',
                yaxis=dict(autorange=True, fixedrange=False)
            )
            

            
            st.plotly_chart(fig, use_container_width=True)
            
            # Current stats
            col1, col2, col3, col4 = st.columns(4)
            latest = df_resampled.iloc[-1]
            
            with col1:
                st.metric("Last Price", f"${latest['close']:.2f}")
            with col2:
                change = ((latest['close'] - latest['open']) / latest['open']) * 100
                st.metric("Change %", f"{change:.2f}%", delta=f"{change:.2f}%")
            with col3:
                st.metric("Volume", f"{latest['volume']:.2f}")
            with col4:
                st.metric("Trades", int(latest['trades']))
        else:
            st.info(f"Accumulating data... Need at least {timeframe} of data")
    else:
        st.warning(f"No data available for {symbol1}. Waiting for WebSocket data...")

# Continue in Part 2...

# Tab 2: Pair Analytics
with tab2:
    st.subheader(f"Pair Analysis: {symbol1.upper()} vs {symbol2.upper()}")
    
    if symbol1 != symbol2:
        # Get data for both symbols from same time window (last 10 minutes)
        from datetime import datetime, timedelta
        start_time = (datetime.now() - timedelta(minutes=10)).isoformat()
        
        df1 = st.session_state.storage.get_ticks(symbol=symbol1, start_time=start_time)
        df2 = st.session_state.storage.get_ticks(symbol=symbol2, start_time=start_time)
        
        if not df1.empty and not df2.empty:
            # Align timestamps
            # Use ffill() to handle seconds with no ticks (use last known price)
            df1_rs = df1.set_index('timestamp').resample('1S')['price'].mean().ffill()
            df2_rs = df2.set_index('timestamp').resample('1S')['price'].mean().ffill()
            
            df_aligned = pd.DataFrame({
                'price1': df1_rs,
                'price2': df2_rs
            }).dropna()
            
            if len(df_aligned) > window_size:
                # Compute hedge ratio
                hedge_result = st.session_state.analytics.compute_hedge_ratio(
                    df_aligned['price1'],
                    df_aligned['price2'],
                    method=hedge_method
                )
                
                if hedge_result:
                    beta = hedge_result['beta']
                    
                    # For spread calculation, use latest beta if it's a Series
                    beta_for_spread = beta.iloc[-1] if isinstance(beta, pd.Series) else beta
                    
                    # Compute spread and z-score
                    spread = st.session_state.analytics.compute_spread(
                        df_aligned['price1'],
                        df_aligned['price2'],
                        beta_for_spread
                    )
                    
                    zscore = st.session_state.analytics.compute_zscore(
                        spread, window=window_size
                    )
                    
                    # Rolling correlation
                    correlation = st.session_state.analytics.rolling_correlation(
                        df_aligned['price1'],
                        df_aligned['price2'],
                        window=window_size
                    )
                    
                    # Display metrics
                    col1, col2, col3 = st.columns(3)
                    
                    # Handle beta - can be scalar (OLS/Huber) or Series (Kalman)
                    if isinstance(beta, pd.Series):
                        beta_value = beta.iloc[-1]  # Get latest value for Kalman
                    else:
                        beta_value = beta
                    
                    with col1:
                        st.metric("Hedge Ratio (β)", f"{beta_value:.4f}")
                    with col2:
                        # R² may not exist for Kalman filter
                        if 'r_squared' in hedge_result and hedge_result['r_squared'] is not None:
                            st.metric("R²", f"{hedge_result['r_squared']:.4f}")
                        else:
                            st.metric("R²", "N/A (Kalman)")
                    with col3:
                        if not zscore.empty:
                            latest_z = zscore.iloc[-1]
                            st.metric("Current Z-Score", f"{latest_z:.2f}")
                            
                            # Check for alerts
                            if abs(latest_z) > zscore_threshold:
                                alert_msg = f"Z-Score alert: {latest_z:.2f}"
                                if alert_msg not in [a['message'] for a in st.session_state.alerts]:
                                    st.session_state.alerts.append({
                                        'timestamp': datetime.now(),
                                        'type': 'Z-Score',
                                        'message': alert_msg,
                                        'severity': 'warning' if abs(latest_z) < 3 else 'danger'
                                    })
                    
                    # Spread and Z-Score chart
                    fig = make_subplots(
                        rows=3, cols=1,
                        shared_xaxes=True,
                        vertical_spacing=0.05,
                        subplot_titles=('Normalized Prices', 'Spread', 'Z-Score'),
                        row_heights=[0.35, 0.35, 0.3]
                    )
                    
                    # Normalized prices
                    norm_p1 = df_aligned['price1'] / df_aligned['price1'].iloc[0] * 100
                    norm_p2 = df_aligned['price2'] / df_aligned['price2'].iloc[0] * 100
                    
                    fig.add_trace(
                        go.Scatter(x=df_aligned.index, y=norm_p1, name=symbol1.upper(),
                                  line=dict(color='cyan')),
                        row=1, col=1
                    )
                    fig.add_trace(
                        go.Scatter(x=df_aligned.index, y=norm_p2, name=symbol2.upper(),
                                  line=dict(color='magenta')),
                        row=1, col=1
                    )
                    
                    # Spread
                    fig.add_trace(
                        go.Scatter(x=spread.index, y=spread, name='Spread',
                                  line=dict(color='yellow')),
                        row=2, col=1
                    )
                    
                    # Z-Score with threshold lines
                    fig.add_trace(
                        go.Scatter(x=zscore.index, y=zscore, name='Z-Score',
                                  line=dict(color='white')),
                        row=3, col=1
                    )
                    
                    # Add threshold lines
                    fig.add_hline(y=zscore_threshold, line_dash="dash", 
                                 line_color="red", row=3, col=1)
                    fig.add_hline(y=-zscore_threshold, line_dash="dash", 
                                 line_color="red", row=3, col=1)
                    fig.add_hline(y=0, line_dash="dot", 
                                 line_color="gray", row=3, col=1)
                    
                    fig.update_layout(
                        height=800,
                        showlegend=True,
                        template='plotly_dark'
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Correlation chart
                    st.subheader("Rolling Correlation")
                    
                    fig_corr = go.Figure()
                    fig_corr.add_trace(
                        go.Scatter(
                            x=correlation.index,
                            y=correlation,
                            mode='lines',
                            name='Correlation',
                            line=dict(color='lime', width=2)
                        )
                    )
                    
                    fig_corr.update_layout(
                        height=300,
                        template='plotly_dark',
                        yaxis=dict(range=[-1, 1])
                    )
                    
                    st.plotly_chart(fig_corr, use_container_width=True)
                    
                    # ADF Test
                    st.subheader("Stationarity Test (ADF)")
                    if st.button("Run ADF Test on Spread"):
                        adf_result = st.session_state.analytics.adf_test(spread)
                        
                        if 'error' not in adf_result:
                            col1, col2 = st.columns(2)
                            
                            with col1:
                                st.metric("Test Statistic", f"{adf_result['test_statistic']:.4f}")
                                st.metric("P-Value", f"{adf_result['p_value']:.4f}")
                            
                            with col2:
                                if adf_result['is_stationary']:
                                    st.success("✅ Spread is STATIONARY")
                                else:
                                    st.warning("⚠️ Spread is NON-STATIONARY")
                                
                                st.write("**Critical Values:**")
                                for key, value in adf_result['critical_values'].items():
                                    st.write(f"  {key}: {value:.4f}")
                        else:
                            st.error(f"ADF test failed: {adf_result['error']}")
                
            else:
                current_points = len(df_aligned)
                required_points = window_size
                wait_estimate = max(0, required_points - current_points)
                st.info(
                    f"📊 Collecting data for pair analysis: {current_points}/{required_points} points "
                    f"(~{wait_estimate}s remaining at 1s timeframe)"
                )
        else:
            st.warning("Insufficient data for both symbols")
    else:
        st.info("Select different symbols for pair analysis")

# Tab 3: Statistics
with tab3:
    st.subheader("Statistical Summary")
    
    df_stats = st.session_state.storage.get_ticks(symbol=symbol1, limit=1000)
    
    if not df_stats.empty:
        prices = df_stats['price']
        volumes = df_stats['size']
        
        # Compute statistics
        returns = st.session_state.analytics.compute_returns(prices)
        volatility = st.session_state.analytics.compute_volatility(returns, window=window_size)
        momentum = st.session_state.analytics.compute_price_momentum(prices, window=window_size)
        
        # Display summary stats
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Price Statistics**")
            stats_df = pd.DataFrame({
                'Metric': ['Mean', 'Std Dev', 'Min', 'Max', 'Current'],
                'Value': [
                    f"${prices.mean():.2f}",
                    f"${prices.std():.2f}",
                    f"${prices.min():.2f}",
                    f"${prices.max():.2f}",
                    f"${prices.iloc[-1]:.2f}"
                ]
            })
            st.dataframe(stats_df, hide_index=True, use_container_width=True)
        
        with col2:
            st.write("**Volume Statistics**")
            vol_df = pd.DataFrame({
                'Metric': ['Mean', 'Std Dev', 'Min', 'Max', 'Total'],
                'Value': [
                    f"{volumes.mean():.4f}",
                    f"{volumes.std():.4f}",
                    f"{volumes.min():.4f}",
                    f"{volumes.max():.4f}",
                    f"{volumes.sum():.4f}"
                ]
            })
            st.dataframe(vol_df, hide_index=True, use_container_width=True)
        
        # Time-series stats table
        st.subheader("Time-Series Statistics")
        
        # Resample to 1-minute intervals
        df_ts = df_stats.set_index('timestamp').resample('1T').agg({
            'price': ['first', 'max', 'min', 'last', 'std'],
            'size': 'sum'
        })
        
        df_ts.columns = ['open', 'high', 'low', 'close', 'volatility', 'volume']
        df_ts = df_ts.dropna().tail(20)  # Last 20 minutes
        
        # Add derived features
        df_ts['return'] = df_ts['close'].pct_change() * 100
        df_ts['range'] = df_ts['high'] - df_ts['low']
        
        # Format for display
        df_display = df_ts.reset_index()
        df_display['timestamp'] = df_display['timestamp'].dt.strftime('%H:%M:%S')
        
        st.dataframe(
            df_display.style.format({
                'open': '${:.2f}',
                'high': '${:.2f}',
                'low': '${:.2f}',
                'close': '${:.2f}',
                'volatility': '{:.4f}',
                'volume': '{:.4f}',
                'return': '{:.3f}%',
                'range': '${:.2f}'
            }),
            use_container_width=True,
            height=400
        )
        
        # Download button for stats
        csv = df_display.to_csv(index=False)
        st.download_button(
            label="📥 Download Statistics CSV",
            data=csv,
            file_name=f"{symbol1}_stats_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv"
        )
        
        # Volatility chart
        st.subheader("Rolling Volatility")
        
        if not volatility.empty and len(volatility.dropna()) > 0:
            fig_vol = go.Figure()
            fig_vol.add_trace(
                go.Scatter(
                    x=volatility.index,
                    y=volatility,
                    mode='lines',
                    fill='tozeroy',
                    name='Volatility',
                    line=dict(color='orange')
                )
            )
            
            fig_vol.update_layout(
                height=300,
                template='plotly_dark',
                yaxis_title='Volatility (Annualized)'
            )
            
            st.plotly_chart(fig_vol, use_container_width=True)
        
        # Cross-Correlation Heatmap
        st.subheader("📊 Cross-Asset Correlation Heatmap")
        
        # Get all available symbols
        all_symbols = available_symbols
        
        if len(all_symbols) >= 2:
            # Fetch price data for all symbols
            price_dict = {}
            for sym in all_symbols:
                sym_data = st.session_state.storage.get_ticks(symbol=sym, limit=500)
                if not sym_data.empty:
                    # Resample to 1-second for alignment
                    prices = sym_data.set_index('timestamp').resample('1S')['price'].mean()
                    price_dict[sym.upper()] = prices
            
            if len(price_dict) >= 2:
                corr_window = st.slider("Correlation Window (data points)", 20, 500, 50, 10)
                corr_matrix = st.session_state.analytics.compute_correlation_matrix(
                    price_dict, window=corr_window
                )
                
                if corr_matrix is not None:
                    # Create heatmap
                    fig_heatmap = go.Figure(data=go.Heatmap(
                        z=corr_matrix.values,
                        x=corr_matrix.columns,
                        y=corr_matrix.index,
                        colorscale='RdYlGn',  # Red-Yellow-Green
                        zmid=0,
                        text=corr_matrix.values,
                        texttemplate='%{text:.2f}',
                        textfont={"size": 12},
                        colorbar=dict(title="Correlation")
                    ))
                    
                    fig_heatmap.update_layout(
                        title="Price Return Correlations",
                        xaxis_title="Symbol",
                        yaxis_title="Symbol",
                        height=500,
                        template='plotly_dark'
                    )
                    
                    st.plotly_chart(fig_heatmap, use_container_width=True)
                    
                    # Insights
                    st.markdown("**📌 Interpretation:**")
                    st.markdown("""
                    - **Green (≥0.7)**: Strong positive correlation - move together
                    - **Yellow (0.3-0.7)**: Moderate correlation
                    - **Red (≤0.3)**: Weak/negative correlation - good for diversification
                    """)
                else:
                    st.info(
                        "⏳ Accumulating data for correlation analysis... "
                        "Reduce 'Correlation Window' slider or wait ~50s after startup for best results."
                    )
            else:
                st.warning("Need at least 2 symbols with data for correlation heatmap")
        else:
            st.info(
                "⏳ Waiting for data from multiple symbols... "
                "Please wait ~30s after starting the application."
            )
    else:
        st.warning("No data available for statistics")

# Tab 4: Signals & Backtest
with tab4:
    st.subheader("Mean Reversion Signals & Backtest")
    
    if symbol1 != symbol2:
        # Fetch last 10 minutes of data for both symbols to ensure overlap
        from datetime import datetime, timedelta
        start_time = (datetime.now() - timedelta(minutes=10)).isoformat()
        
        df1 = st.session_state.storage.get_ticks(symbol=symbol1, start_time=start_time)
        df2 = st.session_state.storage.get_ticks(symbol=symbol2, start_time=start_time)
        
        if not df1.empty and not df2.empty:
            # Align and compute spread/z-score
            df1_rs = df1.set_index('timestamp').resample('1S')['price'].mean()
            df2_rs = df2.set_index('timestamp').resample('1S')['price'].mean()
            
            df_aligned = pd.DataFrame({
                'price1': df1_rs,
                'price2': df2_rs
            }).dropna()
            
            if len(df_aligned) > window_size:
                hedge_result = st.session_state.analytics.compute_hedge_ratio(
                    df_aligned['price1'], df_aligned['price2'], method=hedge_method
                )
                
                if hedge_result:
                    spread = st.session_state.analytics.compute_spread(
                        df_aligned['price1'], df_aligned['price2'], hedge_result['beta']
                    )
                    zscore = st.session_state.analytics.compute_zscore(spread, window=window_size)
                    
                    # Backtest parameters
                    col1, col2 = st.columns(2)
                    with col1:
                        entry_threshold = st.slider("Entry Threshold", 1.0, 4.0, 2.0, 0.5)
                    with col2:
                        exit_threshold = st.slider("Exit Threshold", -1.0, 1.0, 0.0, 0.5)
                    
                    if st.button("Run Backtest"):
                        backtest_result = st.session_state.analytics.backtest_mean_reversion(
                            df_aligned['price1'], spread, zscore, entry_threshold, exit_threshold
                        )
                        
                        # Display results
                        col1, col2, col3, col4 = st.columns(4)
                        
                        with col1:
                            st.metric("Total Return", f"{backtest_result['total_return']:.2f}%")
                        with col2:
                            st.metric("Sharpe Ratio", f"{backtest_result['sharpe_ratio']:.2f}")
                        with col3:
                            st.metric("Max Drawdown", f"{backtest_result['max_drawdown']:.2f}%")
                        with col4:
                            st.metric("Num Trades", int(backtest_result['num_trades']))
                        
                        # Plot cumulative returns
                        fig_bt = go.Figure()
                        fig_bt.add_trace(
                            go.Scatter(
                                x=backtest_result['cumulative_returns'].index,
                                y=backtest_result['cumulative_returns'],
                                mode='lines',
                                name='Strategy Returns',
                                line=dict(color='lime', width=2)
                            )
                        )
                        
                        fig_bt.update_layout(
                            height=400,
                            template='plotly_dark',
                            title='Cumulative Returns',
                            yaxis_title='Cumulative Return'
                        )
                        
                        st.plotly_chart(fig_bt, use_container_width=True)
                    
                    # Current signals
                    signals = st.session_state.analytics.detect_mean_reversion_signals(
                        zscore, entry_threshold, exit_threshold
                    )
                    
                    current_signal = signals.iloc[-1] if not signals.empty else 0
                    
                    st.subheader("Current Signal")
                    
                    if current_signal == 1:
                        st.success("🟢 LONG Signal - Buy spread (Long asset 1, Short asset 2)")
                    elif current_signal == -1:
                        st.error("🔴 SHORT Signal - Sell spread (Short asset 1, Long asset 2)")
                    else:
                        st.info("⚪ NEUTRAL - No position")
                else:
                    st.error("❌ Failed to compute hedge ratio. Try a different method or wait for more data.")
            else:
                current_points = len(df_aligned)
                required_points = window_size
                wait_estimate = max(0, required_points - current_points)
                st.info(
                    f"📊 Collecting data for backtest: {current_points}/{required_points} points "
                    f"(~{wait_estimate}s remaining at 1s timeframe)"
                )
        else:
            st.warning("⚠️ Insufficient data for both symbols. Please wait for data collection...")
    else:
        st.info("Select different symbols to generate trading signals")

# Tab 5: Alerts
with tab5:
    st.subheader("🔔 Active Alerts")
    
    if st.session_state.alerts:
        # Sort by timestamp (most recent first)
        sorted_alerts = sorted(
            st.session_state.alerts,
            key=lambda x: x['timestamp'],
            reverse=True
        )
        
        for alert in sorted_alerts[:10]:  # Show last 10 alerts
            severity_class = 'alert-warning' if alert['severity'] == 'warning' else 'alert-danger'
            st.markdown(
                f'<div class="alert-box {severity_class}">'
                f'<strong>{alert["timestamp"].strftime("%H:%M:%S")}</strong> - '
                f'{alert["type"]}: {alert["message"]}'
                f'</div>',
                unsafe_allow_html=True
            )
        
        if st.button("Clear All Alerts"):
            st.session_state.alerts = []
            st.rerun()
    else:
        st.info("No alerts triggered yet")
    
    # Alert configuration summary
    st.subheader("Alert Configuration")
    config_df = pd.DataFrame({
        'Alert Type': ['Z-Score High/Low', 'Price Change %'],
        'Threshold': [f'±{zscore_threshold}', f'{price_change_threshold}%'],
        'Status': ['✅ Active', '✅ Active']
    })
    st.table(config_df)

# Tab 6: Data Export
with tab6:
    st.subheader("💾 Data Export")
    
    export_symbol = st.selectbox("Select Symbol for Export", available_symbols)
    export_limit = st.number_input("Number of Records", min_value=100, max_value=100000, 
                                   value=1000, step=100)
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("Export Tick Data"):
            df_export = st.session_state.storage.get_ticks(export_symbol, limit=export_limit)
            
            if not df_export.empty:
                csv_data = df_export.to_csv(index=False)
                st.download_button(
                    label="📥 Download Tick Data CSV",
                    data=csv_data,
                    file_name=f"{export_symbol}_ticks_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )
                st.success(f"Prepared {len(df_export)} records for download")
            else:
                st.warning("No data to export")
    
    with col2:
        if st.button("Export OHLCV Data"):
            # Resample and export
            df_export = st.session_state.storage.get_ticks(export_symbol, limit=export_limit)
            
            if not df_export.empty:
                df_ohlcv = df_export.set_index('timestamp').resample('1T').agg({
                    'price': ['first', 'max', 'min', 'last'],
                    'size': 'sum'
                })
                df_ohlcv.columns = ['open', 'high', 'low', 'close', 'volume']
                df_ohlcv = df_ohlcv.dropna().reset_index()
                
                csv_data = df_ohlcv.to_csv(index=False)
                st.download_button(
                    label="📥 Download OHLCV CSV",
                    data=csv_data,
                    file_name=f"{export_symbol}_ohlcv_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )
                st.success(f"Prepared {len(df_ohlcv)} OHLCV records for download")
            else:
                st.warning("No data to export")

# Auto-refresh
if auto_refresh:
    time.sleep(refresh_interval)
    st.rerun()

# Footer
st.markdown("""
<div class="custom-footer">
    <strong>Real-Time Crypto Analytics Platform</strong>
    <span class="status-badge">● LIVE</span>
    <br>
    <span style="font-size: 0.75rem;">
        Powered by Binance WebSocket | Advanced Quantitative Analytics
    </span>
    <br>
    <span style="font-size: 0.75rem; color: #475569;">
        Last updated: {}</span>
</div>
""".format(datetime.now().strftime('%Y-%m-%d %H:%M:%S')), unsafe_allow_html=True)