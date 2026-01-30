"""
Configuration and constants for the Stock Analyzer application.
"""

# Application settings
APP_NAME = "Indian Stock Market Analyzer"
APP_VERSION = "1.0.0"

# Yahoo Finance settings
YAHOO_FINANCE_SUFFIX_NSE = ".NS"
YAHOO_FINANCE_SUFFIX_BSE = ".BO"

# Default stock symbols for testing (Nifty 50 and popular stocks)
DEFAULT_STOCKS = [
    # Large Cap - IT
    "TCS.NS", "INFY.NS", "WIPRO.NS", "HCLTECH.NS", "TECHM.NS",
    # Large Cap - Banking
    "HDFCBANK.NS", "ICICIBANK.NS", "SBIN.NS", "KOTAKBANK.NS", "AXISBANK.NS",
    # Large Cap - Oil & Gas
    "RELIANCE.NS", "ONGC.NS", "NTPC.NS", "POWERGRID.NS", "COALINDIA.NS",
    # Large Cap - Auto
    "MARUTI.NS", "TATAMOTORS.NS", "M&M.NS", "BAJAJ-AUTO.NS", "HEROMOTOCO.NS",
    # Large Cap - Pharma
    "SUNPHARMA.NS", "DRREDDY.NS", "CIPLA.NS", "DIVISLAB.NS", "APOLLOHOSP.NS",
    # Large Cap - FMCG
    "HINDUNILVR.NS", "ITC.NS", "NESTLEIND.NS", "BRITANNIA.NS", "DABUR.NS",
    # Large Cap - Metals
    "TATASTEEL.NS", "HINDALCO.NS", "JSWSTEEL.NS", "VEDL.NS",
    # Large Cap - Infrastructure
    "LT.NS", "ADANIENT.NS", "ADANIPORTS.NS", "ULTRACEMCO.NS", "GRASIM.NS",
    # Large Cap - Others
    "BHARTIARTL.NS", "ASIANPAINT.NS", "BAJFINANCE.NS", "TITAN.NS", "HDFC.NS",
]

# Popular sectors and their representative stocks
SECTORS = {
    "IT": ["TCS.NS", "INFY.NS", "WIPRO.NS", "HCLTECH.NS", "TECHM.NS"],
    "Banking": ["HDFCBANK.NS", "ICICIBANK.NS", "SBIN.NS", "KOTAKBANK.NS", "AXISBANK.NS"],
    "Oil & Gas": ["RELIANCE.NS", "ONGC.NS", "IOC.NS", "BPCL.NS", "GAIL.NS"],
    "Pharma": ["SUNPHARMA.NS", "DRREDDY.NS", "CIPLA.NS", "DIVISLAB.NS", "APOLLOHOSP.NS"],
    "Auto": ["MARUTI.NS", "TATAMOTORS.NS", "M&M.NS", "BAJAJ-AUTO.NS", "HEROMOTOCO.NS"],
    "FMCG": ["HINDUNILVR.NS", "ITC.NS", "NESTLEIND.NS", "BRITANNIA.NS", "DABUR.NS"],
    "Metals": ["TATASTEEL.NS", "HINDALCO.NS", "JSWSTEEL.NS", "VEDL.NS", "COALINDIA.NS"],
    "Telecom": ["BHARTIARTL.NS", "IDEA.NS"],
    "Infra": ["LTIM.NS", "ADANIENT.NS", "ADANIPORTS.NS", "ULTRACEMCO.NS", "GRASIM.NS"],
}

# Cache settings
CACHE_DB_PATH = "data/stock_cache.db"
CACHE_EXPIRY_HOURS = 24

# Technical analysis parameters
TECHNICAL_PARAMS = {
    "sma_periods": [20, 50, 200],
    "ema_periods": [12, 26, 50],
    "rsi_period": 14,
    "rsi_overbought": 70,
    "rsi_oversold": 30,
    "macd_fast": 12,
    "macd_slow": 26,
    "macd_signal": 9,
    "bollinger_period": 20,
    "bollinger_std": 2,
}

# Fundamental analysis thresholds
FUNDAMENTAL_THRESHOLDS = {
    "pe_ratio": {"low": 15, "high": 30},
    "pb_ratio": {"low": 1, "high": 5},
    "roe": {"good": 15, "excellent": 20},
    "roa": {"good": 5, "excellent": 10},
    "debt_to_equity": {"safe": 1, "risky": 2},
    "current_ratio": {"safe": 1.5, "excellent": 2},
    "dividend_yield": {"good": 2, "excellent": 4},
}

# DCF Model parameters
DCF_PARAMS = {
    "risk_free_rate": 0.07,  # 7% - India 10-year govt bond yield
    "market_risk_premium": 0.06,  # 6% market risk premium
    "terminal_growth_rate": 0.03,  # 3% terminal growth
    "projection_years": 5,
    "default_beta": 1.0,
}

# ML Model parameters
ML_PARAMS = {
    "lstm_lookback": 60,  # 60 days lookback
    "lstm_units": 50,
    "lstm_epochs": 50,
    "lstm_batch_size": 32,
    "xgboost_n_estimators": 100,
    "xgboost_max_depth": 5,
    "test_size": 0.2,
    "prediction_horizons": [7, 30, 90],  # days
}

# Scoring weights
SCORING_WEIGHTS = {
    "financial_health": 0.25,
    "growth": 0.20,
    "valuation": 0.25,
    "technical_momentum": 0.15,
    "prediction_confidence": 0.15,
}

# Recommendation thresholds
RECOMMENDATION_THRESHOLDS = {
    "strong_buy": 80,
    "buy": 65,
    "hold": 45,
    "sell": 30,
    # Below 30 is "Strong Sell"
}

# News settings
NEWS_SOURCES = [
    "https://news.google.com/rss/search?q={symbol}+stock+india&hl=en-IN&gl=IN&ceid=IN:en",
]
MAX_NEWS_ARTICLES = 10

# Chart settings
CHART_HEIGHT = 500
CHART_COLORS = {
    "primary": "#1f77b4",
    "secondary": "#ff7f0e",
    "positive": "#2ca02c",
    "negative": "#d62728",
    "neutral": "#7f7f7f",
}

# Disclaimer
DISCLAIMER = """
DISCLAIMER: This tool is for educational and informational purposes only.
It does not constitute financial advice. Stock market investments are
subject to market risks. Past performance does not guarantee future results.
Always consult a qualified financial advisor before making investment decisions.
"""
