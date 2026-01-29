"""
Utility functions for the Stock Analyzer application.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Optional, Union, List, Dict, Any


def format_indian_number(num: float, decimal_places: int = 2) -> str:
    """
    Format number in Indian numbering system (lakhs, crores).

    Args:
        num: Number to format
        decimal_places: Number of decimal places

    Returns:
        Formatted string with appropriate suffix
    """
    if pd.isna(num) or num is None:
        return "N/A"

    abs_num = abs(num)
    sign = "-" if num < 0 else ""

    if abs_num >= 1e7:  # Crores
        return f"{sign}₹{abs_num/1e7:.{decimal_places}f} Cr"
    elif abs_num >= 1e5:  # Lakhs
        return f"{sign}₹{abs_num/1e5:.{decimal_places}f} L"
    elif abs_num >= 1e3:  # Thousands
        return f"{sign}₹{abs_num/1e3:.{decimal_places}f} K"
    else:
        return f"{sign}₹{abs_num:.{decimal_places}f}"


def format_percentage(value: float, decimal_places: int = 2) -> str:
    """
    Format value as percentage.

    Args:
        value: Value to format (0.15 = 15%)
        decimal_places: Number of decimal places

    Returns:
        Formatted percentage string
    """
    if pd.isna(value) or value is None:
        return "N/A"
    return f"{value * 100:.{decimal_places}f}%"


def format_ratio(value: float, decimal_places: int = 2) -> str:
    """
    Format ratio value.

    Args:
        value: Ratio value
        decimal_places: Number of decimal places

    Returns:
        Formatted ratio string
    """
    if pd.isna(value) or value is None:
        return "N/A"
    return f"{value:.{decimal_places}f}"


def calculate_cagr(start_value: float, end_value: float, years: int) -> Optional[float]:
    """
    Calculate Compound Annual Growth Rate.

    Args:
        start_value: Starting value
        end_value: Ending value
        years: Number of years

    Returns:
        CAGR as decimal (0.15 = 15%)
    """
    if start_value <= 0 or end_value <= 0 or years <= 0:
        return None
    return (end_value / start_value) ** (1 / years) - 1


def calculate_volatility(returns: pd.Series, annualize: bool = True) -> float:
    """
    Calculate volatility (standard deviation of returns).

    Args:
        returns: Series of returns
        annualize: Whether to annualize (multiply by sqrt(252))

    Returns:
        Volatility value
    """
    vol = returns.std()
    if annualize:
        vol *= np.sqrt(252)
    return vol


def calculate_sharpe_ratio(
    returns: pd.Series,
    risk_free_rate: float = 0.07,
    periods_per_year: int = 252
) -> float:
    """
    Calculate Sharpe Ratio.

    Args:
        returns: Series of returns
        risk_free_rate: Annual risk-free rate
        periods_per_year: Trading periods per year

    Returns:
        Sharpe Ratio
    """
    excess_returns = returns - risk_free_rate / periods_per_year
    if excess_returns.std() == 0:
        return 0
    return np.sqrt(periods_per_year) * excess_returns.mean() / excess_returns.std()


def calculate_max_drawdown(prices: pd.Series) -> float:
    """
    Calculate maximum drawdown.

    Args:
        prices: Series of prices

    Returns:
        Maximum drawdown as decimal
    """
    rolling_max = prices.expanding().max()
    drawdowns = prices / rolling_max - 1
    return drawdowns.min()


def get_date_range(period: str) -> tuple:
    """
    Get start and end dates for a given period.

    Args:
        period: Period string ('1M', '3M', '6M', '1Y', '2Y', '5Y', 'MAX')

    Returns:
        Tuple of (start_date, end_date)
    """
    end_date = datetime.now()

    period_map = {
        "1M": timedelta(days=30),
        "3M": timedelta(days=90),
        "6M": timedelta(days=180),
        "1Y": timedelta(days=365),
        "2Y": timedelta(days=730),
        "5Y": timedelta(days=1825),
        "MAX": timedelta(days=3650),
    }

    delta = period_map.get(period, timedelta(days=365))
    start_date = end_date - delta

    return start_date, end_date


def normalize_stock_symbol(symbol: str) -> str:
    """
    Normalize stock symbol to Yahoo Finance format.

    Args:
        symbol: Stock symbol (e.g., 'RELIANCE', 'RELIANCE.NS')

    Returns:
        Normalized symbol with .NS suffix
    """
    symbol = symbol.upper().strip()
    if not symbol.endswith('.NS') and not symbol.endswith('.BO'):
        symbol += '.NS'
    return symbol


def get_sector_for_stock(symbol: str, sectors: Dict[str, List[str]]) -> Optional[str]:
    """
    Get sector for a given stock symbol.

    Args:
        symbol: Stock symbol
        sectors: Dictionary of sectors and their stocks

    Returns:
        Sector name or None
    """
    symbol = normalize_stock_symbol(symbol)
    for sector, stocks in sectors.items():
        if symbol in stocks:
            return sector
    return None


def safe_divide(numerator: float, denominator: float, default: float = 0.0) -> float:
    """
    Safe division that handles zero denominator.

    Args:
        numerator: Numerator value
        denominator: Denominator value
        default: Default value if division not possible

    Returns:
        Result of division or default
    """
    if denominator == 0 or pd.isna(denominator) or pd.isna(numerator):
        return default
    return numerator / denominator


def calculate_returns(prices: pd.Series, periods: int = 1) -> pd.Series:
    """
    Calculate returns over specified periods.

    Args:
        prices: Series of prices
        periods: Number of periods for return calculation

    Returns:
        Series of returns
    """
    return prices.pct_change(periods=periods)


def moving_average(data: pd.Series, window: int) -> pd.Series:
    """
    Calculate simple moving average.

    Args:
        data: Input data series
        window: Window size

    Returns:
        Moving average series
    """
    return data.rolling(window=window).mean()


def exponential_moving_average(data: pd.Series, span: int) -> pd.Series:
    """
    Calculate exponential moving average.

    Args:
        data: Input data series
        span: EMA span

    Returns:
        EMA series
    """
    return data.ewm(span=span, adjust=False).mean()


def detect_trend(prices: pd.Series, short_window: int = 20, long_window: int = 50) -> str:
    """
    Detect price trend using moving average crossover.

    Args:
        prices: Series of prices
        short_window: Short MA window
        long_window: Long MA window

    Returns:
        'Bullish', 'Bearish', or 'Neutral'
    """
    short_ma = moving_average(prices, short_window)
    long_ma = moving_average(prices, long_window)

    if len(prices) < long_window:
        return "Neutral"

    latest_short = short_ma.iloc[-1]
    latest_long = long_ma.iloc[-1]

    if latest_short > latest_long * 1.02:
        return "Bullish"
    elif latest_short < latest_long * 0.98:
        return "Bearish"
    else:
        return "Neutral"


def calculate_support_resistance(
    prices: pd.Series,
    window: int = 20,
    num_levels: int = 3
) -> Dict[str, List[float]]:
    """
    Calculate support and resistance levels using pivot points.

    Args:
        prices: Series of prices (high, low, close in DataFrame expected)
        window: Window for finding local extrema
        num_levels: Number of support/resistance levels to return

    Returns:
        Dictionary with 'support' and 'resistance' levels
    """
    if isinstance(prices, pd.Series):
        # Simple approach using price series
        rolling_min = prices.rolling(window=window).min()
        rolling_max = prices.rolling(window=window).max()

        current_price = prices.iloc[-1]

        # Get recent lows as support
        supports = rolling_min.dropna().unique()
        supports = sorted([s for s in supports if s < current_price], reverse=True)[:num_levels]

        # Get recent highs as resistance
        resistances = rolling_max.dropna().unique()
        resistances = sorted([r for r in resistances if r > current_price])[:num_levels]

        return {
            "support": supports,
            "resistance": resistances
        }

    return {"support": [], "resistance": []}


def score_to_recommendation(score: float) -> str:
    """
    Convert numerical score to recommendation.

    Args:
        score: Score from 0-100

    Returns:
        Recommendation string
    """
    if score >= 80:
        return "Strong Buy"
    elif score >= 65:
        return "Buy"
    elif score >= 45:
        return "Hold"
    elif score >= 30:
        return "Sell"
    else:
        return "Strong Sell"


def get_recommendation_color(recommendation: str) -> str:
    """
    Get color for recommendation.

    Args:
        recommendation: Recommendation string

    Returns:
        Color hex code
    """
    colors = {
        "Strong Buy": "#00c853",
        "Buy": "#4caf50",
        "Hold": "#ff9800",
        "Sell": "#f44336",
        "Strong Sell": "#b71c1c",
    }
    return colors.get(recommendation, "#7f7f7f")


def validate_dataframe(df: pd.DataFrame, required_columns: List[str]) -> bool:
    """
    Validate DataFrame has required columns.

    Args:
        df: DataFrame to validate
        required_columns: List of required column names

    Returns:
        True if valid, False otherwise
    """
    if df is None or df.empty:
        return False
    return all(col in df.columns for col in required_columns)


def resample_ohlc(df: pd.DataFrame, freq: str = 'W') -> pd.DataFrame:
    """
    Resample OHLC data to different frequency.

    Args:
        df: DataFrame with OHLC data
        freq: Frequency ('D', 'W', 'M')

    Returns:
        Resampled DataFrame
    """
    ohlc_dict = {
        'Open': 'first',
        'High': 'max',
        'Low': 'min',
        'Close': 'last',
        'Volume': 'sum'
    }

    available_cols = {k: v for k, v in ohlc_dict.items() if k in df.columns}

    return df.resample(freq).agg(available_cols)
