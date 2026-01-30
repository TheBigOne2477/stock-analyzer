"""
Stock data fetching module using Yahoo Finance.
"""

import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, List
import logging

from .cache import CacheManager
from utils.helpers import normalize_stock_symbol

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class StockDataFetcher:
    """
    Fetches stock data from Yahoo Finance for Indian stocks (NSE/BSE).
    """

    def __init__(self, use_cache: bool = True, cache_expiry_hours: int = 24):
        """
        Initialize the data fetcher.

        Args:
            use_cache: Whether to use local caching
            cache_expiry_hours: Hours before cache expires
        """
        self.use_cache = use_cache
        if use_cache:
            self.cache = CacheManager(expiry_hours=cache_expiry_hours)
        else:
            self.cache = None

    def get_stock(self, symbol: str) -> yf.Ticker:
        """
        Get Yahoo Finance Ticker object.

        Args:
            symbol: Stock symbol

        Returns:
            yfinance Ticker object
        """
        symbol = normalize_stock_symbol(symbol)
        return yf.Ticker(symbol)

    def get_historical_data(
        self,
        symbol: str,
        period: str = "1y",
        interval: str = "1d",
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> Optional[pd.DataFrame]:
        """
        Get historical price data.

        Args:
            symbol: Stock symbol
            period: Data period ('1mo', '3mo', '6mo', '1y', '2y', '5y', 'max')
            interval: Data interval ('1d', '1wk', '1mo')
            start_date: Optional start date
            end_date: Optional end date

        Returns:
            DataFrame with OHLCV data
        """
        symbol = normalize_stock_symbol(symbol)

        # Try cache first
        if self.use_cache and self.cache:
            cached_data = self.cache.get_price_data(symbol, start_date, end_date)
            if cached_data is not None and not cached_data.empty:
                logger.info(f"Using cached data for {symbol}")
                return cached_data

        # Fetch from Yahoo Finance
        try:
            ticker = yf.Ticker(symbol)

            if start_date and end_date:
                df = ticker.history(start=start_date, end=end_date, interval=interval)
            else:
                df = ticker.history(period=period, interval=interval)

            if df.empty:
                logger.warning(f"No data found for {symbol}")
                return None

            # Clean up column names
            df = df[['Open', 'High', 'Low', 'Close', 'Volume']]

            # Cache the data
            if self.use_cache and self.cache:
                self.cache.save_price_data(symbol, df)

            return df

        except Exception as e:
            logger.error(f"Error fetching data for {symbol}: {e}")
            return None

    def get_stock_info(self, symbol: str) -> Optional[Dict[str, Any]]:
        """
        Get stock information (company details, sector, etc.).

        Args:
            symbol: Stock symbol

        Returns:
            Dictionary with stock info
        """
        symbol = normalize_stock_symbol(symbol)

        # Try cache first
        if self.use_cache and self.cache:
            cached_info = self.cache.get_info(symbol)
            if cached_info is not None:
                logger.info(f"Using cached info for {symbol}")
                return cached_info

        try:
            ticker = yf.Ticker(symbol)
            info = ticker.info

            # More lenient validation - check if we have any meaningful data
            if not info:
                return None

            # Check for common fields that indicate valid stock data
            has_valid_data = any(key in info for key in [
                'symbol', 'shortName', 'longName', 'currentPrice',
                'regularMarketPrice', 'previousClose', 'marketCap'
            ])

            if not has_valid_data:
                logger.warning(f"No valid data fields found for {symbol}")
                return None

            # Cache the info
            if self.use_cache and self.cache:
                self.cache.save_info(symbol, info)

            return info

        except Exception as e:
            logger.error(f"Error fetching info for {symbol}: {e}")
            return None

    def get_financials(self, symbol: str) -> Dict[str, Optional[pd.DataFrame]]:
        """
        Get financial statements.

        Args:
            symbol: Stock symbol

        Returns:
            Dictionary with income statement, balance sheet, cash flow
        """
        symbol = normalize_stock_symbol(symbol)

        try:
            ticker = yf.Ticker(symbol)

            return {
                'income_statement': ticker.income_stmt,
                'balance_sheet': ticker.balance_sheet,
                'cash_flow': ticker.cashflow,
                'quarterly_income': ticker.quarterly_income_stmt,
                'quarterly_balance': ticker.quarterly_balance_sheet,
                'quarterly_cashflow': ticker.quarterly_cashflow
            }

        except Exception as e:
            logger.error(f"Error fetching financials for {symbol}: {e}")
            return {
                'income_statement': None,
                'balance_sheet': None,
                'cash_flow': None,
                'quarterly_income': None,
                'quarterly_balance': None,
                'quarterly_cashflow': None
            }

    def get_key_metrics(self, symbol: str) -> Dict[str, Any]:
        """
        Get key financial metrics.

        Args:
            symbol: Stock symbol

        Returns:
            Dictionary with key metrics
        """
        info = self.get_stock_info(symbol)

        if info is None:
            return {}

        metrics = {
            # Valuation
            'market_cap': info.get('marketCap'),
            'enterprise_value': info.get('enterpriseValue'),
            'pe_ratio': info.get('trailingPE'),
            'forward_pe': info.get('forwardPE'),
            'peg_ratio': info.get('pegRatio'),
            'pb_ratio': info.get('priceToBook'),
            'ps_ratio': info.get('priceToSalesTrailing12Months'),

            # Profitability
            'profit_margin': info.get('profitMargins'),
            'operating_margin': info.get('operatingMargins'),
            'gross_margin': info.get('grossMargins'),
            'roe': info.get('returnOnEquity'),
            'roa': info.get('returnOnAssets'),

            # Growth
            'revenue_growth': info.get('revenueGrowth'),
            'earnings_growth': info.get('earningsGrowth'),
            'earnings_quarterly_growth': info.get('earningsQuarterlyGrowth'),

            # Dividends
            'dividend_yield': info.get('dividendYield'),
            'dividend_rate': info.get('dividendRate'),
            'payout_ratio': info.get('payoutRatio'),

            # Financial Health
            'debt_to_equity': info.get('debtToEquity'),
            'current_ratio': info.get('currentRatio'),
            'quick_ratio': info.get('quickRatio'),
            'total_debt': info.get('totalDebt'),
            'total_cash': info.get('totalCash'),

            # Per Share
            'eps': info.get('trailingEps'),
            'forward_eps': info.get('forwardEps'),
            'book_value': info.get('bookValue'),
            'revenue_per_share': info.get('revenuePerShare'),

            # Trading Info
            'current_price': info.get('currentPrice') or info.get('regularMarketPrice'),
            'previous_close': info.get('previousClose'),
            'open': info.get('open'),
            'day_high': info.get('dayHigh'),
            'day_low': info.get('dayLow'),
            'fifty_two_week_high': info.get('fiftyTwoWeekHigh'),
            'fifty_two_week_low': info.get('fiftyTwoWeekLow'),
            'fifty_day_average': info.get('fiftyDayAverage'),
            'two_hundred_day_average': info.get('twoHundredDayAverage'),
            'volume': info.get('volume'),
            'average_volume': info.get('averageVolume'),
            'beta': info.get('beta'),

            # Company Info
            'company_name': info.get('longName') or info.get('shortName'),
            'sector': info.get('sector'),
            'industry': info.get('industry'),
            'description': info.get('longBusinessSummary'),
            'website': info.get('website'),
            'employees': info.get('fullTimeEmployees'),
        }

        return metrics

    def get_multiple_stocks_data(
        self,
        symbols: List[str],
        period: str = "1y"
    ) -> Dict[str, pd.DataFrame]:
        """
        Get historical data for multiple stocks.

        Args:
            symbols: List of stock symbols
            period: Data period

        Returns:
            Dictionary mapping symbols to DataFrames
        """
        data = {}
        for symbol in symbols:
            df = self.get_historical_data(symbol, period=period)
            if df is not None:
                data[normalize_stock_symbol(symbol)] = df
        return data

    def get_peer_comparison_data(
        self,
        symbol: str,
        peers: List[str]
    ) -> Dict[str, Dict[str, Any]]:
        """
        Get comparison data for a stock and its peers.

        Args:
            symbol: Main stock symbol
            peers: List of peer stock symbols

        Returns:
            Dictionary with metrics for each stock
        """
        all_symbols = [symbol] + peers
        comparison = {}

        for sym in all_symbols:
            metrics = self.get_key_metrics(sym)
            if metrics:
                comparison[normalize_stock_symbol(sym)] = metrics

        return comparison

    def get_dividend_history(self, symbol: str) -> Optional[pd.DataFrame]:
        """
        Get dividend payment history.

        Args:
            symbol: Stock symbol

        Returns:
            DataFrame with dividend history
        """
        symbol = normalize_stock_symbol(symbol)

        try:
            ticker = yf.Ticker(symbol)
            dividends = ticker.dividends

            if dividends.empty:
                return None

            return dividends.to_frame(name='Dividend')

        except Exception as e:
            logger.error(f"Error fetching dividends for {symbol}: {e}")
            return None

    def get_splits_history(self, symbol: str) -> Optional[pd.DataFrame]:
        """
        Get stock split history.

        Args:
            symbol: Stock symbol

        Returns:
            DataFrame with split history
        """
        symbol = normalize_stock_symbol(symbol)

        try:
            ticker = yf.Ticker(symbol)
            splits = ticker.splits

            if splits.empty:
                return None

            return splits.to_frame(name='Split Ratio')

        except Exception as e:
            logger.error(f"Error fetching splits for {symbol}: {e}")
            return None

    def search_stocks(self, query: str) -> List[Dict[str, str]]:
        """
        Search for stocks by name or symbol.

        Args:
            query: Search query

        Returns:
            List of matching stocks with symbol and name
        """
        # Yahoo Finance doesn't have a direct search API
        # This is a simple approach using known Indian stocks
        from config import DEFAULT_STOCKS, SECTORS

        results = []

        # Check default stocks
        all_stocks = set(DEFAULT_STOCKS)
        for sector_stocks in SECTORS.values():
            all_stocks.update(sector_stocks)

        query_upper = query.upper()
        for symbol in all_stocks:
            if query_upper in symbol:
                info = self.get_stock_info(symbol)
                if info:
                    results.append({
                        'symbol': symbol,
                        'name': info.get('longName', symbol),
                        'sector': info.get('sector', 'Unknown')
                    })

        return results

    def get_intraday_data(
        self,
        symbol: str,
        period: str = "1d",
        interval: str = "5m"
    ) -> Optional[pd.DataFrame]:
        """
        Get intraday price data.

        Args:
            symbol: Stock symbol
            period: Period ('1d', '5d')
            interval: Interval ('1m', '5m', '15m', '30m', '1h')

        Returns:
            DataFrame with intraday OHLCV data
        """
        symbol = normalize_stock_symbol(symbol)

        try:
            ticker = yf.Ticker(symbol)
            df = ticker.history(period=period, interval=interval)

            if df.empty:
                return None

            return df[['Open', 'High', 'Low', 'Close', 'Volume']]

        except Exception as e:
            logger.error(f"Error fetching intraday data for {symbol}: {e}")
            return None

    def validate_symbol(self, symbol: str) -> bool:
        """
        Validate if a stock symbol exists.

        Args:
            symbol: Stock symbol

        Returns:
            True if valid, False otherwise
        """
        symbol = normalize_stock_symbol(symbol)

        try:
            ticker = yf.Ticker(symbol)
            info = ticker.info
            return info is not None and 'symbol' in info

        except Exception:
            return False

    def get_actions(self, symbol: str) -> Dict[str, pd.DataFrame]:
        """
        Get corporate actions (dividends and splits).

        Args:
            symbol: Stock symbol

        Returns:
            Dictionary with dividends and splits DataFrames
        """
        symbol = normalize_stock_symbol(symbol)

        try:
            ticker = yf.Ticker(symbol)
            return {
                'dividends': ticker.dividends,
                'splits': ticker.splits
            }
        except Exception as e:
            logger.error(f"Error fetching actions for {symbol}: {e}")
            return {'dividends': pd.Series(), 'splits': pd.Series()}
