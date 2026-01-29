"""
SQLite-based caching system for stock data.
"""

import sqlite3
import pandas as pd
import json
from datetime import datetime, timedelta
from typing import Optional, Any, Dict
import os
from pathlib import Path


class CacheManager:
    """
    Manages local SQLite cache for stock data.
    """

    def __init__(self, db_path: str = "data/stock_cache.db", expiry_hours: int = 24):
        """
        Initialize cache manager.

        Args:
            db_path: Path to SQLite database
            expiry_hours: Hours before cache entries expire
        """
        self.db_path = db_path
        self.expiry_hours = expiry_hours
        self._ensure_db_directory()
        self._init_database()

    def _ensure_db_directory(self):
        """Ensure the database directory exists."""
        db_dir = os.path.dirname(self.db_path)
        if db_dir:
            Path(db_dir).mkdir(parents=True, exist_ok=True)

    def _get_connection(self) -> sqlite3.Connection:
        """Get database connection."""
        return sqlite3.connect(self.db_path)

    def _init_database(self):
        """Initialize database tables."""
        with self._get_connection() as conn:
            cursor = conn.cursor()

            # Price data cache
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS price_cache (
                    symbol TEXT,
                    date TEXT,
                    open REAL,
                    high REAL,
                    low REAL,
                    close REAL,
                    volume INTEGER,
                    cached_at TEXT,
                    PRIMARY KEY (symbol, date)
                )
            """)

            # Fundamentals cache
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS fundamentals_cache (
                    symbol TEXT PRIMARY KEY,
                    data TEXT,
                    cached_at TEXT
                )
            """)

            # Info cache
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS info_cache (
                    symbol TEXT PRIMARY KEY,
                    data TEXT,
                    cached_at TEXT
                )
            """)

            # Generic cache for any data
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS generic_cache (
                    cache_key TEXT PRIMARY KEY,
                    data TEXT,
                    cached_at TEXT
                )
            """)

            conn.commit()

    def _is_expired(self, cached_at: str) -> bool:
        """Check if cache entry is expired."""
        cached_time = datetime.fromisoformat(cached_at)
        return datetime.now() - cached_time > timedelta(hours=self.expiry_hours)

    def get_price_data(
        self,
        symbol: str,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> Optional[pd.DataFrame]:
        """
        Get cached price data.

        Args:
            symbol: Stock symbol
            start_date: Start date
            end_date: End date

        Returns:
            DataFrame with price data or None
        """
        with self._get_connection() as conn:
            query = "SELECT * FROM price_cache WHERE symbol = ?"
            params = [symbol]

            if start_date:
                query += " AND date >= ?"
                params.append(start_date.strftime('%Y-%m-%d'))

            if end_date:
                query += " AND date <= ?"
                params.append(end_date.strftime('%Y-%m-%d'))

            query += " ORDER BY date"

            df = pd.read_sql_query(query, conn, params=params)

            if df.empty:
                return None

            # Check if most recent entry is expired
            if self._is_expired(df['cached_at'].iloc[-1]):
                return None

            # Format DataFrame
            df['date'] = pd.to_datetime(df['date'])
            df.set_index('date', inplace=True)
            df = df.rename(columns={
                'open': 'Open',
                'high': 'High',
                'low': 'Low',
                'close': 'Close',
                'volume': 'Volume'
            })

            return df[['Open', 'High', 'Low', 'Close', 'Volume']]

    def save_price_data(self, symbol: str, df: pd.DataFrame):
        """
        Save price data to cache.

        Args:
            symbol: Stock symbol
            df: DataFrame with price data
        """
        if df is None or df.empty:
            return

        with self._get_connection() as conn:
            cursor = conn.cursor()
            cached_at = datetime.now().isoformat()

            # Prepare data
            df_copy = df.copy()
            if isinstance(df_copy.index, pd.DatetimeIndex):
                df_copy = df_copy.reset_index()
                date_col = df_copy.columns[0]
                df_copy = df_copy.rename(columns={date_col: 'Date'})

            for _, row in df_copy.iterrows():
                date_str = row['Date'].strftime('%Y-%m-%d') if hasattr(row['Date'], 'strftime') else str(row['Date'])[:10]

                cursor.execute("""
                    INSERT OR REPLACE INTO price_cache
                    (symbol, date, open, high, low, close, volume, cached_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    symbol,
                    date_str,
                    row.get('Open', row.get('open')),
                    row.get('High', row.get('high')),
                    row.get('Low', row.get('low')),
                    row.get('Close', row.get('close')),
                    row.get('Volume', row.get('volume')),
                    cached_at
                ))

            conn.commit()

    def get_fundamentals(self, symbol: str) -> Optional[Dict[str, Any]]:
        """
        Get cached fundamentals data.

        Args:
            symbol: Stock symbol

        Returns:
            Dictionary with fundamentals or None
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT data, cached_at FROM fundamentals_cache WHERE symbol = ?",
                (symbol,)
            )
            result = cursor.fetchone()

            if result is None:
                return None

            data, cached_at = result
            if self._is_expired(cached_at):
                return None

            return json.loads(data)

    def save_fundamentals(self, symbol: str, data: Dict[str, Any]):
        """
        Save fundamentals to cache.

        Args:
            symbol: Stock symbol
            data: Fundamentals data dictionary
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT OR REPLACE INTO fundamentals_cache (symbol, data, cached_at)
                VALUES (?, ?, ?)
            """, (symbol, json.dumps(data), datetime.now().isoformat()))
            conn.commit()

    def get_info(self, symbol: str) -> Optional[Dict[str, Any]]:
        """
        Get cached stock info.

        Args:
            symbol: Stock symbol

        Returns:
            Dictionary with info or None
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT data, cached_at FROM info_cache WHERE symbol = ?",
                (symbol,)
            )
            result = cursor.fetchone()

            if result is None:
                return None

            data, cached_at = result
            if self._is_expired(cached_at):
                return None

            return json.loads(data)

    def save_info(self, symbol: str, data: Dict[str, Any]):
        """
        Save stock info to cache.

        Args:
            symbol: Stock symbol
            data: Info data dictionary
        """
        # Convert any non-serializable values
        clean_data = {}
        for key, value in data.items():
            try:
                json.dumps(value)
                clean_data[key] = value
            except (TypeError, ValueError):
                clean_data[key] = str(value)

        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT OR REPLACE INTO info_cache (symbol, data, cached_at)
                VALUES (?, ?, ?)
            """, (symbol, json.dumps(clean_data), datetime.now().isoformat()))
            conn.commit()

    def get_generic(self, cache_key: str) -> Optional[Any]:
        """
        Get generic cached data.

        Args:
            cache_key: Cache key

        Returns:
            Cached data or None
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT data, cached_at FROM generic_cache WHERE cache_key = ?",
                (cache_key,)
            )
            result = cursor.fetchone()

            if result is None:
                return None

            data, cached_at = result
            if self._is_expired(cached_at):
                return None

            return json.loads(data)

    def save_generic(self, cache_key: str, data: Any):
        """
        Save generic data to cache.

        Args:
            cache_key: Cache key
            data: Data to cache
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT OR REPLACE INTO generic_cache (cache_key, data, cached_at)
                VALUES (?, ?, ?)
            """, (cache_key, json.dumps(data), datetime.now().isoformat()))
            conn.commit()

    def clear_cache(self, symbol: Optional[str] = None):
        """
        Clear cache entries.

        Args:
            symbol: If provided, clear only for this symbol. Otherwise clear all.
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()

            if symbol:
                cursor.execute("DELETE FROM price_cache WHERE symbol = ?", (symbol,))
                cursor.execute("DELETE FROM fundamentals_cache WHERE symbol = ?", (symbol,))
                cursor.execute("DELETE FROM info_cache WHERE symbol = ?", (symbol,))
            else:
                cursor.execute("DELETE FROM price_cache")
                cursor.execute("DELETE FROM fundamentals_cache")
                cursor.execute("DELETE FROM info_cache")
                cursor.execute("DELETE FROM generic_cache")

            conn.commit()

    def clear_expired(self):
        """Remove all expired cache entries."""
        cutoff = (datetime.now() - timedelta(hours=self.expiry_hours)).isoformat()

        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM price_cache WHERE cached_at < ?", (cutoff,))
            cursor.execute("DELETE FROM fundamentals_cache WHERE cached_at < ?", (cutoff,))
            cursor.execute("DELETE FROM info_cache WHERE cached_at < ?", (cutoff,))
            cursor.execute("DELETE FROM generic_cache WHERE cached_at < ?", (cutoff,))
            conn.commit()

    def get_cache_stats(self) -> Dict[str, int]:
        """
        Get cache statistics.

        Returns:
            Dictionary with cache stats
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()

            stats = {}
            for table in ['price_cache', 'fundamentals_cache', 'info_cache', 'generic_cache']:
                cursor.execute(f"SELECT COUNT(*) FROM {table}")
                stats[table] = cursor.fetchone()[0]

            return stats
