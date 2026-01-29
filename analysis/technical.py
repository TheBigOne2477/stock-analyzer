"""
Technical analysis module for stock evaluation.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, Tuple, List
import ta

from config import TECHNICAL_PARAMS


class TechnicalAnalyzer:
    """
    Performs technical analysis on stock price data.
    """

    def __init__(self, params: Optional[Dict] = None):
        """
        Initialize the technical analyzer.

        Args:
            params: Custom parameters for technical indicators
        """
        self.params = params or TECHNICAL_PARAMS

    def analyze(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Perform comprehensive technical analysis.

        Args:
            df: DataFrame with OHLCV data

        Returns:
            Dictionary with all technical indicators and signals
        """
        if df is None or df.empty:
            return {}

        analysis = {
            'moving_averages': self._analyze_moving_averages(df),
            'rsi': self._analyze_rsi(df),
            'macd': self._analyze_macd(df),
            'bollinger_bands': self._analyze_bollinger(df),
            'volume': self._analyze_volume(df),
            'support_resistance': self._find_support_resistance(df),
            'trend': self._analyze_trend(df),
        }

        # Calculate overall technical score
        analysis['score'] = self._calculate_technical_score(analysis)
        analysis['signal'] = self._generate_signal(analysis)
        analysis['summary'] = self._generate_summary(analysis)

        return analysis

    def add_all_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add all technical indicators to the DataFrame.

        Args:
            df: DataFrame with OHLCV data

        Returns:
            DataFrame with added indicator columns
        """
        if df is None or df.empty:
            return df

        df = df.copy()

        # Moving Averages
        for period in self.params['sma_periods']:
            df[f'SMA_{period}'] = ta.trend.sma_indicator(df['Close'], window=period)

        for period in self.params['ema_periods']:
            df[f'EMA_{period}'] = ta.trend.ema_indicator(df['Close'], window=period)

        # RSI
        df['RSI'] = ta.momentum.rsi(df['Close'], window=self.params['rsi_period'])

        # MACD
        macd = ta.trend.MACD(
            df['Close'],
            window_slow=self.params['macd_slow'],
            window_fast=self.params['macd_fast'],
            window_sign=self.params['macd_signal']
        )
        df['MACD'] = macd.macd()
        df['MACD_Signal'] = macd.macd_signal()
        df['MACD_Histogram'] = macd.macd_diff()

        # Bollinger Bands
        bollinger = ta.volatility.BollingerBands(
            df['Close'],
            window=self.params['bollinger_period'],
            window_dev=self.params['bollinger_std']
        )
        df['BB_Upper'] = bollinger.bollinger_hband()
        df['BB_Middle'] = bollinger.bollinger_mavg()
        df['BB_Lower'] = bollinger.bollinger_lband()
        df['BB_Width'] = bollinger.bollinger_wband()

        # Volume indicators
        df['Volume_SMA'] = df['Volume'].rolling(window=20).mean()
        df['Volume_Ratio'] = df['Volume'] / df['Volume_SMA']

        # ATR (Average True Range)
        df['ATR'] = ta.volatility.average_true_range(df['High'], df['Low'], df['Close'])

        # Stochastic
        stoch = ta.momentum.StochasticOscillator(df['High'], df['Low'], df['Close'])
        df['Stoch_K'] = stoch.stoch()
        df['Stoch_D'] = stoch.stoch_signal()

        # ADX (Average Directional Index)
        df['ADX'] = ta.trend.adx(df['High'], df['Low'], df['Close'])

        return df

    def _analyze_moving_averages(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze moving average indicators."""
        close = df['Close'].iloc[-1]

        # Calculate SMAs
        sma_20 = df['Close'].rolling(20).mean().iloc[-1]
        sma_50 = df['Close'].rolling(50).mean().iloc[-1]
        sma_200 = df['Close'].rolling(200).mean().iloc[-1] if len(df) >= 200 else None

        # Calculate EMAs
        ema_12 = df['Close'].ewm(span=12).mean().iloc[-1]
        ema_26 = df['Close'].ewm(span=26).mean().iloc[-1]

        # Determine signals
        signals = []

        # Golden/Death Cross
        if sma_50 > sma_200 if sma_200 else False:
            signals.append('Golden Cross (Bullish)')
        elif sma_50 < sma_200 if sma_200 else False:
            signals.append('Death Cross (Bearish)')

        # Price vs MAs
        if close > sma_20:
            signals.append('Above SMA20 (Bullish)')
        else:
            signals.append('Below SMA20 (Bearish)')

        if close > sma_50:
            signals.append('Above SMA50 (Bullish)')
        else:
            signals.append('Below SMA50 (Bearish)')

        # Trend determination
        if close > sma_20 > sma_50:
            trend = 'Strong Uptrend'
            score = 80
        elif close > sma_20:
            trend = 'Uptrend'
            score = 65
        elif close < sma_20 < sma_50:
            trend = 'Strong Downtrend'
            score = 20
        elif close < sma_20:
            trend = 'Downtrend'
            score = 35
        else:
            trend = 'Sideways'
            score = 50

        return {
            'sma_20': sma_20,
            'sma_50': sma_50,
            'sma_200': sma_200,
            'ema_12': ema_12,
            'ema_26': ema_26,
            'current_price': close,
            'signals': signals,
            'trend': trend,
            'score': score
        }

    def _analyze_rsi(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze RSI indicator."""
        rsi = ta.momentum.rsi(df['Close'], window=self.params['rsi_period'])
        current_rsi = rsi.iloc[-1]
        prev_rsi = rsi.iloc[-2] if len(rsi) > 1 else current_rsi

        # Determine signal
        overbought = self.params['rsi_overbought']
        oversold = self.params['rsi_oversold']

        if current_rsi > overbought:
            signal = 'Overbought'
            score = 30
        elif current_rsi < oversold:
            signal = 'Oversold'
            score = 70
        elif current_rsi > 50:
            signal = 'Bullish'
            score = 60
        else:
            signal = 'Bearish'
            score = 40

        # Check for divergence
        divergence = None
        price_change = df['Close'].iloc[-1] - df['Close'].iloc[-10] if len(df) > 10 else 0
        rsi_change = current_rsi - rsi.iloc[-10] if len(rsi) > 10 else 0

        if price_change > 0 and rsi_change < 0:
            divergence = 'Bearish Divergence'
            score -= 10
        elif price_change < 0 and rsi_change > 0:
            divergence = 'Bullish Divergence'
            score += 10

        return {
            'value': current_rsi,
            'previous': prev_rsi,
            'signal': signal,
            'divergence': divergence,
            'overbought_threshold': overbought,
            'oversold_threshold': oversold,
            'score': max(0, min(100, score))
        }

    def _analyze_macd(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze MACD indicator."""
        macd = ta.trend.MACD(
            df['Close'],
            window_slow=self.params['macd_slow'],
            window_fast=self.params['macd_fast'],
            window_sign=self.params['macd_signal']
        )

        macd_line = macd.macd().iloc[-1]
        signal_line = macd.macd_signal().iloc[-1]
        histogram = macd.macd_diff().iloc[-1]
        prev_histogram = macd.macd_diff().iloc[-2] if len(macd.macd_diff()) > 1 else histogram

        # Determine signal
        if macd_line > signal_line:
            if histogram > prev_histogram:
                signal = 'Strong Bullish'
                score = 75
            else:
                signal = 'Bullish'
                score = 60
        else:
            if histogram < prev_histogram:
                signal = 'Strong Bearish'
                score = 25
            else:
                signal = 'Bearish'
                score = 40

        # Check for crossover
        prev_macd = macd.macd().iloc[-2] if len(macd.macd()) > 1 else macd_line
        prev_signal = macd.macd_signal().iloc[-2] if len(macd.macd_signal()) > 1 else signal_line

        crossover = None
        if prev_macd < prev_signal and macd_line > signal_line:
            crossover = 'Bullish Crossover'
            score = 80
        elif prev_macd > prev_signal and macd_line < signal_line:
            crossover = 'Bearish Crossover'
            score = 20

        return {
            'macd_line': macd_line,
            'signal_line': signal_line,
            'histogram': histogram,
            'signal': signal,
            'crossover': crossover,
            'score': score
        }

    def _analyze_bollinger(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze Bollinger Bands."""
        bollinger = ta.volatility.BollingerBands(
            df['Close'],
            window=self.params['bollinger_period'],
            window_dev=self.params['bollinger_std']
        )

        upper = bollinger.bollinger_hband().iloc[-1]
        middle = bollinger.bollinger_mavg().iloc[-1]
        lower = bollinger.bollinger_lband().iloc[-1]
        width = bollinger.bollinger_wband().iloc[-1]

        close = df['Close'].iloc[-1]

        # Calculate position within bands (0 = lower, 1 = upper)
        band_position = (close - lower) / (upper - lower) if upper != lower else 0.5

        # Determine signal
        if close > upper:
            signal = 'Above Upper Band'
            score = 30  # Overbought
        elif close < lower:
            signal = 'Below Lower Band'
            score = 70  # Oversold
        elif band_position > 0.8:
            signal = 'Near Upper Band'
            score = 40
        elif band_position < 0.2:
            signal = 'Near Lower Band'
            score = 60
        else:
            signal = 'Within Bands'
            score = 50

        # Volatility assessment
        volatility = 'High' if width > 0.1 else 'Low' if width < 0.03 else 'Normal'

        return {
            'upper': upper,
            'middle': middle,
            'lower': lower,
            'width': width,
            'band_position': band_position,
            'signal': signal,
            'volatility': volatility,
            'score': score
        }

    def _analyze_volume(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze volume indicators."""
        current_volume = df['Volume'].iloc[-1]
        avg_volume = df['Volume'].rolling(20).mean().iloc[-1]
        volume_ratio = current_volume / avg_volume if avg_volume > 0 else 1

        # Price change
        price_change = (df['Close'].iloc[-1] - df['Close'].iloc[-2]) / df['Close'].iloc[-2] if len(df) > 1 else 0

        # Determine signal
        if volume_ratio > 2 and price_change > 0:
            signal = 'High Volume Buying'
            score = 75
        elif volume_ratio > 2 and price_change < 0:
            signal = 'High Volume Selling'
            score = 25
        elif volume_ratio > 1.5 and price_change > 0:
            signal = 'Above Average Volume (Bullish)'
            score = 65
        elif volume_ratio > 1.5 and price_change < 0:
            signal = 'Above Average Volume (Bearish)'
            score = 35
        elif volume_ratio < 0.5:
            signal = 'Low Volume'
            score = 50
        else:
            signal = 'Normal Volume'
            score = 50

        return {
            'current_volume': current_volume,
            'average_volume': avg_volume,
            'volume_ratio': volume_ratio,
            'signal': signal,
            'score': score
        }

    def _find_support_resistance(self, df: pd.DataFrame, window: int = 20) -> Dict[str, Any]:
        """Find support and resistance levels."""
        highs = df['High']
        lows = df['Low']
        close = df['Close'].iloc[-1]

        # Find local maxima and minima
        resistance_levels = []
        support_levels = []

        # Use rolling windows to find pivots
        for i in range(window, len(df) - window):
            # Check for local maximum
            if highs.iloc[i] == highs.iloc[i-window:i+window+1].max():
                resistance_levels.append(highs.iloc[i])

            # Check for local minimum
            if lows.iloc[i] == lows.iloc[i-window:i+window+1].min():
                support_levels.append(lows.iloc[i])

        # Get nearest levels
        resistance_levels = sorted(set([r for r in resistance_levels if r > close]))[:3]
        support_levels = sorted(set([s for s in support_levels if s < close]), reverse=True)[:3]

        # If no levels found, use recent high/low
        if not resistance_levels:
            resistance_levels = [df['High'].tail(50).max()]
        if not support_levels:
            support_levels = [df['Low'].tail(50).min()]

        return {
            'resistance_levels': resistance_levels,
            'support_levels': support_levels,
            'nearest_resistance': resistance_levels[0] if resistance_levels else None,
            'nearest_support': support_levels[0] if support_levels else None,
            'current_price': close
        }

    def _analyze_trend(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze overall trend."""
        # ADX for trend strength
        adx = ta.trend.adx(df['High'], df['Low'], df['Close']).iloc[-1]

        # Price change over different periods
        changes = {}
        for period in [5, 20, 50, 200]:
            if len(df) >= period:
                change = (df['Close'].iloc[-1] - df['Close'].iloc[-period]) / df['Close'].iloc[-period] * 100
                changes[f'{period}d'] = change

        # Determine trend direction and strength
        if adx > 25:
            strength = 'Strong'
        elif adx > 20:
            strength = 'Moderate'
        else:
            strength = 'Weak'

        # Overall direction based on multiple timeframes
        bullish_count = sum(1 for v in changes.values() if v > 0)
        total_count = len(changes)

        if bullish_count >= total_count * 0.75:
            direction = 'Bullish'
            score = 70 + (adx / 2 if adx > 25 else 0)
        elif bullish_count >= total_count * 0.5:
            direction = 'Neutral'
            score = 50
        else:
            direction = 'Bearish'
            score = 30 - (adx / 2 if adx > 25 else 0)

        return {
            'adx': adx,
            'strength': strength,
            'direction': direction,
            'price_changes': changes,
            'score': max(0, min(100, score))
        }

    def _calculate_technical_score(self, analysis: Dict[str, Any]) -> float:
        """Calculate overall technical score."""
        scores = []
        weights = {
            'moving_averages': 0.25,
            'rsi': 0.20,
            'macd': 0.20,
            'bollinger_bands': 0.15,
            'volume': 0.10,
            'trend': 0.10
        }

        for key, weight in weights.items():
            if key in analysis and 'score' in analysis[key]:
                scores.append(analysis[key]['score'] * weight)

        return sum(scores) / sum(weights.values()) if scores else 50

    def _generate_signal(self, analysis: Dict[str, Any]) -> str:
        """Generate overall trading signal."""
        score = analysis.get('score', 50)

        if score >= 70:
            return 'Strong Buy'
        elif score >= 60:
            return 'Buy'
        elif score >= 40:
            return 'Hold'
        elif score >= 30:
            return 'Sell'
        else:
            return 'Strong Sell'

    def _generate_summary(self, analysis: Dict[str, Any]) -> str:
        """Generate text summary of technical analysis."""
        parts = []

        # Trend
        if 'trend' in analysis:
            trend = analysis['trend']
            parts.append(f"{trend['strength']} {trend['direction']} trend (ADX: {trend['adx']:.1f}).")

        # Moving Averages
        if 'moving_averages' in analysis:
            ma = analysis['moving_averages']
            parts.append(f"Price is {ma['trend']}.")

        # RSI
        if 'rsi' in analysis:
            rsi = analysis['rsi']
            parts.append(f"RSI at {rsi['value']:.1f} ({rsi['signal']}).")

        # MACD
        if 'macd' in analysis:
            macd = analysis['macd']
            if macd['crossover']:
                parts.append(f"MACD shows {macd['crossover']}.")
            else:
                parts.append(f"MACD is {macd['signal']}.")

        return " ".join(parts)

    def get_indicator_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Get DataFrame with all indicators for charting.

        Args:
            df: DataFrame with OHLCV data

        Returns:
            DataFrame with indicator columns added
        """
        return self.add_all_indicators(df)
