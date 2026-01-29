"""
Backtesting engine for prediction models.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, List, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass

from .ensemble import EnsemblePredictor


@dataclass
class BacktestResult:
    """Container for backtest results."""
    total_predictions: int
    correct_directions: int
    direction_accuracy: float
    mean_error: float
    mean_absolute_error: float
    rmse: float
    profitable_trades: int
    profit_factor: float
    max_drawdown: float
    sharpe_ratio: float
    predictions: List[Dict]


class Backtester:
    """
    Walk-forward backtesting engine for prediction models.
    """

    def __init__(self, initial_train_size: int = 252):
        """
        Initialize backtester.

        Args:
            initial_train_size: Initial training window size (trading days)
        """
        self.initial_train_size = initial_train_size

    def run_backtest(
        self,
        df: pd.DataFrame,
        horizon: int = 7,
        step_size: int = 7,
        use_lstm: bool = True,
        use_xgboost: bool = True
    ) -> BacktestResult:
        """
        Run walk-forward backtest.

        Args:
            df: DataFrame with OHLCV data
            horizon: Prediction horizon
            step_size: Days between predictions
            use_lstm: Whether to use LSTM
            use_xgboost: Whether to use XGBoost

        Returns:
            BacktestResult with performance metrics
        """
        if len(df) < self.initial_train_size + horizon + 50:
            return BacktestResult(
                total_predictions=0,
                correct_directions=0,
                direction_accuracy=0,
                mean_error=0,
                mean_absolute_error=0,
                rmse=0,
                profitable_trades=0,
                profit_factor=0,
                max_drawdown=0,
                sharpe_ratio=0,
                predictions=[]
            )

        predictions = []
        predictor = EnsemblePredictor(use_lstm=use_lstm, use_xgboost=use_xgboost)

        # Walk-forward loop
        start_idx = self.initial_train_size

        while start_idx + horizon < len(df):
            # Get training data
            train_df = df.iloc[:start_idx]

            # Train and predict
            try:
                predictor.train(train_df, horizons=[horizon])
                result = predictor.predict(train_df, horizon)

                if result and 'ensemble' in result:
                    pred_change = result['ensemble'].get('predicted_change_pct', 0)
                    pred_direction = result['ensemble'].get('direction', 'Unknown')

                    # Get actual change
                    actual_price_at_pred = df['Close'].iloc[start_idx - 1]
                    actual_price_after = df['Close'].iloc[min(start_idx + horizon - 1, len(df) - 1)]
                    actual_change = (actual_price_after / actual_price_at_pred - 1) * 100
                    actual_direction = 'Up' if actual_change > 0 else 'Down'

                    predictions.append({
                        'date': df.index[start_idx - 1] if isinstance(df.index, pd.DatetimeIndex) else start_idx,
                        'predicted_change': pred_change,
                        'actual_change': actual_change,
                        'predicted_direction': pred_direction,
                        'actual_direction': actual_direction,
                        'direction_correct': pred_direction == actual_direction,
                        'error': pred_change - actual_change,
                        'confidence': result['ensemble'].get('confidence', 0)
                    })

            except Exception as e:
                print(f"Error at index {start_idx}: {e}")

            start_idx += step_size

        return self._calculate_metrics(predictions)

    def _calculate_metrics(self, predictions: List[Dict]) -> BacktestResult:
        """Calculate backtest performance metrics."""
        if not predictions:
            return BacktestResult(
                total_predictions=0,
                correct_directions=0,
                direction_accuracy=0,
                mean_error=0,
                mean_absolute_error=0,
                rmse=0,
                profitable_trades=0,
                profit_factor=0,
                max_drawdown=0,
                sharpe_ratio=0,
                predictions=[]
            )

        total = len(predictions)
        correct = sum(1 for p in predictions if p['direction_correct'])

        errors = [p['error'] for p in predictions]
        actual_changes = [p['actual_change'] for p in predictions]

        # Calculate metrics
        mean_error = np.mean(errors)
        mae = np.mean([abs(e) for e in errors])
        rmse = np.sqrt(np.mean([e ** 2 for e in errors]))

        # Trading metrics (assuming we trade in direction of prediction)
        profitable = sum(1 for p in predictions
                        if (p['predicted_direction'] == 'Up' and p['actual_change'] > 0) or
                        (p['predicted_direction'] == 'Down' and p['actual_change'] < 0))

        # Profit factor
        gains = sum(abs(p['actual_change']) for p in predictions
                   if (p['predicted_direction'] == 'Up' and p['actual_change'] > 0) or
                   (p['predicted_direction'] == 'Down' and p['actual_change'] < 0))
        losses = sum(abs(p['actual_change']) for p in predictions
                    if (p['predicted_direction'] == 'Up' and p['actual_change'] < 0) or
                    (p['predicted_direction'] == 'Down' and p['actual_change'] > 0))
        profit_factor = gains / losses if losses > 0 else float('inf')

        # Simulated returns (assuming we follow predictions)
        returns = []
        for p in predictions:
            if p['predicted_direction'] == 'Up':
                returns.append(p['actual_change'] / 100)
            else:
                returns.append(-p['actual_change'] / 100)

        # Calculate drawdown
        cumulative = np.cumprod([1 + r for r in returns])
        running_max = np.maximum.accumulate(cumulative)
        drawdowns = (cumulative - running_max) / running_max
        max_drawdown = abs(min(drawdowns)) if len(drawdowns) > 0 else 0

        # Sharpe ratio (annualized, assuming weekly predictions)
        if returns:
            sharpe = np.mean(returns) / (np.std(returns) + 0.0001) * np.sqrt(52)
        else:
            sharpe = 0

        return BacktestResult(
            total_predictions=total,
            correct_directions=correct,
            direction_accuracy=correct / total if total > 0 else 0,
            mean_error=mean_error,
            mean_absolute_error=mae,
            rmse=rmse,
            profitable_trades=profitable,
            profit_factor=profit_factor,
            max_drawdown=max_drawdown,
            sharpe_ratio=sharpe,
            predictions=predictions
        )

    def run_quick_backtest(
        self,
        df: pd.DataFrame,
        horizon: int = 7,
        num_tests: int = 10
    ) -> Dict[str, Any]:
        """
        Run quick backtest with fewer iterations.

        Args:
            df: DataFrame with OHLCV data
            horizon: Prediction horizon
            num_tests: Number of test points

        Returns:
            Dictionary with quick backtest results
        """
        if len(df) < self.initial_train_size + horizon * 2:
            return {'error': 'Insufficient data'}

        # Select evenly spaced test points
        available_range = len(df) - self.initial_train_size - horizon
        step = max(1, available_range // num_tests)

        predictions = []
        predictor = EnsemblePredictor(use_lstm=False, use_xgboost=True)  # Faster without LSTM

        for i in range(num_tests):
            idx = self.initial_train_size + i * step

            if idx + horizon >= len(df):
                break

            train_df = df.iloc[:idx]

            try:
                predictor.train(train_df, horizons=[horizon])
                result = predictor.predict(train_df, horizon)

                if result and 'ensemble' in result:
                    pred_change = result['ensemble'].get('predicted_change_pct', 0)
                    pred_direction = result['ensemble'].get('direction', 'Unknown')

                    actual_price_at_pred = df['Close'].iloc[idx - 1]
                    actual_price_after = df['Close'].iloc[idx + horizon - 1]
                    actual_change = (actual_price_after / actual_price_at_pred - 1) * 100
                    actual_direction = 'Up' if actual_change > 0 else 'Down'

                    predictions.append({
                        'predicted_change': pred_change,
                        'actual_change': actual_change,
                        'direction_correct': pred_direction == actual_direction
                    })

            except Exception:
                continue

        if not predictions:
            return {'error': 'No predictions made'}

        correct = sum(1 for p in predictions if p['direction_correct'])
        mae = np.mean([abs(p['predicted_change'] - p['actual_change']) for p in predictions])

        return {
            'tests': len(predictions),
            'direction_accuracy': correct / len(predictions),
            'mean_absolute_error': mae,
            'predictions': predictions
        }

    def analyze_by_market_condition(
        self,
        backtest_result: BacktestResult,
        df: pd.DataFrame
    ) -> Dict[str, Any]:
        """
        Analyze backtest performance by market condition.

        Args:
            backtest_result: Backtest results
            df: Original DataFrame

        Returns:
            Dictionary with condition-based analysis
        """
        if not backtest_result.predictions:
            return {}

        # Calculate market volatility for each prediction
        results_by_condition = {
            'high_volatility': [],
            'low_volatility': [],
            'uptrend': [],
            'downtrend': []
        }

        for pred in backtest_result.predictions:
            idx = pred.get('date')
            if isinstance(idx, (datetime, pd.Timestamp)):
                # Find index in DataFrame
                try:
                    loc = df.index.get_loc(idx)
                except:
                    continue
            else:
                loc = idx

            if loc < 20:
                continue

            # Calculate volatility
            returns = df['Close'].iloc[loc-20:loc].pct_change().dropna()
            volatility = returns.std() * np.sqrt(252)

            # Calculate trend
            sma_20 = df['Close'].iloc[loc-20:loc].mean()
            current_price = df['Close'].iloc[loc]
            trend = 'uptrend' if current_price > sma_20 else 'downtrend'

            # Categorize
            if volatility > 0.3:
                results_by_condition['high_volatility'].append(pred)
            else:
                results_by_condition['low_volatility'].append(pred)

            results_by_condition[trend].append(pred)

        # Calculate accuracy for each condition
        analysis = {}
        for condition, preds in results_by_condition.items():
            if preds:
                correct = sum(1 for p in preds if p['direction_correct'])
                analysis[condition] = {
                    'count': len(preds),
                    'accuracy': correct / len(preds),
                    'avg_error': np.mean([abs(p['error']) for p in preds])
                }

        return analysis

    def get_confidence_calibration(
        self,
        backtest_result: BacktestResult
    ) -> Dict[str, Any]:
        """
        Analyze if model confidence correlates with accuracy.

        Args:
            backtest_result: Backtest results

        Returns:
            Dictionary with calibration analysis
        """
        if not backtest_result.predictions:
            return {}

        # Bin predictions by confidence
        bins = {
            'low (0-40%)': [],
            'medium (40-60%)': [],
            'high (60-80%)': [],
            'very_high (80-100%)': []
        }

        for pred in backtest_result.predictions:
            conf = pred.get('confidence', 0.5) * 100

            if conf < 40:
                bins['low (0-40%)'].append(pred)
            elif conf < 60:
                bins['medium (40-60%)'].append(pred)
            elif conf < 80:
                bins['high (60-80%)'].append(pred)
            else:
                bins['very_high (80-100%)'].append(pred)

        calibration = {}
        for bin_name, preds in bins.items():
            if preds:
                correct = sum(1 for p in preds if p['direction_correct'])
                calibration[bin_name] = {
                    'count': len(preds),
                    'accuracy': correct / len(preds),
                    'avg_confidence': np.mean([p.get('confidence', 0.5) for p in preds])
                }

        return calibration

    def generate_backtest_report(self, result: BacktestResult) -> str:
        """
        Generate human-readable backtest report.

        Args:
            result: Backtest results

        Returns:
            Formatted report string
        """
        lines = [
            "=" * 50,
            "BACKTEST REPORT",
            "=" * 50,
            "",
            f"Total Predictions: {result.total_predictions}",
            f"Direction Accuracy: {result.direction_accuracy * 100:.1f}%",
            f"Profitable Trades: {result.profitable_trades} ({result.profitable_trades / max(1, result.total_predictions) * 100:.1f}%)",
            "",
            "Error Metrics:",
            f"  Mean Error: {result.mean_error:.2f}%",
            f"  Mean Absolute Error: {result.mean_absolute_error:.2f}%",
            f"  RMSE: {result.rmse:.2f}%",
            "",
            "Risk Metrics:",
            f"  Profit Factor: {result.profit_factor:.2f}",
            f"  Max Drawdown: {result.max_drawdown * 100:.1f}%",
            f"  Sharpe Ratio: {result.sharpe_ratio:.2f}",
            "",
            "=" * 50
        ]

        return "\n".join(lines)
