"""
Ensemble prediction combining multiple models.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, List
from dataclasses import dataclass

from .models import LSTMModel, XGBoostModel, SimpleMovingAverageModel


@dataclass
class PredictionResult:
    """Container for prediction results."""
    model_name: str
    predicted_change_pct: float
    confidence: float
    direction: str
    details: Dict[str, Any]


class EnsemblePredictor:
    """
    Combines predictions from multiple models.
    """

    def __init__(self, use_lstm: bool = True, use_xgboost: bool = True):
        """
        Initialize ensemble predictor.

        Args:
            use_lstm: Whether to use LSTM model
            use_xgboost: Whether to use XGBoost model
        """
        self.use_lstm = use_lstm
        self.use_xgboost = use_xgboost

        self.lstm_model = LSTMModel() if use_lstm else None
        self.xgboost_model = XGBoostModel() if use_xgboost else None
        self.sma_model = SimpleMovingAverageModel()

        # Model weights based on typical performance
        self.weights = {
            'lstm': 0.35,
            'xgboost': 0.40,
            'sma': 0.25
        }

        self.training_metrics = {}

    def train(self, df: pd.DataFrame, horizons: List[int] = None) -> Dict[str, Any]:
        """
        Train all models in the ensemble.

        Args:
            df: DataFrame with OHLCV data
            horizons: List of prediction horizons

        Returns:
            Dictionary with training metrics for each model
        """
        if horizons is None:
            horizons = [7, 30]

        prices = df['Close']
        results = {}

        # Train LSTM
        if self.lstm_model:
            try:
                lstm_metrics = self.lstm_model.train(prices, verbose=0)
                results['lstm'] = lstm_metrics
                if 'error' not in lstm_metrics:
                    self.training_metrics['lstm'] = lstm_metrics
            except Exception as e:
                results['lstm'] = {'error': str(e)}

        # Train XGBoost for each horizon
        if self.xgboost_model:
            try:
                # Train for primary horizon
                xgb_metrics = self.xgboost_model.train(df, horizon=horizons[0])
                results['xgboost'] = xgb_metrics
                if 'error' not in xgb_metrics:
                    self.training_metrics['xgboost'] = xgb_metrics
            except Exception as e:
                results['xgboost'] = {'error': str(e)}

        results['sma'] = {'status': 'No training required'}

        return results

    def predict(self, df: pd.DataFrame, horizon: int = 7) -> Dict[str, Any]:
        """
        Make ensemble prediction.

        Args:
            df: DataFrame with OHLCV data
            horizon: Prediction horizon in days

        Returns:
            Dictionary with ensemble prediction and individual model results
        """
        prices = df['Close']
        predictions = []
        model_results = {}

        # Get LSTM prediction
        if self.lstm_model:
            try:
                lstm_pred = self.lstm_model.predict(prices, horizon)
                if 'error' not in lstm_pred:
                    predictions.append(PredictionResult(
                        model_name='LSTM',
                        predicted_change_pct=lstm_pred['predicted_change_pct'],
                        confidence=self._calculate_confidence('lstm'),
                        direction='Up' if lstm_pred['predicted_change_pct'] > 0 else 'Down',
                        details=lstm_pred
                    ))
                    model_results['lstm'] = lstm_pred
            except Exception as e:
                model_results['lstm'] = {'error': str(e)}

        # Get XGBoost prediction
        if self.xgboost_model:
            try:
                xgb_pred = self.xgboost_model.predict(df, horizon)
                if 'error' not in xgb_pred:
                    predictions.append(PredictionResult(
                        model_name='XGBoost',
                        predicted_change_pct=xgb_pred['predicted_change_pct'],
                        confidence=self._calculate_confidence('xgboost'),
                        direction=xgb_pred.get('direction', 'Unknown'),
                        details=xgb_pred
                    ))
                    model_results['xgboost'] = xgb_pred
            except Exception as e:
                model_results['xgboost'] = {'error': str(e)}

        # Get SMA prediction
        try:
            sma_pred = self.sma_model.predict(prices, horizon)
            predictions.append(PredictionResult(
                model_name='SMA Trend',
                predicted_change_pct=sma_pred['predicted_change_pct'],
                confidence=0.5,  # Lower confidence for simple model
                direction='Up' if sma_pred['predicted_change_pct'] > 0 else 'Down',
                details=sma_pred
            ))
            model_results['sma'] = sma_pred
        except Exception as e:
            model_results['sma'] = {'error': str(e)}

        # Calculate ensemble prediction
        ensemble_result = self._combine_predictions(predictions, prices.iloc[-1])

        return {
            'ensemble': ensemble_result,
            'individual_models': model_results,
            'horizon': horizon,
            'current_price': prices.iloc[-1]
        }

    def _combine_predictions(
        self,
        predictions: List[PredictionResult],
        current_price: float
    ) -> Dict[str, Any]:
        """Combine individual predictions into ensemble."""
        if not predictions:
            return {
                'predicted_change_pct': 0,
                'confidence': 0,
                'direction': 'Unknown',
                'agreement': 0
            }

        # Weight predictions by model performance
        weighted_changes = []
        total_weight = 0

        for pred in predictions:
            model_key = pred.model_name.lower().replace(' ', '_').replace('_trend', '')
            weight = self.weights.get(model_key, 0.2)

            # Adjust weight by confidence
            adjusted_weight = weight * pred.confidence
            weighted_changes.append(pred.predicted_change_pct * adjusted_weight)
            total_weight += adjusted_weight

        # Calculate weighted average
        if total_weight > 0:
            ensemble_change = sum(weighted_changes) / total_weight
        else:
            ensemble_change = np.mean([p.predicted_change_pct for p in predictions])

        # Calculate agreement (how many models agree on direction)
        directions = [p.direction for p in predictions]
        up_count = directions.count('Up')
        down_count = directions.count('Down')
        agreement = max(up_count, down_count) / len(directions)

        # Determine final direction
        if up_count > down_count:
            direction = 'Up'
        elif down_count > up_count:
            direction = 'Down'
        else:
            direction = 'Uncertain'

        # Calculate confidence
        confidence = self._calculate_ensemble_confidence(predictions, agreement)

        # Calculate predicted price
        predicted_price = current_price * (1 + ensemble_change / 100)

        return {
            'predicted_change_pct': ensemble_change,
            'predicted_price': predicted_price,
            'confidence': confidence,
            'direction': direction,
            'agreement': agreement,
            'models_used': len(predictions),
            'signal': self._generate_signal(ensemble_change, confidence, agreement)
        }

    def _calculate_confidence(self, model_name: str) -> float:
        """Calculate confidence based on training metrics."""
        metrics = self.training_metrics.get(model_name, {})

        if not metrics or 'error' in metrics:
            return 0.5

        if model_name == 'xgboost':
            # Use direction accuracy if available
            accuracy = metrics.get('direction_accuracy', 0.5)
            return min(accuracy, 0.9)  # Cap at 90%

        elif model_name == 'lstm':
            # Use validation loss to estimate confidence
            val_loss = metrics.get('final_val_loss', 1.0)
            # Lower loss = higher confidence
            return max(0.3, min(0.9, 1 - val_loss))

        return 0.5

    def _calculate_ensemble_confidence(
        self,
        predictions: List[PredictionResult],
        agreement: float
    ) -> float:
        """Calculate overall ensemble confidence."""
        if not predictions:
            return 0

        # Average individual confidences
        avg_confidence = np.mean([p.confidence for p in predictions])

        # Boost confidence when models agree
        agreement_bonus = (agreement - 0.5) * 0.2

        # Penalize when predictions diverge significantly
        changes = [p.predicted_change_pct for p in predictions]
        std_changes = np.std(changes) if len(changes) > 1 else 0
        divergence_penalty = min(0.2, std_changes / 50)

        final_confidence = avg_confidence + agreement_bonus - divergence_penalty
        return max(0.1, min(0.95, final_confidence))

    def _generate_signal(
        self,
        change_pct: float,
        confidence: float,
        agreement: float
    ) -> str:
        """Generate trading signal from ensemble prediction."""
        if confidence < 0.4 or agreement < 0.5:
            return 'Hold - Low confidence'

        if change_pct > 10 and confidence > 0.6:
            return 'Strong Buy'
        elif change_pct > 5 and confidence > 0.5:
            return 'Buy'
        elif change_pct > 0:
            return 'Weak Buy'
        elif change_pct > -5:
            return 'Weak Sell'
        elif change_pct > -10 and confidence > 0.5:
            return 'Sell'
        else:
            return 'Strong Sell'

    def get_prediction_summary(self, result: Dict[str, Any]) -> str:
        """Generate human-readable prediction summary."""
        ensemble = result.get('ensemble', {})

        summary_parts = []

        direction = ensemble.get('direction', 'Unknown')
        change = ensemble.get('predicted_change_pct', 0)
        confidence = ensemble.get('confidence', 0) * 100
        agreement = ensemble.get('agreement', 0) * 100
        signal = ensemble.get('signal', 'Unknown')

        summary_parts.append(f"Prediction: {direction} by {abs(change):.1f}%")
        summary_parts.append(f"Signal: {signal}")
        summary_parts.append(f"Confidence: {confidence:.0f}%")
        summary_parts.append(f"Model Agreement: {agreement:.0f}%")

        return " | ".join(summary_parts)

    def predict_multiple_horizons(
        self,
        df: pd.DataFrame,
        horizons: List[int] = None
    ) -> Dict[int, Dict[str, Any]]:
        """
        Make predictions for multiple time horizons.

        Args:
            df: DataFrame with OHLCV data
            horizons: List of prediction horizons

        Returns:
            Dictionary mapping horizon to prediction result
        """
        if horizons is None:
            horizons = [7, 30, 90]

        results = {}
        for horizon in horizons:
            results[horizon] = self.predict(df, horizon)

        return results

    def get_model_contributions(self, result: Dict[str, Any]) -> Dict[str, float]:
        """Get contribution of each model to the ensemble prediction."""
        individual = result.get('individual_models', {})
        contributions = {}

        for model_name, pred in individual.items():
            if 'error' not in pred:
                change = pred.get('predicted_change_pct', 0)
                weight = self.weights.get(model_name, 0.2)
                contributions[model_name] = {
                    'prediction': change,
                    'weight': weight,
                    'contribution': change * weight
                }

        return contributions
