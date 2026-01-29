"""
Machine learning models for stock price prediction.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, Tuple, List
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import warnings
warnings.filterwarnings('ignore')

from config import ML_PARAMS


class LSTMModel:
    """
    LSTM model for time series price prediction.
    """

    def __init__(self, params: Optional[Dict] = None):
        """
        Initialize LSTM model.

        Args:
            params: Model parameters
        """
        self.params = params or ML_PARAMS
        self.model = None
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.lookback = self.params['lstm_lookback']
        self.is_trained = False

    def _create_model(self, input_shape: Tuple[int, int]):
        """Create LSTM model architecture."""
        try:
            from tensorflow.keras.models import Sequential
            from tensorflow.keras.layers import LSTM, Dense, Dropout

            model = Sequential([
                LSTM(self.params['lstm_units'], return_sequences=True, input_shape=input_shape),
                Dropout(0.2),
                LSTM(self.params['lstm_units'], return_sequences=False),
                Dropout(0.2),
                Dense(25),
                Dense(1)
            ])

            model.compile(optimizer='adam', loss='mean_squared_error')
            return model

        except ImportError:
            print("TensorFlow not available. LSTM model disabled.")
            return None

    def _prepare_data(self, data: pd.Series) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare data for LSTM training."""
        # Scale data
        scaled_data = self.scaler.fit_transform(data.values.reshape(-1, 1))

        # Create sequences
        X, y = [], []
        for i in range(self.lookback, len(scaled_data)):
            X.append(scaled_data[i - self.lookback:i, 0])
            y.append(scaled_data[i, 0])

        return np.array(X), np.array(y)

    def train(self, prices: pd.Series, verbose: int = 0) -> Dict[str, float]:
        """
        Train the LSTM model.

        Args:
            prices: Historical prices
            verbose: Training verbosity

        Returns:
            Dictionary with training metrics
        """
        if len(prices) < self.lookback + 50:
            return {'error': 'Insufficient data for training'}

        X, y = self._prepare_data(prices)
        X = X.reshape((X.shape[0], X.shape[1], 1))

        # Train-test split
        split = int(len(X) * (1 - self.params['test_size']))
        X_train, X_test = X[:split], X[split:]
        y_train, y_test = y[:split], y[split:]

        # Create and train model
        self.model = self._create_model((X.shape[1], 1))
        if self.model is None:
            return {'error': 'Model creation failed'}

        history = self.model.fit(
            X_train, y_train,
            epochs=self.params['lstm_epochs'],
            batch_size=self.params['lstm_batch_size'],
            validation_data=(X_test, y_test),
            verbose=verbose
        )

        self.is_trained = True

        # Evaluate
        predictions = self.model.predict(X_test, verbose=0)
        predictions = self.scaler.inverse_transform(predictions)
        y_test_actual = self.scaler.inverse_transform(y_test.reshape(-1, 1))

        rmse = np.sqrt(mean_squared_error(y_test_actual, predictions))
        mae = mean_absolute_error(y_test_actual, predictions)

        return {
            'rmse': rmse,
            'mae': mae,
            'final_loss': history.history['loss'][-1],
            'final_val_loss': history.history['val_loss'][-1]
        }

    def predict(self, prices: pd.Series, horizon: int = 7) -> Dict[str, Any]:
        """
        Make price predictions.

        Args:
            prices: Historical prices
            horizon: Prediction horizon in days

        Returns:
            Dictionary with predictions
        """
        if not self.is_trained or self.model is None:
            # Quick train if not trained
            self.train(prices, verbose=0)

        if not self.is_trained:
            return {'error': 'Model not trained'}

        # Prepare last sequence
        scaled_data = self.scaler.transform(prices.values.reshape(-1, 1))
        last_sequence = scaled_data[-self.lookback:].reshape(1, self.lookback, 1)

        predictions = []
        current_sequence = last_sequence.copy()

        for _ in range(horizon):
            pred = self.model.predict(current_sequence, verbose=0)
            predictions.append(pred[0, 0])

            # Update sequence
            current_sequence = np.roll(current_sequence, -1, axis=1)
            current_sequence[0, -1, 0] = pred[0, 0]

        # Inverse transform predictions
        predictions = self.scaler.inverse_transform(np.array(predictions).reshape(-1, 1))

        return {
            'predictions': predictions.flatten().tolist(),
            'current_price': prices.iloc[-1],
            'predicted_change_pct': (predictions[-1, 0] / prices.iloc[-1] - 1) * 100,
            'horizon': horizon
        }


class XGBoostModel:
    """
    XGBoost model for feature-based price prediction.
    """

    def __init__(self, params: Optional[Dict] = None):
        """
        Initialize XGBoost model.

        Args:
            params: Model parameters
        """
        self.params = params or ML_PARAMS
        self.model = None
        self.feature_names = []
        self.is_trained = False

    def _create_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create features for prediction."""
        features = pd.DataFrame(index=df.index)

        # Price-based features
        features['return_1d'] = df['Close'].pct_change(1)
        features['return_5d'] = df['Close'].pct_change(5)
        features['return_20d'] = df['Close'].pct_change(20)

        # Moving averages
        for period in [5, 10, 20, 50]:
            features[f'sma_{period}'] = df['Close'].rolling(period).mean()
            features[f'sma_ratio_{period}'] = df['Close'] / features[f'sma_{period}']

        # Volatility
        features['volatility_20d'] = df['Close'].pct_change().rolling(20).std()

        # Volume features
        features['volume_sma_20'] = df['Volume'].rolling(20).mean()
        features['volume_ratio'] = df['Volume'] / features['volume_sma_20']

        # Price position
        features['high_low_ratio'] = (df['Close'] - df['Low']) / (df['High'] - df['Low'] + 0.001)

        # RSI
        delta = df['Close'].diff()
        gain = delta.where(delta > 0, 0).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / (loss + 0.001)
        features['rsi'] = 100 - (100 / (1 + rs))

        # MACD
        ema_12 = df['Close'].ewm(span=12).mean()
        ema_26 = df['Close'].ewm(span=26).mean()
        features['macd'] = ema_12 - ema_26
        features['macd_signal'] = features['macd'].ewm(span=9).mean()

        # Bollinger Bands position
        sma_20 = df['Close'].rolling(20).mean()
        std_20 = df['Close'].rolling(20).std()
        features['bb_position'] = (df['Close'] - (sma_20 - 2 * std_20)) / (4 * std_20 + 0.001)

        # Day of week (if datetime index)
        if isinstance(df.index, pd.DatetimeIndex):
            features['day_of_week'] = df.index.dayofweek

        return features.dropna()

    def _create_target(self, prices: pd.Series, horizon: int) -> pd.Series:
        """Create target variable (future return)."""
        return prices.pct_change(horizon).shift(-horizon)

    def train(self, df: pd.DataFrame, horizon: int = 7) -> Dict[str, float]:
        """
        Train the XGBoost model.

        Args:
            df: DataFrame with OHLCV data
            horizon: Prediction horizon

        Returns:
            Dictionary with training metrics
        """
        try:
            import xgboost as xgb
        except ImportError:
            return {'error': 'XGBoost not installed'}

        # Create features and target
        features = self._create_features(df)
        target = self._create_target(df['Close'], horizon)

        # Align features and target
        common_idx = features.index.intersection(target.dropna().index)
        X = features.loc[common_idx]
        y = target.loc[common_idx]

        if len(X) < 100:
            return {'error': 'Insufficient data for training'}

        self.feature_names = X.columns.tolist()

        # Train-test split
        split = int(len(X) * (1 - self.params['test_size']))
        X_train, X_test = X.iloc[:split], X.iloc[split:]
        y_train, y_test = y.iloc[:split], y.iloc[split:]

        # Train model
        self.model = xgb.XGBRegressor(
            n_estimators=self.params['xgboost_n_estimators'],
            max_depth=self.params['xgboost_max_depth'],
            learning_rate=0.1,
            random_state=42
        )

        self.model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)
        self.is_trained = True

        # Evaluate
        predictions = self.model.predict(X_test)
        rmse = np.sqrt(mean_squared_error(y_test, predictions))
        mae = mean_absolute_error(y_test, predictions)

        # Direction accuracy
        direction_correct = ((predictions > 0) == (y_test > 0)).mean()

        return {
            'rmse': rmse,
            'mae': mae,
            'direction_accuracy': direction_correct,
            'feature_importance': dict(zip(self.feature_names,
                                          self.model.feature_importances_.tolist()))
        }

    def predict(self, df: pd.DataFrame, horizon: int = 7) -> Dict[str, Any]:
        """
        Make price predictions.

        Args:
            df: DataFrame with OHLCV data
            horizon: Prediction horizon

        Returns:
            Dictionary with predictions
        """
        if not self.is_trained or self.model is None:
            self.train(df, horizon)

        if not self.is_trained:
            return {'error': 'Model not trained'}

        # Create features for latest data
        features = self._create_features(df)
        if features.empty:
            return {'error': 'Could not create features'}

        latest_features = features.iloc[[-1]]
        predicted_return = self.model.predict(latest_features)[0]

        current_price = df['Close'].iloc[-1]
        predicted_price = current_price * (1 + predicted_return)

        return {
            'predicted_return': predicted_return * 100,
            'current_price': current_price,
            'predicted_price': predicted_price,
            'predicted_change_pct': predicted_return * 100,
            'direction': 'Up' if predicted_return > 0 else 'Down',
            'horizon': horizon
        }


class ARIMAModel:
    """
    ARIMA model for statistical time series forecasting.
    """

    def __init__(self, order: Tuple[int, int, int] = (5, 1, 0)):
        """
        Initialize ARIMA model.

        Args:
            order: ARIMA order (p, d, q)
        """
        self.order = order
        self.model = None
        self.is_fitted = False

    def fit(self, prices: pd.Series) -> Dict[str, Any]:
        """
        Fit ARIMA model.

        Args:
            prices: Historical prices

        Returns:
            Dictionary with fit statistics
        """
        try:
            from statsmodels.tsa.arima.model import ARIMA
        except ImportError:
            return {'error': 'statsmodels not installed'}

        try:
            self.model = ARIMA(prices, order=self.order)
            self.result = self.model.fit()
            self.is_fitted = True

            return {
                'aic': self.result.aic,
                'bic': self.result.bic,
                'order': self.order
            }

        except Exception as e:
            return {'error': str(e)}

    def predict(self, prices: pd.Series, horizon: int = 7) -> Dict[str, Any]:
        """
        Make predictions.

        Args:
            prices: Historical prices
            horizon: Prediction horizon

        Returns:
            Dictionary with predictions
        """
        if not self.is_fitted:
            self.fit(prices)

        if not self.is_fitted:
            return {'error': 'Model not fitted'}

        try:
            forecast = self.result.forecast(steps=horizon)
            current_price = prices.iloc[-1]

            return {
                'predictions': forecast.tolist(),
                'current_price': current_price,
                'predicted_change_pct': (forecast.iloc[-1] / current_price - 1) * 100,
                'horizon': horizon
            }

        except Exception as e:
            return {'error': str(e)}


class SimpleMovingAverageModel:
    """
    Simple baseline model using moving average crossover.
    """

    def __init__(self, short_window: int = 20, long_window: int = 50):
        """
        Initialize SMA model.

        Args:
            short_window: Short moving average period
            long_window: Long moving average period
        """
        self.short_window = short_window
        self.long_window = long_window

    def predict(self, prices: pd.Series, horizon: int = 7) -> Dict[str, Any]:
        """
        Make predictions based on trend.

        Args:
            prices: Historical prices
            horizon: Prediction horizon

        Returns:
            Dictionary with predictions
        """
        short_ma = prices.rolling(self.short_window).mean()
        long_ma = prices.rolling(self.long_window).mean()

        current_price = prices.iloc[-1]
        current_short = short_ma.iloc[-1]
        current_long = long_ma.iloc[-1]

        # Calculate trend strength
        trend_strength = (current_short - current_long) / current_long

        # Simple projection based on recent trend
        recent_returns = prices.pct_change(5).iloc[-1]
        projected_return = (recent_returns * 0.5 + trend_strength * 0.5) * (horizon / 5)

        predicted_price = current_price * (1 + projected_return)

        return {
            'predicted_price': predicted_price,
            'current_price': current_price,
            'predicted_change_pct': projected_return * 100,
            'trend': 'Bullish' if current_short > current_long else 'Bearish',
            'trend_strength': abs(trend_strength) * 100,
            'horizon': horizon
        }
