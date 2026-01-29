"""
Risk-adjusted scoring system for investment recommendations.
"""

import numpy as np
from typing import Dict, Any, Optional, List
from dataclasses import dataclass

from config import SCORING_WEIGHTS, RECOMMENDATION_THRESHOLDS


@dataclass
class ScoreBreakdown:
    """Container for score breakdown."""
    financial_health: float
    growth: float
    valuation: float
    technical_momentum: float
    prediction_confidence: float
    total_score: float
    recommendation: str
    confidence_level: str
    reasoning: List[str]


class RiskScorer:
    """
    Multi-factor scoring system for stock analysis.
    """

    def __init__(
        self,
        weights: Optional[Dict[str, float]] = None,
        thresholds: Optional[Dict[str, int]] = None
    ):
        """
        Initialize risk scorer.

        Args:
            weights: Custom weights for score categories
            thresholds: Custom recommendation thresholds
        """
        self.weights = weights or SCORING_WEIGHTS
        self.thresholds = thresholds or RECOMMENDATION_THRESHOLDS

    def calculate_score(
        self,
        fundamental_analysis: Dict[str, Any],
        technical_analysis: Dict[str, Any],
        valuation_analysis: Dict[str, Any],
        prediction_result: Optional[Dict[str, Any]] = None,
        news_sentiment: Optional[Dict[str, Any]] = None
    ) -> ScoreBreakdown:
        """
        Calculate comprehensive investment score.

        Args:
            fundamental_analysis: Results from fundamental analyzer
            technical_analysis: Results from technical analyzer
            valuation_analysis: Results from valuation model
            prediction_result: Results from prediction ensemble
            news_sentiment: News sentiment analysis

        Returns:
            ScoreBreakdown with detailed scores and recommendation
        """
        reasoning = []

        # Financial Health Score (25%)
        financial_score = self._calculate_financial_health_score(
            fundamental_analysis, reasoning
        )

        # Growth Score (20%)
        growth_score = self._calculate_growth_score(
            fundamental_analysis, reasoning
        )

        # Valuation Score (25%)
        valuation_score = self._calculate_valuation_score(
            fundamental_analysis, valuation_analysis, reasoning
        )

        # Technical Momentum Score (15%)
        technical_score = self._calculate_technical_score(
            technical_analysis, reasoning
        )

        # Prediction Confidence Score (15%)
        prediction_score = self._calculate_prediction_score(
            prediction_result, reasoning
        )

        # Calculate weighted total
        total_score = (
            financial_score * self.weights['financial_health'] +
            growth_score * self.weights['growth'] +
            valuation_score * self.weights['valuation'] +
            technical_score * self.weights['technical_momentum'] +
            prediction_score * self.weights['prediction_confidence']
        )

        # Adjust for news sentiment if available
        if news_sentiment:
            sentiment_adjustment = self._get_sentiment_adjustment(news_sentiment)
            total_score = total_score * (1 + sentiment_adjustment)
            if abs(sentiment_adjustment) > 0.05:
                direction = 'positive' if sentiment_adjustment > 0 else 'negative'
                reasoning.append(f"News sentiment is {direction}, adjusting score")

        # Cap score between 0-100
        total_score = max(0, min(100, total_score))

        # Get recommendation
        recommendation = self._get_recommendation(total_score)
        confidence_level = self._get_confidence_level(
            financial_score, growth_score, valuation_score,
            technical_score, prediction_score
        )

        return ScoreBreakdown(
            financial_health=financial_score,
            growth=growth_score,
            valuation=valuation_score,
            technical_momentum=technical_score,
            prediction_confidence=prediction_score,
            total_score=total_score,
            recommendation=recommendation,
            confidence_level=confidence_level,
            reasoning=reasoning
        )

    def _calculate_financial_health_score(
        self,
        fundamental: Dict[str, Any],
        reasoning: List[str]
    ) -> float:
        """Calculate financial health score component."""
        scores = []

        # Get financial health metrics
        health = fundamental.get('financial_health', {})

        # Debt-to-Equity
        de_data = health.get('debt_to_equity', {})
        if de_data.get('value') is not None:
            de = de_data['value']
            if de < 0.5:
                scores.append(90)
                reasoning.append(f"Low debt (D/E: {de:.2f})")
            elif de < 1:
                scores.append(70)
            elif de < 2:
                scores.append(50)
            else:
                scores.append(30)
                reasoning.append(f"High debt concerns (D/E: {de:.2f})")

        # Current Ratio
        cr_data = health.get('current_ratio', {})
        if cr_data.get('value') is not None:
            cr = cr_data['value']
            if cr > 2:
                scores.append(85)
            elif cr > 1.5:
                scores.append(70)
            elif cr > 1:
                scores.append(50)
            else:
                scores.append(30)
                reasoning.append(f"Liquidity concerns (CR: {cr:.2f})")

        # Profitability from fundamental analysis
        profitability = fundamental.get('profitability', {})

        # ROE
        roe_data = profitability.get('roe', {})
        if roe_data.get('value') is not None:
            roe = roe_data['value']
            roe_pct = roe * 100 if roe < 1 else roe
            if roe_pct > 20:
                scores.append(90)
                reasoning.append(f"Excellent ROE ({roe_pct:.1f}%)")
            elif roe_pct > 15:
                scores.append(75)
            elif roe_pct > 10:
                scores.append(55)
            else:
                scores.append(35)

        # Profit Margin
        margin_data = profitability.get('profit_margin', {})
        if margin_data.get('value') is not None:
            margin = margin_data['value']
            if margin > 0.2:
                scores.append(85)
            elif margin > 0.1:
                scores.append(70)
            elif margin > 0.05:
                scores.append(50)
            else:
                scores.append(30)

        return np.mean(scores) if scores else 50

    def _calculate_growth_score(
        self,
        fundamental: Dict[str, Any],
        reasoning: List[str]
    ) -> float:
        """Calculate growth score component."""
        scores = []
        growth = fundamental.get('growth', {})

        # Revenue Growth
        rev_data = growth.get('revenue_growth', {})
        if rev_data.get('value') is not None:
            rev_growth = rev_data['value']
            g = rev_growth * 100 if -1 < rev_growth < 1 else rev_growth
            if g > 20:
                scores.append(90)
                reasoning.append(f"Strong revenue growth ({g:.1f}%)")
            elif g > 10:
                scores.append(75)
            elif g > 0:
                scores.append(55)
            else:
                scores.append(30)
                reasoning.append(f"Revenue declining ({g:.1f}%)")

        # Earnings Growth
        earn_data = growth.get('earnings_growth', {})
        if earn_data.get('value') is not None:
            earn_growth = earn_data['value']
            g = earn_growth * 100 if -1 < earn_growth < 1 else earn_growth
            if g > 25:
                scores.append(90)
                reasoning.append(f"Strong earnings growth ({g:.1f}%)")
            elif g > 15:
                scores.append(75)
            elif g > 0:
                scores.append(55)
            else:
                scores.append(25)

        return np.mean(scores) if scores else 50

    def _calculate_valuation_score(
        self,
        fundamental: Dict[str, Any],
        valuation: Dict[str, Any],
        reasoning: List[str]
    ) -> float:
        """Calculate valuation score component."""
        scores = []

        # P/E Ratio
        val_data = fundamental.get('valuation', {})
        pe_data = val_data.get('pe_ratio', {})
        if pe_data.get('value') is not None:
            pe = pe_data['value']
            if pe > 0:
                if pe < 15:
                    scores.append(85)
                    reasoning.append(f"Attractive P/E ({pe:.1f})")
                elif pe < 25:
                    scores.append(65)
                elif pe < 40:
                    scores.append(45)
                else:
                    scores.append(25)
                    reasoning.append(f"High P/E concerns ({pe:.1f})")

        # P/B Ratio
        pb_data = val_data.get('pb_ratio', {})
        if pb_data.get('value') is not None:
            pb = pb_data['value']
            if pb < 1:
                scores.append(90)
                reasoning.append(f"Trading below book value (P/B: {pb:.2f})")
            elif pb < 3:
                scores.append(70)
            elif pb < 5:
                scores.append(50)
            else:
                scores.append(30)

        # Margin of Safety from valuation analysis
        if valuation:
            mos = valuation.get('margin_of_safety')
            if mos is not None:
                if mos > 30:
                    scores.append(95)
                    reasoning.append(f"Significant margin of safety ({mos:.1f}%)")
                elif mos > 15:
                    scores.append(75)
                elif mos > 0:
                    scores.append(55)
                elif mos > -15:
                    scores.append(40)
                else:
                    scores.append(20)
                    reasoning.append(f"Appears overvalued by {abs(mos):.1f}%")

        return np.mean(scores) if scores else 50

    def _calculate_technical_score(
        self,
        technical: Dict[str, Any],
        reasoning: List[str]
    ) -> float:
        """Calculate technical momentum score component."""
        if not technical:
            return 50

        scores = []

        # Moving Average Analysis
        ma = technical.get('moving_averages', {})
        if ma.get('score'):
            scores.append(ma['score'])
            trend = ma.get('trend', '')
            if 'Strong' in trend:
                reasoning.append(f"Technical trend: {trend}")

        # RSI
        rsi = technical.get('rsi', {})
        if rsi.get('score'):
            scores.append(rsi['score'])
            if rsi.get('signal') in ['Overbought', 'Oversold']:
                reasoning.append(f"RSI signal: {rsi['signal']}")

        # MACD
        macd = technical.get('macd', {})
        if macd.get('score'):
            scores.append(macd['score'])
            if macd.get('crossover'):
                reasoning.append(f"MACD: {macd['crossover']}")

        # Bollinger Bands
        bb = technical.get('bollinger_bands', {})
        if bb.get('score'):
            scores.append(bb['score'])

        # Volume
        vol = technical.get('volume', {})
        if vol.get('score'):
            scores.append(vol['score'])

        # Overall trend
        trend = technical.get('trend', {})
        if trend.get('score'):
            scores.append(trend['score'])

        return np.mean(scores) if scores else 50

    def _calculate_prediction_score(
        self,
        prediction: Optional[Dict[str, Any]],
        reasoning: List[str]
    ) -> float:
        """Calculate prediction confidence score component."""
        if not prediction:
            return 50

        ensemble = prediction.get('ensemble', {})

        # Base score from predicted direction and magnitude
        pred_change = ensemble.get('predicted_change_pct', 0)
        confidence = ensemble.get('confidence', 0.5)
        agreement = ensemble.get('agreement', 0.5)

        # Start with neutral score
        score = 50

        # Adjust based on prediction
        if pred_change > 10:
            score = 80
        elif pred_change > 5:
            score = 70
        elif pred_change > 0:
            score = 60
        elif pred_change > -5:
            score = 40
        elif pred_change > -10:
            score = 30
        else:
            score = 20

        # Adjust by confidence
        score = score * (0.5 + confidence * 0.5)

        # Adjust by model agreement
        score = score * (0.7 + agreement * 0.3)

        if confidence > 0.7 and agreement > 0.8:
            direction = 'upside' if pred_change > 0 else 'downside'
            reasoning.append(f"ML models predict {abs(pred_change):.1f}% {direction}")

        return max(0, min(100, score))

    def _get_sentiment_adjustment(self, news_sentiment: Dict[str, Any]) -> float:
        """Get score adjustment based on news sentiment."""
        sentiment = news_sentiment.get('sentiment', 'Neutral')
        score = news_sentiment.get('score', 0.5)

        if sentiment == 'Positive':
            return (score - 0.5) * 0.1  # Up to +5%
        elif sentiment == 'Negative':
            return (score - 0.5) * 0.1  # Up to -5%
        return 0

    def _get_recommendation(self, score: float) -> str:
        """Convert score to recommendation."""
        if score >= self.thresholds['strong_buy']:
            return 'Strong Buy'
        elif score >= self.thresholds['buy']:
            return 'Buy'
        elif score >= self.thresholds['hold']:
            return 'Hold'
        elif score >= self.thresholds['sell']:
            return 'Sell'
        else:
            return 'Strong Sell'

    def _get_confidence_level(
        self,
        financial: float,
        growth: float,
        valuation: float,
        technical: float,
        prediction: float
    ) -> str:
        """Determine confidence level based on score consistency."""
        scores = [financial, growth, valuation, technical, prediction]
        std = np.std(scores)

        if std < 10:
            return 'High'
        elif std < 20:
            return 'Medium'
        else:
            return 'Low'

    def get_score_breakdown(self, score: ScoreBreakdown) -> Dict[str, Any]:
        """Get detailed score breakdown for display."""
        return {
            'components': {
                'Financial Health': {
                    'score': score.financial_health,
                    'weight': self.weights['financial_health'] * 100,
                    'contribution': score.financial_health * self.weights['financial_health']
                },
                'Growth': {
                    'score': score.growth,
                    'weight': self.weights['growth'] * 100,
                    'contribution': score.growth * self.weights['growth']
                },
                'Valuation': {
                    'score': score.valuation,
                    'weight': self.weights['valuation'] * 100,
                    'contribution': score.valuation * self.weights['valuation']
                },
                'Technical Momentum': {
                    'score': score.technical_momentum,
                    'weight': self.weights['technical_momentum'] * 100,
                    'contribution': score.technical_momentum * self.weights['technical_momentum']
                },
                'Prediction': {
                    'score': score.prediction_confidence,
                    'weight': self.weights['prediction_confidence'] * 100,
                    'contribution': score.prediction_confidence * self.weights['prediction_confidence']
                }
            },
            'total_score': score.total_score,
            'recommendation': score.recommendation,
            'confidence': score.confidence_level,
            'key_factors': score.reasoning
        }

    def generate_recommendation_report(self, score: ScoreBreakdown) -> str:
        """Generate human-readable recommendation report."""
        lines = [
            "=" * 50,
            "INVESTMENT RECOMMENDATION REPORT",
            "=" * 50,
            "",
            f"Overall Score: {score.total_score:.1f}/100",
            f"Recommendation: {score.recommendation}",
            f"Confidence Level: {score.confidence_level}",
            "",
            "Score Breakdown:",
            f"  Financial Health: {score.financial_health:.1f} (weight: {self.weights['financial_health']*100:.0f}%)",
            f"  Growth: {score.growth:.1f} (weight: {self.weights['growth']*100:.0f}%)",
            f"  Valuation: {score.valuation:.1f} (weight: {self.weights['valuation']*100:.0f}%)",
            f"  Technical Momentum: {score.technical_momentum:.1f} (weight: {self.weights['technical_momentum']*100:.0f}%)",
            f"  Prediction Confidence: {score.prediction_confidence:.1f} (weight: {self.weights['prediction_confidence']*100:.0f}%)",
            "",
            "Key Factors:",
        ]

        for reason in score.reasoning:
            lines.append(f"  - {reason}")

        lines.append("")
        lines.append("=" * 50)

        return "\n".join(lines)
