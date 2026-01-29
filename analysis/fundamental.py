"""
Fundamental analysis module for stock evaluation.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, List
from dataclasses import dataclass

from config import FUNDAMENTAL_THRESHOLDS


@dataclass
class FundamentalMetrics:
    """Container for fundamental metrics."""
    pe_ratio: Optional[float] = None
    pb_ratio: Optional[float] = None
    ps_ratio: Optional[float] = None
    peg_ratio: Optional[float] = None
    eps: Optional[float] = None
    roe: Optional[float] = None
    roa: Optional[float] = None
    roce: Optional[float] = None
    debt_to_equity: Optional[float] = None
    current_ratio: Optional[float] = None
    quick_ratio: Optional[float] = None
    dividend_yield: Optional[float] = None
    payout_ratio: Optional[float] = None
    profit_margin: Optional[float] = None
    operating_margin: Optional[float] = None
    gross_margin: Optional[float] = None
    revenue_growth: Optional[float] = None
    earnings_growth: Optional[float] = None
    book_value: Optional[float] = None


class FundamentalAnalyzer:
    """
    Analyzes fundamental metrics of stocks.
    """

    def __init__(self, thresholds: Optional[Dict] = None):
        """
        Initialize the analyzer.

        Args:
            thresholds: Custom thresholds for metric evaluation
        """
        self.thresholds = thresholds or FUNDAMENTAL_THRESHOLDS

    def analyze(self, metrics: Dict[str, Any]) -> Dict[str, Any]:
        """
        Perform comprehensive fundamental analysis.

        Args:
            metrics: Dictionary of raw metrics from data fetcher

        Returns:
            Dictionary with analysis results and scores
        """
        analysis = {
            'valuation': self._analyze_valuation(metrics),
            'profitability': self._analyze_profitability(metrics),
            'financial_health': self._analyze_financial_health(metrics),
            'growth': self._analyze_growth(metrics),
            'dividends': self._analyze_dividends(metrics),
        }

        # Calculate overall score
        scores = []
        for category, data in analysis.items():
            if 'score' in data:
                scores.append(data['score'])

        analysis['overall_score'] = np.mean(scores) if scores else 50
        analysis['summary'] = self._generate_summary(analysis)

        return analysis

    def _analyze_valuation(self, metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze valuation metrics."""
        pe = metrics.get('pe_ratio')
        pb = metrics.get('pb_ratio')
        ps = metrics.get('ps_ratio')
        peg = metrics.get('peg_ratio')

        results = {
            'pe_ratio': {
                'value': pe,
                'rating': self._rate_pe(pe),
                'explanation': self._explain_pe(pe)
            },
            'pb_ratio': {
                'value': pb,
                'rating': self._rate_pb(pb),
                'explanation': self._explain_pb(pb)
            },
            'ps_ratio': {
                'value': ps,
                'rating': self._rate_metric(ps, lower_is_better=True, thresholds=(2, 5)),
                'explanation': f"P/S ratio of {ps:.2f}" if ps else "N/A"
            },
            'peg_ratio': {
                'value': peg,
                'rating': self._rate_peg(peg),
                'explanation': self._explain_peg(peg)
            }
        }

        # Calculate valuation score
        score = self._calculate_valuation_score(pe, pb, peg)
        results['score'] = score

        return results

    def _analyze_profitability(self, metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze profitability metrics."""
        roe = metrics.get('roe')
        roa = metrics.get('roa')
        profit_margin = metrics.get('profit_margin')
        operating_margin = metrics.get('operating_margin')
        gross_margin = metrics.get('gross_margin')

        results = {
            'roe': {
                'value': roe,
                'rating': self._rate_roe(roe),
                'explanation': self._explain_roe(roe)
            },
            'roa': {
                'value': roa,
                'rating': self._rate_roa(roa),
                'explanation': self._explain_roa(roa)
            },
            'profit_margin': {
                'value': profit_margin,
                'rating': self._rate_margin(profit_margin),
                'explanation': f"Net profit margin of {profit_margin*100:.1f}%" if profit_margin else "N/A"
            },
            'operating_margin': {
                'value': operating_margin,
                'rating': self._rate_margin(operating_margin),
                'explanation': f"Operating margin of {operating_margin*100:.1f}%" if operating_margin else "N/A"
            },
            'gross_margin': {
                'value': gross_margin,
                'rating': self._rate_margin(gross_margin, thresholds=(0.3, 0.5)),
                'explanation': f"Gross margin of {gross_margin*100:.1f}%" if gross_margin else "N/A"
            }
        }

        # Calculate profitability score
        score = self._calculate_profitability_score(roe, roa, profit_margin)
        results['score'] = score

        return results

    def _analyze_financial_health(self, metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze financial health metrics."""
        de = metrics.get('debt_to_equity')
        current = metrics.get('current_ratio')
        quick = metrics.get('quick_ratio')

        # Convert debt_to_equity from percentage if needed
        if de is not None and de > 10:
            de = de / 100

        results = {
            'debt_to_equity': {
                'value': de,
                'rating': self._rate_debt_to_equity(de),
                'explanation': self._explain_debt_to_equity(de)
            },
            'current_ratio': {
                'value': current,
                'rating': self._rate_current_ratio(current),
                'explanation': self._explain_current_ratio(current)
            },
            'quick_ratio': {
                'value': quick,
                'rating': self._rate_quick_ratio(quick),
                'explanation': f"Quick ratio of {quick:.2f}" if quick else "N/A"
            }
        }

        # Calculate financial health score
        score = self._calculate_health_score(de, current, quick)
        results['score'] = score

        return results

    def _analyze_growth(self, metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze growth metrics."""
        revenue_growth = metrics.get('revenue_growth')
        earnings_growth = metrics.get('earnings_growth')

        results = {
            'revenue_growth': {
                'value': revenue_growth,
                'rating': self._rate_growth(revenue_growth),
                'explanation': self._explain_growth(revenue_growth, 'Revenue')
            },
            'earnings_growth': {
                'value': earnings_growth,
                'rating': self._rate_growth(earnings_growth),
                'explanation': self._explain_growth(earnings_growth, 'Earnings')
            }
        }

        # Calculate growth score
        score = self._calculate_growth_score(revenue_growth, earnings_growth)
        results['score'] = score

        return results

    def _analyze_dividends(self, metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze dividend metrics."""
        dividend_yield = metrics.get('dividend_yield')
        payout_ratio = metrics.get('payout_ratio')

        results = {
            'dividend_yield': {
                'value': dividend_yield,
                'rating': self._rate_dividend_yield(dividend_yield),
                'explanation': self._explain_dividend_yield(dividend_yield)
            },
            'payout_ratio': {
                'value': payout_ratio,
                'rating': self._rate_payout_ratio(payout_ratio),
                'explanation': self._explain_payout_ratio(payout_ratio)
            }
        }

        # Calculate dividend score (less weight for non-dividend payers)
        score = self._calculate_dividend_score(dividend_yield, payout_ratio)
        results['score'] = score

        return results

    # Rating methods
    def _rate_pe(self, pe: Optional[float]) -> str:
        if pe is None:
            return 'N/A'
        if pe < 0:
            return 'Poor'
        if pe < 15:
            return 'Excellent'
        if pe < 25:
            return 'Good'
        if pe < 40:
            return 'Fair'
        return 'Poor'

    def _rate_pb(self, pb: Optional[float]) -> str:
        if pb is None:
            return 'N/A'
        if pb < 1:
            return 'Excellent'
        if pb < 3:
            return 'Good'
        if pb < 5:
            return 'Fair'
        return 'Poor'

    def _rate_peg(self, peg: Optional[float]) -> str:
        if peg is None:
            return 'N/A'
        if peg < 1:
            return 'Excellent'
        if peg < 1.5:
            return 'Good'
        if peg < 2:
            return 'Fair'
        return 'Poor'

    def _rate_roe(self, roe: Optional[float]) -> str:
        if roe is None:
            return 'N/A'
        roe_pct = roe * 100 if roe < 1 else roe
        if roe_pct > 20:
            return 'Excellent'
        if roe_pct > 15:
            return 'Good'
        if roe_pct > 10:
            return 'Fair'
        return 'Poor'

    def _rate_roa(self, roa: Optional[float]) -> str:
        if roa is None:
            return 'N/A'
        roa_pct = roa * 100 if roa < 1 else roa
        if roa_pct > 10:
            return 'Excellent'
        if roa_pct > 5:
            return 'Good'
        if roa_pct > 2:
            return 'Fair'
        return 'Poor'

    def _rate_margin(self, margin: Optional[float], thresholds: tuple = (0.1, 0.2)) -> str:
        if margin is None:
            return 'N/A'
        if margin > thresholds[1]:
            return 'Excellent'
        if margin > thresholds[0]:
            return 'Good'
        if margin > 0:
            return 'Fair'
        return 'Poor'

    def _rate_debt_to_equity(self, de: Optional[float]) -> str:
        if de is None:
            return 'N/A'
        if de < 0.5:
            return 'Excellent'
        if de < 1:
            return 'Good'
        if de < 2:
            return 'Fair'
        return 'Poor'

    def _rate_current_ratio(self, cr: Optional[float]) -> str:
        if cr is None:
            return 'N/A'
        if cr > 2:
            return 'Excellent'
        if cr > 1.5:
            return 'Good'
        if cr > 1:
            return 'Fair'
        return 'Poor'

    def _rate_quick_ratio(self, qr: Optional[float]) -> str:
        if qr is None:
            return 'N/A'
        if qr > 1.5:
            return 'Excellent'
        if qr > 1:
            return 'Good'
        if qr > 0.5:
            return 'Fair'
        return 'Poor'

    def _rate_growth(self, growth: Optional[float]) -> str:
        if growth is None:
            return 'N/A'
        growth_pct = growth * 100 if -1 < growth < 1 else growth
        if growth_pct > 20:
            return 'Excellent'
        if growth_pct > 10:
            return 'Good'
        if growth_pct > 0:
            return 'Fair'
        return 'Poor'

    def _rate_dividend_yield(self, dy: Optional[float]) -> str:
        if dy is None or dy == 0:
            return 'N/A'
        dy_pct = dy * 100 if dy < 1 else dy
        if dy_pct > 4:
            return 'Excellent'
        if dy_pct > 2:
            return 'Good'
        if dy_pct > 1:
            return 'Fair'
        return 'Low'

    def _rate_payout_ratio(self, pr: Optional[float]) -> str:
        if pr is None:
            return 'N/A'
        pr_pct = pr * 100 if pr < 1 else pr
        if 30 < pr_pct < 60:
            return 'Excellent'
        if 20 < pr_pct < 70:
            return 'Good'
        if pr_pct < 80:
            return 'Fair'
        return 'Poor'

    def _rate_metric(self, value: Optional[float], lower_is_better: bool = False,
                     thresholds: tuple = (1, 2)) -> str:
        if value is None:
            return 'N/A'
        if lower_is_better:
            if value < thresholds[0]:
                return 'Excellent'
            if value < thresholds[1]:
                return 'Good'
            return 'Fair'
        else:
            if value > thresholds[1]:
                return 'Excellent'
            if value > thresholds[0]:
                return 'Good'
            return 'Fair'

    # Explanation methods
    def _explain_pe(self, pe: Optional[float]) -> str:
        if pe is None:
            return "P/E ratio not available"
        if pe < 0:
            return f"Negative P/E ({pe:.2f}) indicates losses"
        if pe < 15:
            return f"Low P/E of {pe:.2f} suggests undervaluation"
        if pe < 25:
            return f"P/E of {pe:.2f} is within normal range"
        return f"High P/E of {pe:.2f} may indicate overvaluation"

    def _explain_pb(self, pb: Optional[float]) -> str:
        if pb is None:
            return "P/B ratio not available"
        if pb < 1:
            return f"P/B of {pb:.2f} - stock trading below book value"
        if pb < 3:
            return f"P/B of {pb:.2f} is reasonable"
        return f"P/B of {pb:.2f} is high relative to book value"

    def _explain_peg(self, peg: Optional[float]) -> str:
        if peg is None:
            return "PEG ratio not available"
        if peg < 1:
            return f"PEG of {peg:.2f} suggests growth at reasonable price"
        if peg < 2:
            return f"PEG of {peg:.2f} indicates fair valuation for growth"
        return f"PEG of {peg:.2f} - may be expensive for its growth rate"

    def _explain_roe(self, roe: Optional[float]) -> str:
        if roe is None:
            return "ROE not available"
        roe_pct = roe * 100 if roe < 1 else roe
        if roe_pct > 20:
            return f"Excellent ROE of {roe_pct:.1f}% - efficient use of equity"
        if roe_pct > 15:
            return f"Good ROE of {roe_pct:.1f}%"
        return f"ROE of {roe_pct:.1f}% is below average"

    def _explain_roa(self, roa: Optional[float]) -> str:
        if roa is None:
            return "ROA not available"
        roa_pct = roa * 100 if roa < 1 else roa
        if roa_pct > 10:
            return f"Excellent ROA of {roa_pct:.1f}% - efficient asset utilization"
        if roa_pct > 5:
            return f"Good ROA of {roa_pct:.1f}%"
        return f"ROA of {roa_pct:.1f}% indicates room for improvement"

    def _explain_debt_to_equity(self, de: Optional[float]) -> str:
        if de is None:
            return "Debt-to-Equity not available"
        if de < 0.5:
            return f"Low D/E of {de:.2f} - conservative capital structure"
        if de < 1:
            return f"D/E of {de:.2f} is balanced"
        return f"High D/E of {de:.2f} - significant debt levels"

    def _explain_current_ratio(self, cr: Optional[float]) -> str:
        if cr is None:
            return "Current ratio not available"
        if cr > 2:
            return f"Strong liquidity with current ratio of {cr:.2f}"
        if cr > 1:
            return f"Adequate liquidity with current ratio of {cr:.2f}"
        return f"Low current ratio of {cr:.2f} - potential liquidity concerns"

    def _explain_growth(self, growth: Optional[float], metric: str) -> str:
        if growth is None:
            return f"{metric} growth not available"
        growth_pct = growth * 100 if -1 < growth < 1 else growth
        if growth_pct > 20:
            return f"Strong {metric.lower()} growth of {growth_pct:.1f}%"
        if growth_pct > 0:
            return f"Positive {metric.lower()} growth of {growth_pct:.1f}%"
        return f"Negative {metric.lower()} growth of {growth_pct:.1f}%"

    def _explain_dividend_yield(self, dy: Optional[float]) -> str:
        if dy is None or dy == 0:
            return "No dividend"
        dy_pct = dy * 100 if dy < 1 else dy
        if dy_pct > 4:
            return f"High dividend yield of {dy_pct:.2f}%"
        if dy_pct > 2:
            return f"Decent dividend yield of {dy_pct:.2f}%"
        return f"Modest dividend yield of {dy_pct:.2f}%"

    def _explain_payout_ratio(self, pr: Optional[float]) -> str:
        if pr is None:
            return "Payout ratio not available"
        pr_pct = pr * 100 if pr < 1 else pr
        if pr_pct < 30:
            return f"Low payout ratio of {pr_pct:.1f}% - room for dividend growth"
        if pr_pct < 60:
            return f"Sustainable payout ratio of {pr_pct:.1f}%"
        return f"High payout ratio of {pr_pct:.1f}% - may not be sustainable"

    # Score calculation methods
    def _calculate_valuation_score(self, pe, pb, peg) -> float:
        scores = []

        if pe is not None and pe > 0:
            if pe < 15:
                scores.append(90)
            elif pe < 25:
                scores.append(70)
            elif pe < 40:
                scores.append(50)
            else:
                scores.append(30)

        if pb is not None:
            if pb < 1:
                scores.append(90)
            elif pb < 3:
                scores.append(70)
            elif pb < 5:
                scores.append(50)
            else:
                scores.append(30)

        if peg is not None and peg > 0:
            if peg < 1:
                scores.append(90)
            elif peg < 1.5:
                scores.append(70)
            elif peg < 2:
                scores.append(50)
            else:
                scores.append(30)

        return np.mean(scores) if scores else 50

    def _calculate_profitability_score(self, roe, roa, margin) -> float:
        scores = []

        if roe is not None:
            roe_pct = roe * 100 if roe < 1 else roe
            if roe_pct > 20:
                scores.append(90)
            elif roe_pct > 15:
                scores.append(70)
            elif roe_pct > 10:
                scores.append(50)
            else:
                scores.append(30)

        if roa is not None:
            roa_pct = roa * 100 if roa < 1 else roa
            if roa_pct > 10:
                scores.append(90)
            elif roa_pct > 5:
                scores.append(70)
            elif roa_pct > 2:
                scores.append(50)
            else:
                scores.append(30)

        if margin is not None:
            if margin > 0.2:
                scores.append(90)
            elif margin > 0.1:
                scores.append(70)
            elif margin > 0:
                scores.append(50)
            else:
                scores.append(20)

        return np.mean(scores) if scores else 50

    def _calculate_health_score(self, de, current, quick) -> float:
        scores = []

        if de is not None:
            if de < 0.5:
                scores.append(90)
            elif de < 1:
                scores.append(70)
            elif de < 2:
                scores.append(50)
            else:
                scores.append(30)

        if current is not None:
            if current > 2:
                scores.append(90)
            elif current > 1.5:
                scores.append(70)
            elif current > 1:
                scores.append(50)
            else:
                scores.append(30)

        if quick is not None:
            if quick > 1.5:
                scores.append(90)
            elif quick > 1:
                scores.append(70)
            elif quick > 0.5:
                scores.append(50)
            else:
                scores.append(30)

        return np.mean(scores) if scores else 50

    def _calculate_growth_score(self, revenue_growth, earnings_growth) -> float:
        scores = []

        for growth in [revenue_growth, earnings_growth]:
            if growth is not None:
                g = growth * 100 if -1 < growth < 1 else growth
                if g > 20:
                    scores.append(90)
                elif g > 10:
                    scores.append(70)
                elif g > 0:
                    scores.append(50)
                else:
                    scores.append(30)

        return np.mean(scores) if scores else 50

    def _calculate_dividend_score(self, dividend_yield, payout_ratio) -> float:
        if dividend_yield is None or dividend_yield == 0:
            return 50  # Neutral for non-dividend payers

        scores = []

        dy = dividend_yield * 100 if dividend_yield < 1 else dividend_yield
        if dy > 4:
            scores.append(90)
        elif dy > 2:
            scores.append(70)
        else:
            scores.append(50)

        if payout_ratio is not None:
            pr = payout_ratio * 100 if payout_ratio < 1 else payout_ratio
            if 30 < pr < 60:
                scores.append(90)
            elif 20 < pr < 70:
                scores.append(70)
            else:
                scores.append(50)

        return np.mean(scores) if scores else 50

    def _generate_summary(self, analysis: Dict[str, Any]) -> str:
        """Generate a text summary of the analysis."""
        parts = []

        # Valuation
        val_score = analysis['valuation'].get('score', 50)
        if val_score > 70:
            parts.append("The stock appears undervalued based on P/E and P/B ratios.")
        elif val_score < 40:
            parts.append("Valuation metrics suggest the stock may be overpriced.")

        # Profitability
        prof_score = analysis['profitability'].get('score', 50)
        if prof_score > 70:
            parts.append("Strong profitability with good ROE and margins.")
        elif prof_score < 40:
            parts.append("Profitability metrics are below average.")

        # Financial Health
        health_score = analysis['financial_health'].get('score', 50)
        if health_score > 70:
            parts.append("Solid financial health with manageable debt levels.")
        elif health_score < 40:
            parts.append("Financial health metrics raise some concerns.")

        # Growth
        growth_score = analysis['growth'].get('score', 50)
        if growth_score > 70:
            parts.append("Strong growth trajectory in revenue and earnings.")
        elif growth_score < 40:
            parts.append("Growth has been challenging.")

        return " ".join(parts) if parts else "Fundamentals are within normal ranges."

    def compare_to_peers(
        self,
        stock_metrics: Dict[str, Any],
        peer_metrics: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Compare stock metrics to peer average.

        Args:
            stock_metrics: Metrics for the target stock
            peer_metrics: List of metrics for peer stocks

        Returns:
            Dictionary with comparison results
        """
        comparison = {}

        metrics_to_compare = ['pe_ratio', 'pb_ratio', 'roe', 'profit_margin',
                             'debt_to_equity', 'revenue_growth']

        for metric in metrics_to_compare:
            stock_val = stock_metrics.get(metric)
            peer_vals = [p.get(metric) for p in peer_metrics if p.get(metric) is not None]

            if stock_val is not None and peer_vals:
                peer_avg = np.mean(peer_vals)
                diff_pct = ((stock_val - peer_avg) / peer_avg * 100) if peer_avg != 0 else 0

                comparison[metric] = {
                    'stock_value': stock_val,
                    'peer_average': peer_avg,
                    'difference_pct': diff_pct,
                    'better_than_peers': self._is_better(metric, stock_val, peer_avg)
                }

        return comparison

    def _is_better(self, metric: str, stock_val: float, peer_val: float) -> bool:
        """Determine if stock value is better than peer value."""
        # Lower is better for these metrics
        lower_is_better = ['pe_ratio', 'pb_ratio', 'debt_to_equity']

        if metric in lower_is_better:
            return stock_val < peer_val
        return stock_val > peer_val
