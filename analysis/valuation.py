"""
Valuation models for stock analysis.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, List
from dataclasses import dataclass

from config import DCF_PARAMS


@dataclass
class DCFInputs:
    """Input parameters for DCF model."""
    free_cash_flow: float
    growth_rate_5y: float
    terminal_growth_rate: float
    discount_rate: float
    shares_outstanding: float
    current_price: float
    projection_years: int = 5


class ValuationModel:
    """
    Stock valuation models including DCF and comparable analysis.
    """

    def __init__(self, params: Optional[Dict] = None):
        """
        Initialize valuation model.

        Args:
            params: Custom parameters for valuation
        """
        self.params = params or DCF_PARAMS

    def calculate_dcf(
        self,
        free_cash_flow: float,
        growth_rate: float,
        discount_rate: float,
        terminal_growth: float = None,
        shares_outstanding: float = 1,
        projection_years: int = None
    ) -> Dict[str, Any]:
        """
        Calculate intrinsic value using DCF model.

        Args:
            free_cash_flow: Latest free cash flow
            growth_rate: Expected growth rate for projection period
            discount_rate: Discount rate (WACC)
            terminal_growth: Terminal growth rate
            shares_outstanding: Number of shares
            projection_years: Number of years to project

        Returns:
            Dictionary with DCF results
        """
        if terminal_growth is None:
            terminal_growth = self.params['terminal_growth_rate']
        if projection_years is None:
            projection_years = self.params['projection_years']

        # Project future cash flows
        projected_fcf = []
        fcf = free_cash_flow

        for year in range(1, projection_years + 1):
            fcf = fcf * (1 + growth_rate)
            projected_fcf.append({
                'year': year,
                'fcf': fcf,
                'discount_factor': 1 / (1 + discount_rate) ** year,
                'present_value': fcf / (1 + discount_rate) ** year
            })

        # Calculate terminal value
        terminal_fcf = projected_fcf[-1]['fcf'] * (1 + terminal_growth)
        terminal_value = terminal_fcf / (discount_rate - terminal_growth)
        terminal_pv = terminal_value / (1 + discount_rate) ** projection_years

        # Sum of present values
        sum_pv_fcf = sum(p['present_value'] for p in projected_fcf)
        enterprise_value = sum_pv_fcf + terminal_pv

        # Intrinsic value per share
        intrinsic_value = enterprise_value / shares_outstanding

        return {
            'projected_cash_flows': projected_fcf,
            'terminal_value': terminal_value,
            'terminal_pv': terminal_pv,
            'sum_pv_fcf': sum_pv_fcf,
            'enterprise_value': enterprise_value,
            'intrinsic_value_per_share': intrinsic_value,
            'inputs': {
                'free_cash_flow': free_cash_flow,
                'growth_rate': growth_rate,
                'discount_rate': discount_rate,
                'terminal_growth': terminal_growth,
                'shares_outstanding': shares_outstanding,
                'projection_years': projection_years
            }
        }

    def calculate_wacc(
        self,
        risk_free_rate: float = None,
        beta: float = 1.0,
        market_premium: float = None,
        debt_ratio: float = 0.3,
        cost_of_debt: float = 0.08,
        tax_rate: float = 0.25
    ) -> float:
        """
        Calculate Weighted Average Cost of Capital.

        Args:
            risk_free_rate: Risk-free rate
            beta: Stock beta
            market_premium: Market risk premium
            debt_ratio: Debt to total capital ratio
            cost_of_debt: Pre-tax cost of debt
            tax_rate: Corporate tax rate

        Returns:
            WACC as decimal
        """
        if risk_free_rate is None:
            risk_free_rate = self.params['risk_free_rate']
        if market_premium is None:
            market_premium = self.params['market_risk_premium']

        # Cost of equity using CAPM
        cost_of_equity = risk_free_rate + beta * market_premium

        # WACC calculation
        equity_ratio = 1 - debt_ratio
        after_tax_cost_of_debt = cost_of_debt * (1 - tax_rate)

        wacc = (equity_ratio * cost_of_equity) + (debt_ratio * after_tax_cost_of_debt)

        return wacc

    def calculate_margin_of_safety(
        self,
        intrinsic_value: float,
        current_price: float
    ) -> Dict[str, Any]:
        """
        Calculate margin of safety.

        Args:
            intrinsic_value: Calculated intrinsic value
            current_price: Current market price

        Returns:
            Dictionary with margin of safety metrics
        """
        margin = (intrinsic_value - current_price) / intrinsic_value * 100

        if margin > 30:
            recommendation = 'Strong Buy - Significant undervaluation'
        elif margin > 15:
            recommendation = 'Buy - Reasonable margin of safety'
        elif margin > 0:
            recommendation = 'Hold - Slight undervaluation'
        elif margin > -15:
            recommendation = 'Hold - Fairly valued'
        else:
            recommendation = 'Sell - Overvalued'

        return {
            'intrinsic_value': intrinsic_value,
            'current_price': current_price,
            'margin_of_safety': margin,
            'upside_potential': (intrinsic_value / current_price - 1) * 100,
            'recommendation': recommendation
        }

    def sensitivity_analysis(
        self,
        base_dcf: Dict[str, Any],
        growth_range: List[float] = None,
        discount_range: List[float] = None
    ) -> pd.DataFrame:
        """
        Perform sensitivity analysis on DCF model.

        Args:
            base_dcf: Base DCF calculation results
            growth_range: Range of growth rates to test
            discount_range: Range of discount rates to test

        Returns:
            DataFrame with sensitivity matrix
        """
        if growth_range is None:
            base_growth = base_dcf['inputs']['growth_rate']
            growth_range = [base_growth - 0.05, base_growth - 0.025, base_growth,
                          base_growth + 0.025, base_growth + 0.05]

        if discount_range is None:
            base_discount = base_dcf['inputs']['discount_rate']
            discount_range = [base_discount - 0.02, base_discount - 0.01, base_discount,
                            base_discount + 0.01, base_discount + 0.02]

        inputs = base_dcf['inputs']
        results = []

        for growth in growth_range:
            row = {}
            for discount in discount_range:
                dcf = self.calculate_dcf(
                    free_cash_flow=inputs['free_cash_flow'],
                    growth_rate=growth,
                    discount_rate=discount,
                    terminal_growth=inputs['terminal_growth'],
                    shares_outstanding=inputs['shares_outstanding'],
                    projection_years=inputs['projection_years']
                )
                row[f'{discount*100:.1f}%'] = dcf['intrinsic_value_per_share']
            row['Growth Rate'] = f'{growth*100:.1f}%'
            results.append(row)

        df = pd.DataFrame(results)
        df = df.set_index('Growth Rate')

        return df

    def comparable_valuation(
        self,
        target_metrics: Dict[str, Any],
        peer_metrics: List[Dict[str, Any]],
        multiples: List[str] = None
    ) -> Dict[str, Any]:
        """
        Perform comparable company valuation.

        Args:
            target_metrics: Metrics for target company
            peer_metrics: Metrics for peer companies
            multiples: List of multiples to use

        Returns:
            Dictionary with comparable valuation results
        """
        if multiples is None:
            multiples = ['pe_ratio', 'pb_ratio', 'ps_ratio']

        results = {}

        for multiple in multiples:
            peer_values = [p.get(multiple) for p in peer_metrics
                         if p.get(multiple) is not None and p.get(multiple) > 0]

            if not peer_values:
                continue

            peer_avg = np.mean(peer_values)
            peer_median = np.median(peer_values)
            peer_min = np.min(peer_values)
            peer_max = np.max(peer_values)

            target_value = target_metrics.get(multiple)

            # Calculate implied price based on peer average
            current_price = target_metrics.get('current_price', 0)

            if multiple == 'pe_ratio' and target_metrics.get('eps'):
                implied_price_avg = peer_avg * target_metrics['eps']
                implied_price_median = peer_median * target_metrics['eps']
            elif multiple == 'pb_ratio' and target_metrics.get('book_value'):
                implied_price_avg = peer_avg * target_metrics['book_value']
                implied_price_median = peer_median * target_metrics['book_value']
            elif multiple == 'ps_ratio' and target_metrics.get('revenue_per_share'):
                implied_price_avg = peer_avg * target_metrics['revenue_per_share']
                implied_price_median = peer_median * target_metrics['revenue_per_share']
            else:
                implied_price_avg = None
                implied_price_median = None

            results[multiple] = {
                'target_value': target_value,
                'peer_average': peer_avg,
                'peer_median': peer_median,
                'peer_min': peer_min,
                'peer_max': peer_max,
                'implied_price_avg': implied_price_avg,
                'implied_price_median': implied_price_median,
                'premium_to_peers': ((target_value / peer_avg) - 1) * 100 if target_value and peer_avg else None
            }

        # Calculate weighted average implied price
        implied_prices = []
        for mult, data in results.items():
            if data.get('implied_price_avg'):
                implied_prices.append(data['implied_price_avg'])

        if implied_prices:
            results['weighted_implied_price'] = np.mean(implied_prices)
            current = target_metrics.get('current_price', 0)
            if current > 0:
                results['upside_potential'] = (results['weighted_implied_price'] / current - 1) * 100
        else:
            results['weighted_implied_price'] = None
            results['upside_potential'] = None

        return results

    def calculate_graham_number(
        self,
        eps: float,
        book_value: float
    ) -> float:
        """
        Calculate Graham Number (fair value estimate).

        Args:
            eps: Earnings per share
            book_value: Book value per share

        Returns:
            Graham Number (fair value)
        """
        if eps <= 0 or book_value <= 0:
            return 0

        return np.sqrt(22.5 * eps * book_value)

    def calculate_peg_fair_value(
        self,
        eps: float,
        growth_rate: float,
        target_peg: float = 1.0
    ) -> float:
        """
        Calculate fair value based on PEG ratio.

        Args:
            eps: Earnings per share
            growth_rate: Expected growth rate (as percentage, e.g., 15 for 15%)
            target_peg: Target PEG ratio (1.0 = fairly valued)

        Returns:
            Fair value per share
        """
        if eps <= 0 or growth_rate <= 0:
            return 0

        fair_pe = target_peg * growth_rate
        return eps * fair_pe

    def get_valuation_summary(
        self,
        metrics: Dict[str, Any],
        dcf_result: Optional[Dict] = None,
        comparable_result: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """
        Get comprehensive valuation summary.

        Args:
            metrics: Stock metrics
            dcf_result: DCF calculation result
            comparable_result: Comparable analysis result

        Returns:
            Dictionary with valuation summary
        """
        current_price = metrics.get('current_price', 0)
        valuations = []

        # Add DCF valuation
        if dcf_result:
            valuations.append({
                'method': 'DCF',
                'value': dcf_result['intrinsic_value_per_share'],
                'weight': 0.4
            })

        # Add comparable valuation
        if comparable_result and comparable_result.get('weighted_implied_price'):
            valuations.append({
                'method': 'Comparable',
                'value': comparable_result['weighted_implied_price'],
                'weight': 0.3
            })

        # Add Graham Number
        if metrics.get('eps') and metrics.get('book_value'):
            graham = self.calculate_graham_number(metrics['eps'], metrics['book_value'])
            if graham > 0:
                valuations.append({
                    'method': 'Graham Number',
                    'value': graham,
                    'weight': 0.15
                })

        # Add PEG-based valuation
        if metrics.get('eps') and metrics.get('earnings_growth'):
            growth_pct = metrics['earnings_growth'] * 100 if metrics['earnings_growth'] < 1 else metrics['earnings_growth']
            if growth_pct > 0:
                peg_value = self.calculate_peg_fair_value(metrics['eps'], growth_pct)
                if peg_value > 0:
                    valuations.append({
                        'method': 'PEG Fair Value',
                        'value': peg_value,
                        'weight': 0.15
                    })

        # Calculate weighted average fair value
        if valuations:
            total_weight = sum(v['weight'] for v in valuations)
            weighted_value = sum(v['value'] * v['weight'] for v in valuations) / total_weight

            margin_of_safety = self.calculate_margin_of_safety(weighted_value, current_price)

            return {
                'current_price': current_price,
                'valuations': valuations,
                'weighted_fair_value': weighted_value,
                'margin_of_safety': margin_of_safety['margin_of_safety'],
                'upside_potential': margin_of_safety['upside_potential'],
                'recommendation': margin_of_safety['recommendation']
            }

        return {
            'current_price': current_price,
            'valuations': [],
            'weighted_fair_value': None,
            'margin_of_safety': None,
            'upside_potential': None,
            'recommendation': 'Insufficient data for valuation'
        }

    def auto_dcf(self, metrics: Dict[str, Any], financials: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Automatically calculate DCF using available data.

        Args:
            metrics: Stock metrics
            financials: Financial statements

        Returns:
            DCF result or None if insufficient data
        """
        # Try to get free cash flow from cash flow statement
        fcf = None
        if financials.get('cash_flow') is not None:
            cf = financials['cash_flow']
            if 'Free Cash Flow' in cf.index:
                fcf = cf.loc['Free Cash Flow'].iloc[0]
            elif 'Operating Cash Flow' in cf.index and 'Capital Expenditure' in cf.index:
                ocf = cf.loc['Operating Cash Flow'].iloc[0]
                capex = abs(cf.loc['Capital Expenditure'].iloc[0])
                fcf = ocf - capex

        if fcf is None or fcf <= 0:
            return None

        # Get growth rate
        growth_rate = metrics.get('earnings_growth') or metrics.get('revenue_growth') or 0.10
        if growth_rate > 1:
            growth_rate = growth_rate / 100

        # Cap growth rate at reasonable levels
        growth_rate = min(growth_rate, 0.25)
        growth_rate = max(growth_rate, 0.03)

        # Calculate WACC
        beta = metrics.get('beta', 1.0)
        de_ratio = metrics.get('debt_to_equity', 50)
        if de_ratio > 10:
            de_ratio = de_ratio / 100
        debt_ratio = de_ratio / (1 + de_ratio) if de_ratio > 0 else 0.3

        wacc = self.calculate_wacc(beta=beta, debt_ratio=debt_ratio)

        # Get shares outstanding
        shares = metrics.get('shares_outstanding', 1)
        if shares is None or shares <= 0:
            market_cap = metrics.get('market_cap', 0)
            price = metrics.get('current_price', 1)
            shares = market_cap / price if price > 0 else 1

        return self.calculate_dcf(
            free_cash_flow=fcf,
            growth_rate=growth_rate,
            discount_rate=wacc,
            shares_outstanding=shares
        )
