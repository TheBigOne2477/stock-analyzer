"""
Report generation module.
"""

from datetime import datetime
from typing import Dict, Any, Optional
import pandas as pd

from config import DISCLAIMER


class ReportGenerator:
    """
    Generates comprehensive HTML reports for stock analysis.
    """

    def __init__(self):
        """Initialize report generator."""
        self.css = """
        <style>
            body {
                font-family: 'Segoe UI', Arial, sans-serif;
                margin: 40px;
                color: #333;
                line-height: 1.6;
            }
            .header {
                text-align: center;
                border-bottom: 2px solid #1f77b4;
                padding-bottom: 20px;
                margin-bottom: 30px;
            }
            .header h1 {
                color: #1f77b4;
                margin-bottom: 5px;
            }
            .header p {
                color: #666;
                margin: 0;
            }
            .section {
                margin-bottom: 30px;
                page-break-inside: avoid;
            }
            .section h2 {
                color: #1f77b4;
                border-bottom: 1px solid #ddd;
                padding-bottom: 10px;
            }
            .metric-grid {
                display: grid;
                grid-template-columns: repeat(4, 1fr);
                gap: 15px;
                margin: 20px 0;
            }
            .metric-card {
                background: #f8f9fa;
                border-radius: 8px;
                padding: 15px;
                text-align: center;
            }
            .metric-card .label {
                font-size: 0.9em;
                color: #666;
            }
            .metric-card .value {
                font-size: 1.4em;
                font-weight: bold;
                color: #333;
            }
            .metric-card .change {
                font-size: 0.9em;
            }
            .positive { color: #00c853; }
            .negative { color: #f44336; }
            .neutral { color: #ff9800; }
            .recommendation-box {
                background: linear-gradient(135deg, #1f77b4, #2ca02c);
                color: white;
                padding: 30px;
                border-radius: 10px;
                text-align: center;
                margin: 20px 0;
            }
            .recommendation-box h3 {
                margin: 0;
                font-size: 2em;
            }
            .recommendation-box p {
                margin: 10px 0 0 0;
                font-size: 1.2em;
            }
            table {
                width: 100%;
                border-collapse: collapse;
                margin: 20px 0;
            }
            th, td {
                padding: 12px;
                text-align: left;
                border-bottom: 1px solid #ddd;
            }
            th {
                background: #f8f9fa;
                font-weight: bold;
            }
            tr:hover {
                background: #f8f9fa;
            }
            .score-bar {
                background: #e0e0e0;
                border-radius: 10px;
                height: 20px;
                overflow: hidden;
            }
            .score-fill {
                height: 100%;
                border-radius: 10px;
            }
            .disclaimer {
                background: #fff3cd;
                border: 1px solid #ffc107;
                padding: 15px;
                border-radius: 5px;
                margin-top: 30px;
                font-size: 0.9em;
            }
            .footer {
                text-align: center;
                color: #666;
                font-size: 0.9em;
                margin-top: 40px;
                padding-top: 20px;
                border-top: 1px solid #ddd;
            }
            @media print {
                body { margin: 20px; }
                .section { page-break-inside: avoid; }
            }
        </style>
        """

    def generate_report(
        self,
        symbol: str,
        metrics: Dict[str, Any],
        fundamental_analysis: Dict[str, Any],
        technical_analysis: Dict[str, Any],
        valuation_analysis: Optional[Dict[str, Any]] = None,
        prediction_result: Optional[Dict[str, Any]] = None,
        score_breakdown: Optional[Dict[str, Any]] = None,
        news_articles: Optional[list] = None
    ) -> str:
        """
        Generate comprehensive HTML report.

        Args:
            symbol: Stock symbol
            metrics: Key metrics
            fundamental_analysis: Fundamental analysis results
            technical_analysis: Technical analysis results
            valuation_analysis: Valuation analysis results
            prediction_result: ML prediction results
            score_breakdown: Score breakdown
            news_articles: Recent news articles

        Returns:
            HTML string
        """
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="UTF-8">
            <title>Stock Analysis Report - {symbol}</title>
            {self.css}
        </head>
        <body>
            {self._generate_header(symbol, metrics)}
            {self._generate_recommendation_section(score_breakdown)}
            {self._generate_key_metrics_section(metrics)}
            {self._generate_fundamental_section(fundamental_analysis)}
            {self._generate_technical_section(technical_analysis)}
            {self._generate_valuation_section(valuation_analysis, metrics)}
            {self._generate_prediction_section(prediction_result)}
            {self._generate_score_section(score_breakdown)}
            {self._generate_news_section(news_articles)}
            {self._generate_disclaimer()}
            {self._generate_footer()}
        </body>
        </html>
        """

        return html

    def _generate_header(self, symbol: str, metrics: Dict[str, Any]) -> str:
        """Generate report header."""
        company_name = metrics.get('company_name', symbol)
        sector = metrics.get('sector', 'N/A')
        industry = metrics.get('industry', 'N/A')
        date = datetime.now().strftime('%B %d, %Y')

        return f"""
        <div class="header">
            <h1>{company_name} ({symbol})</h1>
            <p>Sector: {sector} | Industry: {industry}</p>
            <p>Report Generated: {date}</p>
        </div>
        """

    def _generate_recommendation_section(self, score_breakdown: Optional[Dict[str, Any]]) -> str:
        """Generate recommendation section."""
        if not score_breakdown:
            return ""

        recommendation = score_breakdown.get('recommendation', 'N/A')
        score = score_breakdown.get('total_score', 0)
        confidence = score_breakdown.get('confidence', 'N/A')

        # Determine color
        if 'Buy' in recommendation:
            gradient = 'linear-gradient(135deg, #00c853, #1f77b4)'
        elif 'Sell' in recommendation:
            gradient = 'linear-gradient(135deg, #f44336, #d32f2f)'
        else:
            gradient = 'linear-gradient(135deg, #ff9800, #f57c00)'

        return f"""
        <div class="section">
            <div class="recommendation-box" style="background: {gradient};">
                <h3>{recommendation}</h3>
                <p>Score: {score:.1f}/100 | Confidence: {confidence}</p>
            </div>
        </div>
        """

    def _generate_key_metrics_section(self, metrics: Dict[str, Any]) -> str:
        """Generate key metrics section."""
        price = metrics.get('current_price', 0)
        prev_close = metrics.get('previous_close', price)
        change_pct = ((price - prev_close) / prev_close * 100) if prev_close else 0
        change_class = 'positive' if change_pct >= 0 else 'negative'

        market_cap = metrics.get('market_cap', 0)
        if market_cap >= 1e7:
            market_cap_str = f"₹{market_cap/1e7:.2f} Cr"
        elif market_cap >= 1e5:
            market_cap_str = f"₹{market_cap/1e5:.2f} L"
        else:
            market_cap_str = f"₹{market_cap:,.0f}"

        pe = metrics.get('pe_ratio')
        pe_str = f"{pe:.2f}" if pe else "N/A"

        div_yield = metrics.get('dividend_yield')
        div_str = f"{div_yield*100:.2f}%" if div_yield else "N/A"

        return f"""
        <div class="section">
            <h2>Key Metrics</h2>
            <div class="metric-grid">
                <div class="metric-card">
                    <div class="label">Current Price</div>
                    <div class="value">₹{price:,.2f}</div>
                    <div class="change {change_class}">{change_pct:+.2f}%</div>
                </div>
                <div class="metric-card">
                    <div class="label">Market Cap</div>
                    <div class="value">{market_cap_str}</div>
                </div>
                <div class="metric-card">
                    <div class="label">P/E Ratio</div>
                    <div class="value">{pe_str}</div>
                </div>
                <div class="metric-card">
                    <div class="label">Dividend Yield</div>
                    <div class="value">{div_str}</div>
                </div>
                <div class="metric-card">
                    <div class="label">52W High</div>
                    <div class="value">₹{metrics.get('fifty_two_week_high', 0):,.2f}</div>
                </div>
                <div class="metric-card">
                    <div class="label">52W Low</div>
                    <div class="value">₹{metrics.get('fifty_two_week_low', 0):,.2f}</div>
                </div>
                <div class="metric-card">
                    <div class="label">Volume</div>
                    <div class="value">{metrics.get('volume', 0):,.0f}</div>
                </div>
                <div class="metric-card">
                    <div class="label">Beta</div>
                    <div class="value">{metrics.get('beta', 1):.2f}</div>
                </div>
            </div>
        </div>
        """

    def _generate_fundamental_section(self, analysis: Dict[str, Any]) -> str:
        """Generate fundamental analysis section."""
        rows = []

        categories = {
            'valuation': 'Valuation',
            'profitability': 'Profitability',
            'financial_health': 'Financial Health',
            'growth': 'Growth'
        }

        for cat_key, cat_name in categories.items():
            cat_data = analysis.get(cat_key, {})
            for metric, data in cat_data.items():
                if isinstance(data, dict) and 'value' in data:
                    value = data.get('value')
                    rating = data.get('rating', 'N/A')
                    explanation = data.get('explanation', '')

                    # Format value
                    if value is not None:
                        if 'margin' in metric or 'growth' in metric or 'yield' in metric:
                            formatted = f"{value*100:.2f}%" if abs(value) < 1 else f"{value:.2f}%"
                        else:
                            formatted = f"{value:.2f}"
                    else:
                        formatted = "N/A"

                    # Rating class
                    rating_class = 'positive' if rating in ['Excellent', 'Good'] else 'negative' if rating == 'Poor' else 'neutral'

                    rows.append(f"""
                    <tr>
                        <td>{cat_name}</td>
                        <td>{metric.replace('_', ' ').title()}</td>
                        <td>{formatted}</td>
                        <td class="{rating_class}">{rating}</td>
                    </tr>
                    """)

        return f"""
        <div class="section">
            <h2>Fundamental Analysis</h2>
            <table>
                <thead>
                    <tr>
                        <th>Category</th>
                        <th>Metric</th>
                        <th>Value</th>
                        <th>Rating</th>
                    </tr>
                </thead>
                <tbody>
                    {''.join(rows)}
                </tbody>
            </table>
        </div>
        """

    def _generate_technical_section(self, analysis: Dict[str, Any]) -> str:
        """Generate technical analysis section."""
        trend = analysis.get('trend', {})
        rsi = analysis.get('rsi', {})
        macd = analysis.get('macd', {})
        bb = analysis.get('bollinger_bands', {})

        trend_class = 'positive' if trend.get('direction') == 'Bullish' else 'negative' if trend.get('direction') == 'Bearish' else 'neutral'

        return f"""
        <div class="section">
            <h2>Technical Analysis</h2>
            <div class="metric-grid">
                <div class="metric-card">
                    <div class="label">Trend</div>
                    <div class="value {trend_class}">{trend.get('direction', 'N/A')}</div>
                    <div class="change">{trend.get('strength', 'N/A')}</div>
                </div>
                <div class="metric-card">
                    <div class="label">RSI</div>
                    <div class="value">{rsi.get('value', 0):.1f}</div>
                    <div class="change">{rsi.get('signal', 'N/A')}</div>
                </div>
                <div class="metric-card">
                    <div class="label">MACD</div>
                    <div class="value">{macd.get('signal', 'N/A')}</div>
                    <div class="change">{macd.get('crossover', '') or 'No crossover'}</div>
                </div>
                <div class="metric-card">
                    <div class="label">Bollinger</div>
                    <div class="value">{bb.get('signal', 'N/A')}</div>
                    <div class="change">Vol: {bb.get('volatility', 'N/A')}</div>
                </div>
            </div>
            <p><strong>Summary:</strong> {analysis.get('summary', 'N/A')}</p>
        </div>
        """

    def _generate_valuation_section(
        self,
        valuation: Optional[Dict[str, Any]],
        metrics: Dict[str, Any]
    ) -> str:
        """Generate valuation section."""
        if not valuation:
            return ""

        current_price = metrics.get('current_price', 0)
        fair_value = valuation.get('weighted_fair_value')
        margin = valuation.get('margin_of_safety')
        upside = valuation.get('upside_potential')

        if fair_value is None:
            return ""

        margin_class = 'positive' if margin and margin > 0 else 'negative'

        valuations_html = ""
        for v in valuation.get('valuations', []):
            valuations_html += f"""
            <tr>
                <td>{v['method']}</td>
                <td>₹{v['value']:,.2f}</td>
                <td>{v['weight']*100:.0f}%</td>
            </tr>
            """

        return f"""
        <div class="section">
            <h2>Valuation Analysis</h2>
            <div class="metric-grid">
                <div class="metric-card">
                    <div class="label">Current Price</div>
                    <div class="value">₹{current_price:,.2f}</div>
                </div>
                <div class="metric-card">
                    <div class="label">Fair Value</div>
                    <div class="value">₹{fair_value:,.2f}</div>
                </div>
                <div class="metric-card">
                    <div class="label">Margin of Safety</div>
                    <div class="value {margin_class}">{margin:+.1f}%</div>
                </div>
                <div class="metric-card">
                    <div class="label">Upside Potential</div>
                    <div class="value {margin_class}">{upside:+.1f}%</div>
                </div>
            </div>
            <table>
                <thead>
                    <tr>
                        <th>Valuation Method</th>
                        <th>Fair Value</th>
                        <th>Weight</th>
                    </tr>
                </thead>
                <tbody>
                    {valuations_html}
                </tbody>
            </table>
        </div>
        """

    def _generate_prediction_section(self, prediction: Optional[Dict[str, Any]]) -> str:
        """Generate prediction section."""
        if not prediction:
            return ""

        ensemble = prediction.get('ensemble', {})
        horizon = prediction.get('horizon', 7)
        current = prediction.get('current_price', 0)

        pred_change = ensemble.get('predicted_change_pct', 0)
        pred_price = ensemble.get('predicted_price', current)
        confidence = ensemble.get('confidence', 0) * 100
        agreement = ensemble.get('agreement', 0) * 100
        signal = ensemble.get('signal', 'N/A')

        change_class = 'positive' if pred_change > 0 else 'negative'

        return f"""
        <div class="section">
            <h2>Price Prediction ({horizon}-Day)</h2>
            <div class="metric-grid">
                <div class="metric-card">
                    <div class="label">Current Price</div>
                    <div class="value">₹{current:,.2f}</div>
                </div>
                <div class="metric-card">
                    <div class="label">Predicted Price</div>
                    <div class="value">₹{pred_price:,.2f}</div>
                    <div class="change {change_class}">{pred_change:+.2f}%</div>
                </div>
                <div class="metric-card">
                    <div class="label">Confidence</div>
                    <div class="value">{confidence:.0f}%</div>
                </div>
                <div class="metric-card">
                    <div class="label">Model Agreement</div>
                    <div class="value">{agreement:.0f}%</div>
                </div>
            </div>
            <p><strong>Signal:</strong> {signal}</p>
        </div>
        """

    def _generate_score_section(self, score_breakdown: Optional[Dict[str, Any]]) -> str:
        """Generate score breakdown section."""
        if not score_breakdown:
            return ""

        components = score_breakdown.get('components', {})
        rows = ""

        for name, data in components.items():
            score = data.get('score', 0)
            weight = data.get('weight', 0)
            contribution = data.get('contribution', 0)

            # Score bar color
            if score >= 70:
                color = '#00c853'
            elif score >= 40:
                color = '#ff9800'
            else:
                color = '#f44336'

            rows += f"""
            <tr>
                <td>{name}</td>
                <td>
                    <div class="score-bar">
                        <div class="score-fill" style="width: {score}%; background: {color};"></div>
                    </div>
                </td>
                <td>{score:.1f}</td>
                <td>{weight:.0f}%</td>
                <td>{contribution:.1f}</td>
            </tr>
            """

        key_factors = score_breakdown.get('key_factors', [])
        factors_html = "<ul>" + "".join([f"<li>{f}</li>" for f in key_factors]) + "</ul>"

        return f"""
        <div class="section">
            <h2>Score Breakdown</h2>
            <table>
                <thead>
                    <tr>
                        <th>Category</th>
                        <th>Score</th>
                        <th>Value</th>
                        <th>Weight</th>
                        <th>Contribution</th>
                    </tr>
                </thead>
                <tbody>
                    {rows}
                </tbody>
            </table>
            <h3>Key Factors</h3>
            {factors_html}
        </div>
        """

    def _generate_news_section(self, articles: Optional[list]) -> str:
        """Generate news section."""
        if not articles:
            return ""

        news_html = ""
        for article in articles[:5]:
            sentiment = article.get('sentiment', {}).get('sentiment', 'Neutral')
            sentiment_class = 'positive' if sentiment == 'Positive' else 'negative' if sentiment == 'Negative' else 'neutral'

            news_html += f"""
            <div style="margin-bottom: 15px; padding-bottom: 15px; border-bottom: 1px solid #eee;">
                <strong><a href="{article['link']}" target="_blank">{article['title']}</a></strong>
                <br>
                <span style="color: #666; font-size: 0.9em;">
                    {article['source']} - {article['date'][:10] if article['date'] else 'N/A'}
                    | <span class="{sentiment_class}">{sentiment}</span>
                </span>
            </div>
            """

        return f"""
        <div class="section">
            <h2>Recent News</h2>
            {news_html}
        </div>
        """

    def _generate_disclaimer(self) -> str:
        """Generate disclaimer section."""
        return f"""
        <div class="disclaimer">
            <strong>Disclaimer:</strong> {DISCLAIMER}
        </div>
        """

    def _generate_footer(self) -> str:
        """Generate report footer."""
        return f"""
        <div class="footer">
            <p>Generated by Indian Stock Market Analyzer</p>
            <p>Report Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        </div>
        """

    def save_report(self, html: str, filename: str):
        """
        Save report to file.

        Args:
            html: HTML content
            filename: Output filename
        """
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(html)
