"""
Reusable Streamlit UI components.
"""

import streamlit as st
import pandas as pd
from typing import Dict, Any, Optional, List

from config import DEFAULT_STOCKS, SECTORS, DISCLAIMER
from utils.helpers import (
    format_indian_number, format_percentage, format_ratio,
    get_recommendation_color
)


def render_header():
    """Render application header."""
    st.title("Indian Stock Market Analyzer")
    st.markdown("*Comprehensive analysis for NSE/BSE stocks*")


def render_disclaimer():
    """Render disclaimer section."""
    with st.expander("Disclaimer", expanded=False):
        st.warning(DISCLAIMER)


def render_stock_selector() -> str:
    """Render stock selection widget."""
    col1, col2 = st.columns([3, 1])

    with col1:
        symbol = st.text_input(
            "Enter Stock Symbol",
            placeholder="e.g., RELIANCE, TCS, INFY",
            help="Enter NSE stock symbol without .NS suffix"
        )

    with col2:
        st.write("")  # Spacer
        st.write("")
        if st.button("Analyze", type="primary"):
            st.session_state.analyze_clicked = True

    # Quick select from popular stocks
    st.markdown("**Quick Select:**")
    cols = st.columns(5)
    popular = ['RELIANCE', 'TCS', 'INFY', 'HDFCBANK', 'ICICIBANK']

    for i, stock in enumerate(popular):
        with cols[i]:
            if st.button(stock, key=f"quick_{stock}"):
                return stock

    return symbol.upper() if symbol else ""


def render_sector_selector() -> Optional[str]:
    """Render sector selection dropdown."""
    sectors = list(SECTORS.keys())
    selected = st.selectbox(
        "Filter by Sector",
        ["All Sectors"] + sectors,
        help="Select a sector to see related stocks"
    )

    if selected != "All Sectors":
        st.write("**Stocks in sector:**")
        stocks = SECTORS.get(selected, [])
        st.write(", ".join([s.replace('.NS', '') for s in stocks]))
        return selected

    return None


def render_metric_card(
    title: str,
    value: Any,
    subtitle: Optional[str] = None,
    delta: Optional[float] = None,
    help_text: Optional[str] = None
):
    """Render a metric card."""
    st.metric(
        label=title,
        value=value,
        delta=f"{delta:+.2f}%" if delta is not None else None,
        help=help_text
    )
    if subtitle:
        st.caption(subtitle)


def render_key_metrics(metrics: Dict[str, Any]):
    """Render key metrics overview."""
    st.subheader("Key Metrics")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        price = metrics.get('current_price', 0)
        prev_close = metrics.get('previous_close', price)
        change = ((price - prev_close) / prev_close * 100) if prev_close else 0
        st.metric("Current Price", f"₹{price:,.2f}", f"{change:+.2f}%")

    with col2:
        market_cap = metrics.get('market_cap', 0)
        st.metric("Market Cap", format_indian_number(market_cap))

    with col3:
        pe = metrics.get('pe_ratio')
        st.metric("P/E Ratio", format_ratio(pe) if pe else "N/A")

    with col4:
        div_yield = metrics.get('dividend_yield')
        st.metric("Dividend Yield", format_percentage(div_yield) if div_yield else "N/A")

    # Second row
    col5, col6, col7, col8 = st.columns(4)

    with col5:
        high_52 = metrics.get('fifty_two_week_high', 0)
        st.metric("52W High", f"₹{high_52:,.2f}" if high_52 else "N/A")

    with col6:
        low_52 = metrics.get('fifty_two_week_low', 0)
        st.metric("52W Low", f"₹{low_52:,.2f}" if low_52 else "N/A")

    with col7:
        volume = metrics.get('volume', 0)
        avg_volume = metrics.get('average_volume', 0)
        vol_ratio = volume / avg_volume if avg_volume else 1
        st.metric("Volume", f"{volume:,.0f}", f"{(vol_ratio-1)*100:+.1f}% vs avg")

    with col8:
        beta = metrics.get('beta')
        st.metric("Beta", format_ratio(beta) if beta else "N/A")


def render_company_info(info: Dict[str, Any]):
    """Render company information section."""
    st.subheader("Company Information")

    col1, col2 = st.columns([2, 1])

    with col1:
        st.write(f"**{info.get('company_name', 'N/A')}**")
        st.write(f"Sector: {info.get('sector', 'N/A')} | Industry: {info.get('industry', 'N/A')}")

        description = info.get('description', '')
        if description:
            with st.expander("About the Company"):
                st.write(description)

    with col2:
        if info.get('website'):
            st.write(f"[Company Website]({info['website']})")
        if info.get('employees'):
            st.write(f"Employees: {info['employees']:,}")


def render_recommendation_badge(recommendation: str, score: float):
    """Render recommendation badge with color."""
    color = get_recommendation_color(recommendation)

    st.markdown(f"""
    <div style="
        background-color: {color};
        color: white;
        padding: 20px;
        border-radius: 10px;
        text-align: center;
        margin: 10px 0;
    ">
        <h2 style="margin: 0; color: white;">{recommendation}</h2>
        <p style="margin: 5px 0 0 0; font-size: 1.2em;">Score: {score:.1f}/100</p>
    </div>
    """, unsafe_allow_html=True)


def render_score_gauge(score: float, title: str = "Overall Score"):
    """Render a score gauge visualization."""
    # Create a simple progress bar style gauge
    color = (
        "#00c853" if score >= 70
        else "#ff9800" if score >= 40
        else "#f44336"
    )

    st.markdown(f"**{title}**")
    st.progress(score / 100)
    st.markdown(f"<span style='color: {color}; font-size: 1.5em;'>{score:.1f}</span>/100",
                unsafe_allow_html=True)


def render_score_breakdown(breakdown: Dict[str, Any]):
    """Render score breakdown table."""
    st.subheader("Score Breakdown")

    components = breakdown.get('components', {})

    for name, data in components.items():
        col1, col2, col3 = st.columns([3, 1, 1])

        with col1:
            st.write(name)

        with col2:
            score = data.get('score', 0)
            color = "#00c853" if score >= 70 else "#ff9800" if score >= 40 else "#f44336"
            st.markdown(f"<span style='color: {color};'>{score:.1f}</span>",
                       unsafe_allow_html=True)

        with col3:
            weight = data.get('weight', 0)
            st.write(f"{weight:.0f}%")


def render_fundamental_table(analysis: Dict[str, Any]):
    """Render fundamental analysis as a table."""
    rows = []

    categories = ['valuation', 'profitability', 'financial_health', 'growth', 'dividends']

    for category in categories:
        cat_data = analysis.get(category, {})
        for metric, data in cat_data.items():
            if isinstance(data, dict) and 'value' in data:
                value = data.get('value')
                rating = data.get('rating', 'N/A')
                explanation = data.get('explanation', '')

                # Format value
                if value is not None:
                    if 'ratio' in metric.lower() or metric in ['roe', 'roa']:
                        formatted_value = f"{value:.2f}"
                    elif 'margin' in metric.lower() or 'growth' in metric.lower() or 'yield' in metric.lower():
                        formatted_value = f"{value*100:.2f}%" if value < 1 else f"{value:.2f}%"
                    else:
                        formatted_value = f"{value:.2f}"
                else:
                    formatted_value = "N/A"

                rows.append({
                    'Category': category.replace('_', ' ').title(),
                    'Metric': metric.replace('_', ' ').title(),
                    'Value': formatted_value,
                    'Rating': rating,
                    'Explanation': explanation
                })

    if rows:
        df = pd.DataFrame(rows)
        st.dataframe(
            df,
            hide_index=True,
            use_container_width=True
        )


def render_technical_signals(analysis: Dict[str, Any]):
    """Render technical analysis signals."""
    st.subheader("Technical Signals")

    col1, col2 = st.columns(2)

    with col1:
        # Trend
        trend = analysis.get('trend', {})
        direction = trend.get('direction', 'N/A')
        strength = trend.get('strength', 'N/A')
        color = "#00c853" if direction == 'Bullish' else "#f44336" if direction == 'Bearish' else "#ff9800"
        st.markdown(f"**Trend:** <span style='color: {color};'>{direction}</span> ({strength})",
                   unsafe_allow_html=True)

        # RSI
        rsi = analysis.get('rsi', {})
        rsi_value = rsi.get('value', 0)
        rsi_signal = rsi.get('signal', 'N/A')
        st.write(f"**RSI:** {rsi_value:.1f} - {rsi_signal}")

        # MACD
        macd = analysis.get('macd', {})
        macd_signal = macd.get('signal', 'N/A')
        crossover = macd.get('crossover', '')
        st.write(f"**MACD:** {macd_signal}" + (f" ({crossover})" if crossover else ""))

    with col2:
        # Moving Averages
        ma = analysis.get('moving_averages', {})
        st.write(f"**SMA 20:** ₹{ma.get('sma_20', 0):,.2f}")
        st.write(f"**SMA 50:** ₹{ma.get('sma_50', 0):,.2f}")
        if ma.get('sma_200'):
            st.write(f"**SMA 200:** ₹{ma.get('sma_200', 0):,.2f}")

        # Bollinger Bands
        bb = analysis.get('bollinger_bands', {})
        bb_signal = bb.get('signal', 'N/A')
        st.write(f"**Bollinger:** {bb_signal}")

    # Support/Resistance
    sr = analysis.get('support_resistance', {})
    if sr:
        st.write("---")
        col3, col4 = st.columns(2)

        with col3:
            st.write("**Support Levels:**")
            for level in sr.get('support_levels', [])[:3]:
                st.write(f"  ₹{level:,.2f}")

        with col4:
            st.write("**Resistance Levels:**")
            for level in sr.get('resistance_levels', [])[:3]:
                st.write(f"  ₹{level:,.2f}")


def render_prediction_results(prediction: Dict[str, Any]):
    """Render ML prediction results."""
    st.subheader("Price Prediction")

    ensemble = prediction.get('ensemble', {})
    horizon = prediction.get('horizon', 7)
    current_price = prediction.get('current_price', 0)

    col1, col2, col3 = st.columns(3)

    with col1:
        pred_change = ensemble.get('predicted_change_pct', 0)
        pred_price = ensemble.get('predicted_price', current_price)
        color = "#00c853" if pred_change > 0 else "#f44336"

        st.metric(
            f"{horizon}-Day Prediction",
            f"₹{pred_price:,.2f}",
            f"{pred_change:+.2f}%"
        )

    with col2:
        confidence = ensemble.get('confidence', 0) * 100
        st.metric("Confidence", f"{confidence:.0f}%")

    with col3:
        agreement = ensemble.get('agreement', 0) * 100
        st.metric("Model Agreement", f"{agreement:.0f}%")

    # Signal
    signal = ensemble.get('signal', 'N/A')
    st.info(f"**Signal:** {signal}")

    # Individual model predictions
    with st.expander("Individual Model Predictions"):
        individual = prediction.get('individual_models', {})
        for model, result in individual.items():
            if 'error' not in result:
                change = result.get('predicted_change_pct', 0)
                st.write(f"**{model.upper()}:** {change:+.2f}%")


def render_news_feed(articles: List[Dict[str, Any]]):
    """Render news articles feed."""
    st.subheader("Latest News")

    if not articles:
        st.info("No recent news found.")
        return

    for article in articles[:5]:
        with st.container():
            st.markdown(f"**[{article['title']}]({article['link']})**")

            col1, col2 = st.columns([3, 1])
            with col1:
                st.caption(f"{article['source']} - {article['date'][:10] if article['date'] else 'N/A'}")
            with col2:
                sentiment = article.get('sentiment', {}).get('sentiment', 'Neutral')
                color = "#00c853" if sentiment == 'Positive' else "#f44336" if sentiment == 'Negative' else "#ff9800"
                st.markdown(f"<span style='color: {color};'>{sentiment}</span>",
                           unsafe_allow_html=True)

            if article.get('summary'):
                st.write(article['summary'][:150] + "...")

            st.write("---")


def render_peer_comparison_table(comparison: Dict[str, Dict[str, Any]], main_symbol: str):
    """Render peer comparison table."""
    st.subheader("Peer Comparison")

    if not comparison:
        st.info("No peer data available.")
        return

    metrics_to_show = ['current_price', 'pe_ratio', 'pb_ratio', 'roe', 'profit_margin', 'debt_to_equity']

    rows = []
    for symbol, metrics in comparison.items():
        row = {'Symbol': symbol.replace('.NS', '')}
        for metric in metrics_to_show:
            value = metrics.get(metric)
            if value is not None:
                if 'ratio' in metric:
                    row[metric.replace('_', ' ').title()] = f"{value:.2f}"
                elif metric == 'current_price':
                    row['Price'] = f"₹{value:,.2f}"
                else:
                    row[metric.replace('_', ' ').title()] = f"{value*100:.1f}%" if value < 1 else f"{value:.1f}%"
            else:
                row[metric.replace('_', ' ').title()] = "N/A"
        rows.append(row)

    if rows:
        df = pd.DataFrame(rows)
        # Highlight main stock
        st.dataframe(df, hide_index=True, use_container_width=True)


def render_loading_spinner(text: str = "Loading..."):
    """Render loading spinner."""
    with st.spinner(text):
        pass


def render_error(message: str):
    """Render error message."""
    st.error(f"Error: {message}")


def render_warning(message: str):
    """Render warning message."""
    st.warning(message)


def render_success(message: str):
    """Render success message."""
    st.success(message)


def render_info(message: str):
    """Render info message."""
    st.info(message)
