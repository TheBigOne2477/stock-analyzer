"""
Chart generation using Plotly.
"""

import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, List

from config import CHART_HEIGHT, CHART_COLORS


def create_candlestick_chart(
    df: pd.DataFrame,
    title: str = "Price Chart",
    show_volume: bool = True,
    indicators: Optional[Dict[str, pd.Series]] = None
) -> go.Figure:
    """
    Create candlestick chart with optional indicators.

    Args:
        df: DataFrame with OHLCV data
        title: Chart title
        show_volume: Whether to show volume bars
        indicators: Dictionary of indicator name to data series

    Returns:
        Plotly figure
    """
    rows = 2 if show_volume else 1
    row_heights = [0.7, 0.3] if show_volume else [1]

    fig = make_subplots(
        rows=rows, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        row_heights=row_heights
    )

    # Candlestick
    fig.add_trace(
        go.Candlestick(
            x=df.index,
            open=df['Open'],
            high=df['High'],
            low=df['Low'],
            close=df['Close'],
            name='Price',
            increasing_line_color=CHART_COLORS['positive'],
            decreasing_line_color=CHART_COLORS['negative']
        ),
        row=1, col=1
    )

    # Add indicators
    if indicators:
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
        for i, (name, data) in enumerate(indicators.items()):
            if data is not None and len(data) > 0:
                fig.add_trace(
                    go.Scatter(
                        x=data.index if hasattr(data, 'index') else df.index,
                        y=data,
                        mode='lines',
                        name=name,
                        line=dict(color=colors[i % len(colors)], width=1)
                    ),
                    row=1, col=1
                )

    # Volume
    if show_volume:
        colors = [CHART_COLORS['positive'] if df['Close'].iloc[i] >= df['Open'].iloc[i]
                  else CHART_COLORS['negative'] for i in range(len(df))]

        fig.add_trace(
            go.Bar(
                x=df.index,
                y=df['Volume'],
                name='Volume',
                marker_color=colors,
                opacity=0.5
            ),
            row=2, col=1
        )

    fig.update_layout(
        title=title,
        height=CHART_HEIGHT,
        xaxis_rangeslider_visible=False,
        showlegend=True,
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01
        )
    )

    fig.update_xaxes(title_text="Date", row=rows, col=1)
    fig.update_yaxes(title_text="Price (₹)", row=1, col=1)
    if show_volume:
        fig.update_yaxes(title_text="Volume", row=2, col=1)

    return fig


def create_technical_chart(
    df: pd.DataFrame,
    title: str = "Technical Analysis"
) -> go.Figure:
    """
    Create comprehensive technical analysis chart.

    Args:
        df: DataFrame with OHLCV and indicator data
        title: Chart title

    Returns:
        Plotly figure
    """
    fig = make_subplots(
        rows=4, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        row_heights=[0.4, 0.2, 0.2, 0.2],
        subplot_titles=('Price & Moving Averages', 'Volume', 'RSI', 'MACD')
    )

    # Candlestick
    fig.add_trace(
        go.Candlestick(
            x=df.index,
            open=df['Open'],
            high=df['High'],
            low=df['Low'],
            close=df['Close'],
            name='Price'
        ),
        row=1, col=1
    )

    # Moving Averages
    ma_columns = [col for col in df.columns if 'SMA' in col or 'EMA' in col]
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    for i, col in enumerate(ma_columns[:4]):
        fig.add_trace(
            go.Scatter(x=df.index, y=df[col], mode='lines', name=col,
                      line=dict(color=colors[i], width=1)),
            row=1, col=1
        )

    # Bollinger Bands
    if 'BB_Upper' in df.columns and 'BB_Lower' in df.columns:
        fig.add_trace(
            go.Scatter(x=df.index, y=df['BB_Upper'], mode='lines', name='BB Upper',
                      line=dict(color='gray', width=1, dash='dash')),
            row=1, col=1
        )
        fig.add_trace(
            go.Scatter(x=df.index, y=df['BB_Lower'], mode='lines', name='BB Lower',
                      line=dict(color='gray', width=1, dash='dash'),
                      fill='tonexty', fillcolor='rgba(128,128,128,0.1)'),
            row=1, col=1
        )

    # Volume
    colors = [CHART_COLORS['positive'] if df['Close'].iloc[i] >= df['Open'].iloc[i]
              else CHART_COLORS['negative'] for i in range(len(df))]
    fig.add_trace(
        go.Bar(x=df.index, y=df['Volume'], name='Volume', marker_color=colors, opacity=0.5),
        row=2, col=1
    )

    # RSI
    if 'RSI' in df.columns:
        fig.add_trace(
            go.Scatter(x=df.index, y=df['RSI'], mode='lines', name='RSI',
                      line=dict(color=CHART_COLORS['primary'])),
            row=3, col=1
        )
        fig.add_hline(y=70, line_dash="dash", line_color="red", row=3, col=1)
        fig.add_hline(y=30, line_dash="dash", line_color="green", row=3, col=1)

    # MACD
    if 'MACD' in df.columns:
        fig.add_trace(
            go.Scatter(x=df.index, y=df['MACD'], mode='lines', name='MACD',
                      line=dict(color=CHART_COLORS['primary'])),
            row=4, col=1
        )
        if 'MACD_Signal' in df.columns:
            fig.add_trace(
                go.Scatter(x=df.index, y=df['MACD_Signal'], mode='lines', name='Signal',
                          line=dict(color=CHART_COLORS['secondary'])),
                row=4, col=1
            )
        if 'MACD_Histogram' in df.columns:
            colors = [CHART_COLORS['positive'] if val >= 0 else CHART_COLORS['negative']
                     for val in df['MACD_Histogram']]
            fig.add_trace(
                go.Bar(x=df.index, y=df['MACD_Histogram'], name='Histogram',
                      marker_color=colors, opacity=0.5),
                row=4, col=1
            )

    fig.update_layout(
        title=title,
        height=800,
        xaxis_rangeslider_visible=False,
        showlegend=True
    )

    return fig


def create_fundamental_bar_chart(
    metrics: Dict[str, float],
    title: str = "Fundamental Metrics"
) -> go.Figure:
    """
    Create bar chart for fundamental metrics.

    Args:
        metrics: Dictionary of metric names to values
        title: Chart title

    Returns:
        Plotly figure
    """
    names = list(metrics.keys())
    values = list(metrics.values())

    # Color based on typical good/bad thresholds
    colors = []
    for name, value in metrics.items():
        if value is None:
            colors.append(CHART_COLORS['neutral'])
        elif 'growth' in name.lower() or 'roe' in name.lower() or 'roa' in name.lower():
            colors.append(CHART_COLORS['positive'] if value > 0 else CHART_COLORS['negative'])
        elif 'debt' in name.lower():
            colors.append(CHART_COLORS['negative'] if value > 1 else CHART_COLORS['positive'])
        else:
            colors.append(CHART_COLORS['primary'])

    fig = go.Figure(data=[
        go.Bar(
            x=names,
            y=values,
            marker_color=colors,
            text=[f"{v:.2f}" if v else "N/A" for v in values],
            textposition='outside'
        )
    ])

    fig.update_layout(
        title=title,
        height=400,
        xaxis_tickangle=-45
    )

    return fig


def create_score_radar_chart(scores: Dict[str, float], title: str = "Score Breakdown") -> go.Figure:
    """
    Create radar chart for score breakdown.

    Args:
        scores: Dictionary of category to score
        title: Chart title

    Returns:
        Plotly figure
    """
    categories = list(scores.keys())
    values = list(scores.values())

    # Close the radar chart
    categories = categories + [categories[0]]
    values = values + [values[0]]

    fig = go.Figure()

    fig.add_trace(go.Scatterpolar(
        r=values,
        theta=categories,
        fill='toself',
        fillcolor='rgba(31, 119, 180, 0.3)',
        line=dict(color=CHART_COLORS['primary'], width=2),
        name='Score'
    ))

    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 100]
            )
        ),
        showlegend=False,
        title=title,
        height=400
    )

    return fig


def create_peer_comparison_chart(
    comparison_data: Dict[str, Dict[str, float]],
    metric: str,
    title: str = None
) -> go.Figure:
    """
    Create bar chart comparing peers on a specific metric.

    Args:
        comparison_data: Dictionary of symbol to metrics
        metric: Metric to compare
        title: Chart title

    Returns:
        Plotly figure
    """
    symbols = []
    values = []

    for symbol, metrics in comparison_data.items():
        value = metrics.get(metric)
        if value is not None:
            symbols.append(symbol.replace('.NS', ''))
            values.append(value)

    if not symbols:
        return go.Figure()

    fig = go.Figure(data=[
        go.Bar(
            x=symbols,
            y=values,
            marker_color=CHART_COLORS['primary'],
            text=[f"{v:.2f}" for v in values],
            textposition='outside'
        )
    ])

    fig.update_layout(
        title=title or f"Peer Comparison - {metric.replace('_', ' ').title()}",
        height=400,
        xaxis_title="Stock",
        yaxis_title=metric.replace('_', ' ').title()
    )

    return fig


def create_prediction_chart(
    historical_prices: pd.Series,
    predictions: List[float],
    horizon: int,
    title: str = "Price Prediction"
) -> go.Figure:
    """
    Create chart showing historical prices and predictions.

    Args:
        historical_prices: Historical price series
        predictions: List of predicted prices
        horizon: Prediction horizon
        title: Chart title

    Returns:
        Plotly figure
    """
    fig = go.Figure()

    # Historical prices (last 60 days)
    hist_prices = historical_prices.tail(60)
    fig.add_trace(
        go.Scatter(
            x=hist_prices.index,
            y=hist_prices.values,
            mode='lines',
            name='Historical',
            line=dict(color=CHART_COLORS['primary'])
        )
    )

    # Prediction line
    if predictions:
        last_date = hist_prices.index[-1]
        if isinstance(last_date, pd.Timestamp):
            pred_dates = pd.date_range(start=last_date, periods=len(predictions) + 1, freq='D')[1:]
        else:
            pred_dates = range(len(hist_prices), len(hist_prices) + len(predictions))

        # Add connection point
        all_pred = [hist_prices.iloc[-1]] + predictions
        all_dates = [last_date] + list(pred_dates)

        fig.add_trace(
            go.Scatter(
                x=all_dates,
                y=all_pred,
                mode='lines+markers',
                name='Prediction',
                line=dict(color=CHART_COLORS['secondary'], dash='dash')
            )
        )

    fig.update_layout(
        title=title,
        height=CHART_HEIGHT,
        xaxis_title="Date",
        yaxis_title="Price (₹)",
        showlegend=True
    )

    return fig


def create_returns_distribution_chart(
    returns: pd.Series,
    title: str = "Returns Distribution"
) -> go.Figure:
    """
    Create histogram of returns distribution.

    Args:
        returns: Series of returns
        title: Chart title

    Returns:
        Plotly figure
    """
    fig = go.Figure()

    fig.add_trace(
        go.Histogram(
            x=returns * 100,
            nbinsx=50,
            name='Returns',
            marker_color=CHART_COLORS['primary'],
            opacity=0.7
        )
    )

    # Add mean line
    mean_return = returns.mean() * 100
    fig.add_vline(x=mean_return, line_dash="dash", line_color="red",
                  annotation_text=f"Mean: {mean_return:.2f}%")

    fig.update_layout(
        title=title,
        height=400,
        xaxis_title="Daily Return (%)",
        yaxis_title="Frequency",
        showlegend=False
    )

    return fig


def create_correlation_heatmap(
    correlation_matrix: pd.DataFrame,
    title: str = "Correlation Matrix"
) -> go.Figure:
    """
    Create correlation heatmap.

    Args:
        correlation_matrix: DataFrame with correlations
        title: Chart title

    Returns:
        Plotly figure
    """
    fig = go.Figure(data=go.Heatmap(
        z=correlation_matrix.values,
        x=correlation_matrix.columns,
        y=correlation_matrix.index,
        colorscale='RdBu',
        zmid=0,
        text=np.round(correlation_matrix.values, 2),
        texttemplate='%{text}',
        textfont={"size": 10},
        hoverongaps=False
    ))

    fig.update_layout(
        title=title,
        height=500,
        xaxis_tickangle=-45
    )

    return fig


def create_dcf_waterfall_chart(dcf_result: Dict[str, Any], title: str = "DCF Valuation") -> go.Figure:
    """
    Create waterfall chart for DCF breakdown.

    Args:
        dcf_result: DCF calculation results
        title: Chart title

    Returns:
        Plotly figure
    """
    labels = ['PV of FCF', 'Terminal Value PV', 'Enterprise Value']
    values = [
        dcf_result.get('sum_pv_fcf', 0),
        dcf_result.get('terminal_pv', 0),
        dcf_result.get('enterprise_value', 0)
    ]

    fig = go.Figure(go.Waterfall(
        name="DCF",
        orientation="v",
        x=labels,
        y=[values[0], values[1], 0],
        text=[f"₹{v:,.0f}" for v in values],
        textposition="outside",
        measure=["relative", "relative", "total"],
        connector={"line": {"color": "rgb(63, 63, 63)"}},
        increasing={"marker": {"color": CHART_COLORS['positive']}},
        totals={"marker": {"color": CHART_COLORS['primary']}}
    ))

    fig.update_layout(
        title=title,
        height=400,
        showlegend=False
    )

    return fig


def create_sensitivity_heatmap(
    sensitivity_df: pd.DataFrame,
    current_price: float,
    title: str = "Sensitivity Analysis"
) -> go.Figure:
    """
    Create heatmap for DCF sensitivity analysis.

    Args:
        sensitivity_df: DataFrame with sensitivity results
        current_price: Current stock price
        title: Chart title

    Returns:
        Plotly figure
    """
    # Calculate upside/downside relative to current price
    values = sensitivity_df.values
    relative_values = (values - current_price) / current_price * 100

    fig = go.Figure(data=go.Heatmap(
        z=relative_values,
        x=sensitivity_df.columns,
        y=sensitivity_df.index,
        colorscale='RdYlGn',
        zmid=0,
        text=np.round(values, 0),
        texttemplate='₹%{text:,.0f}',
        textfont={"size": 10},
        hoverongaps=False,
        colorbar=dict(title='Upside %')
    ))

    fig.update_layout(
        title=f"{title}<br><sub>Current Price: ₹{current_price:,.2f}</sub>",
        height=400,
        xaxis_title="Discount Rate",
        yaxis_title="Growth Rate"
    )

    return fig
