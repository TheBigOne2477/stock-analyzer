"""
Indian Stock Market Analyzer - Main Streamlit Application
"""

import streamlit as st
import pandas as pd
from datetime import datetime, timedelta
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import (
    APP_NAME, APP_VERSION, DEFAULT_STOCKS, SECTORS,
    DISCLAIMER, ML_PARAMS
)
from data.fetcher import StockDataFetcher
from data.news_fetcher import NewsFetcher
from analysis.fundamental import FundamentalAnalyzer
from analysis.technical import TechnicalAnalyzer
from analysis.valuation import ValuationModel
from prediction.ensemble import EnsemblePredictor
from prediction.backtester import Backtester
from scoring.risk_scorer import RiskScorer
from ui.components import (
    render_header, render_disclaimer, render_key_metrics,
    render_company_info, render_recommendation_badge,
    render_score_breakdown, render_fundamental_table,
    render_technical_signals, render_prediction_results,
    render_news_feed, render_peer_comparison_table,
    render_error, render_warning, render_info
)
from ui.charts import (
    create_candlestick_chart, create_technical_chart,
    create_score_radar_chart, create_peer_comparison_chart,
    create_prediction_chart, create_sensitivity_heatmap
)
from ui.report import ReportGenerator
from utils.helpers import normalize_stock_symbol, get_sector_for_stock


# Page configuration
st.set_page_config(
    page_title=APP_NAME,
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if 'analyzed_symbol' not in st.session_state:
    st.session_state.analyzed_symbol = None
if 'analysis_data' not in st.session_state:
    st.session_state.analysis_data = {}


def main():
    """Main application entry point."""
    render_header()

    # Sidebar
    with st.sidebar:
        st.header("Settings")

        # Stock input
        symbol_input = st.text_input(
            "Enter Stock Symbol",
            placeholder="e.g., RELIANCE, TCS, HDFCBANK",
            help="Enter any valid NSE stock symbol without .NS suffix. Examples: RELIANCE, TCS, INFY, WIPRO, MARUTI, TATAMOTORS, SUNPHARMA, AXISBANK, LT, ASIANPAINT"
        )

        # Quick select
        st.subheader("Quick Select")
        selected_quick = st.selectbox(
            "Popular Stocks",
            ["Select..."] + [s.replace('.NS', '') for s in DEFAULT_STOCKS]
        )

        if selected_quick != "Select...":
            symbol_input = selected_quick

        # Analysis period
        period = st.selectbox(
            "Analysis Period",
            ["1y", "6mo", "3mo", "2y", "5y"],
            index=0
        )

        # Prediction settings
        st.subheader("Prediction Settings")
        pred_horizon = st.selectbox(
            "Prediction Horizon",
            [7, 30, 90],
            index=0,
            format_func=lambda x: f"{x} days"
        )

        use_lstm = st.checkbox("Use LSTM Model", value=False,
                               help="LSTM requires TensorFlow and takes longer to train")
        use_xgboost = st.checkbox("Use XGBoost Model", value=True)

        # Analyze button
        analyze_button = st.button("Analyze Stock", type="primary", use_container_width=True)

        st.divider()
        render_disclaimer()

    # Main content
    if analyze_button and symbol_input:
        symbol = normalize_stock_symbol(symbol_input.upper())
        run_analysis(symbol, period, pred_horizon, use_lstm, use_xgboost)

    elif st.session_state.analyzed_symbol:
        # Display cached analysis
        display_analysis(st.session_state.analysis_data)


def run_analysis(symbol: str, period: str, pred_horizon: int,
                 use_lstm: bool, use_xgboost: bool):
    """Run complete stock analysis."""
    with st.spinner(f"Analyzing {symbol}..."):
        try:
            # Initialize components (disable cache for cloud deployment)
            fetcher = StockDataFetcher(use_cache=False)
            fundamental_analyzer = FundamentalAnalyzer()
            technical_analyzer = TechnicalAnalyzer()
            valuation_model = ValuationModel()
            news_fetcher = NewsFetcher()
            scorer = RiskScorer()

            # Fetch data
            st.info("Fetching stock data...")

            # Validate symbol
            info = fetcher.get_stock_info(symbol)
            if not info:
                render_error(f"Could not find stock: {symbol}. Please check the symbol and try again.")
                st.info("💡 **Tips:**\n- Use NSE symbols like RELIANCE, TCS, INFY, HDFCBANK\n- Don't add .NS suffix - it's added automatically\n- Make sure the stock is listed on NSE")
                return

            # Get historical data
            df = fetcher.get_historical_data(symbol, period=period)
            if df is None or df.empty:
                render_error(f"No price data available for {symbol}")
                return

            # Get metrics and financials
            metrics = fetcher.get_key_metrics(symbol)
            financials = fetcher.get_financials(symbol)

            # Run fundamental analysis
            st.info("Running fundamental analysis...")
            fundamental_results = fundamental_analyzer.analyze(metrics)

            # Run technical analysis
            st.info("Running technical analysis...")
            technical_results = technical_analyzer.analyze(df)
            df_with_indicators = technical_analyzer.add_all_indicators(df)

            # Run valuation analysis
            st.info("Running valuation analysis...")
            dcf_result = valuation_model.auto_dcf(metrics, financials)
            valuation_summary = valuation_model.get_valuation_summary(
                metrics, dcf_result, None
            )

            # Run ML prediction
            prediction_result = None
            if use_lstm or use_xgboost:
                st.info("Training prediction models...")
                predictor = EnsemblePredictor(use_lstm=use_lstm, use_xgboost=use_xgboost)
                predictor.train(df, horizons=[pred_horizon])
                prediction_result = predictor.predict(df, pred_horizon)

            # Fetch news
            st.info("Fetching news...")
            company_name = metrics.get('company_name')
            news_articles = news_fetcher.get_news_with_sentiment(symbol, company_name)
            news_sentiment = news_fetcher.get_overall_sentiment(symbol, company_name)

            # Calculate overall score
            st.info("Calculating investment score...")
            score = scorer.calculate_score(
                fundamental_results,
                technical_results,
                valuation_summary,
                prediction_result,
                news_sentiment
            )
            score_breakdown = scorer.get_score_breakdown(score)

            # Get peer data
            sector = get_sector_for_stock(symbol, SECTORS)
            peer_data = {}
            if sector:
                peers = SECTORS.get(sector, [])[:5]
                peer_data = fetcher.get_peer_comparison_data(symbol, peers)

            # Store results
            analysis_data = {
                'symbol': symbol,
                'metrics': metrics,
                'df': df,
                'df_with_indicators': df_with_indicators,
                'fundamental': fundamental_results,
                'technical': technical_results,
                'valuation': valuation_summary,
                'dcf': dcf_result,
                'prediction': prediction_result,
                'news': news_articles,
                'news_sentiment': news_sentiment,
                'score': score,
                'score_breakdown': score_breakdown,
                'peer_data': peer_data,
                'sector': sector,
                'pred_horizon': pred_horizon
            }

            st.session_state.analyzed_symbol = symbol
            st.session_state.analysis_data = analysis_data

            # Display results
            display_analysis(analysis_data)

        except Exception as e:
            render_error(f"Analysis failed: {str(e)}")
            import traceback
            st.code(traceback.format_exc())


def display_analysis(data: dict):
    """Display analysis results."""
    if not data:
        return

    symbol = data['symbol']
    metrics = data['metrics']
    score = data['score']
    score_breakdown = data['score_breakdown']

    # Header with recommendation
    col1, col2 = st.columns([2, 1])
    with col1:
        render_company_info(metrics)
    with col2:
        render_recommendation_badge(score.recommendation, score.total_score)

    # Tabs for different sections
    tabs = st.tabs([
        "📊 Overview",
        "📈 Technical",
        "💰 Fundamental",
        "🎯 Valuation",
        "🤖 Prediction",
        "👥 Peers",
        "📰 News",
        "📄 Report"
    ])

    # Overview Tab
    with tabs[0]:
        render_key_metrics(metrics)

        st.divider()

        col1, col2 = st.columns([2, 1])

        with col1:
            # Price chart
            df = data['df']
            indicators = {}
            if 'SMA_20' in data['df_with_indicators'].columns:
                indicators['SMA 20'] = data['df_with_indicators']['SMA_20']
            if 'SMA_50' in data['df_with_indicators'].columns:
                indicators['SMA 50'] = data['df_with_indicators']['SMA_50']

            fig = create_candlestick_chart(df, f"{symbol} Price Chart", indicators=indicators)
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            # Score radar chart
            scores = {
                'Financial': score.financial_health,
                'Growth': score.growth,
                'Valuation': score.valuation,
                'Technical': score.technical_momentum,
                'Prediction': score.prediction_confidence
            }
            fig = create_score_radar_chart(scores)
            st.plotly_chart(fig, use_container_width=True)

            # Key factors
            st.subheader("Key Factors")
            for reason in score.reasoning[:5]:
                st.write(f"• {reason}")

    # Technical Tab
    with tabs[1]:
        technical = data['technical']
        df_ind = data['df_with_indicators']

        col1, col2 = st.columns([2, 1])

        with col1:
            fig = create_technical_chart(df_ind, f"{symbol} Technical Analysis")
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            render_technical_signals(technical)

        st.subheader("Technical Summary")
        st.write(technical.get('summary', 'N/A'))

    # Fundamental Tab
    with tabs[2]:
        fundamental = data['fundamental']

        st.subheader("Fundamental Analysis")

        # Score by category
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Valuation Score", f"{fundamental.get('valuation', {}).get('score', 0):.0f}/100")
        with col2:
            st.metric("Profitability Score", f"{fundamental.get('profitability', {}).get('score', 0):.0f}/100")
        with col3:
            st.metric("Financial Health", f"{fundamental.get('financial_health', {}).get('score', 0):.0f}/100")
        with col4:
            st.metric("Growth Score", f"{fundamental.get('growth', {}).get('score', 0):.0f}/100")

        st.divider()
        render_fundamental_table(fundamental)

        if fundamental.get('summary'):
            st.subheader("Summary")
            st.write(fundamental['summary'])

    # Valuation Tab
    with tabs[3]:
        valuation = data['valuation']
        dcf = data['dcf']

        st.subheader("Valuation Analysis")

        if valuation:
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Current Price", f"₹{valuation.get('current_price', 0):,.2f}")
            with col2:
                fair_value = valuation.get('weighted_fair_value')
                if fair_value:
                    st.metric("Fair Value", f"₹{fair_value:,.2f}")
                else:
                    st.metric("Fair Value", "N/A")
            with col3:
                margin = valuation.get('margin_of_safety')
                if margin is not None:
                    st.metric("Margin of Safety", f"{margin:+.1f}%")
                else:
                    st.metric("Margin of Safety", "N/A")

            # Valuation methods breakdown
            st.subheader("Valuation Methods")
            for v in valuation.get('valuations', []):
                col1, col2, col3 = st.columns([2, 1, 1])
                with col1:
                    st.write(f"**{v['method']}**")
                with col2:
                    st.write(f"₹{v['value']:,.2f}")
                with col3:
                    st.write(f"Weight: {v['weight']*100:.0f}%")

            # DCF Details
            if dcf:
                with st.expander("DCF Model Details"):
                    inputs = dcf.get('inputs', {})
                    st.write(f"**Free Cash Flow:** ₹{inputs.get('free_cash_flow', 0):,.0f}")
                    st.write(f"**Growth Rate:** {inputs.get('growth_rate', 0)*100:.1f}%")
                    st.write(f"**Discount Rate:** {inputs.get('discount_rate', 0)*100:.1f}%")
                    st.write(f"**Terminal Growth:** {inputs.get('terminal_growth', 0)*100:.1f}%")

                    # Sensitivity analysis
                    sensitivity_df = valuation_model = ValuationModel()
                    sens = valuation_model.sensitivity_analysis(dcf)

                    st.subheader("Sensitivity Analysis")
                    fig = create_sensitivity_heatmap(sens, valuation.get('current_price', 0))
                    st.plotly_chart(fig, use_container_width=True)

            st.subheader("Recommendation")
            st.info(valuation.get('recommendation', 'N/A'))
        else:
            render_warning("Insufficient data for valuation analysis")

    # Prediction Tab
    with tabs[4]:
        prediction = data.get('prediction')

        st.subheader("ML Price Prediction")

        if prediction:
            render_prediction_results(prediction)

            # Prediction chart
            df = data['df']
            ensemble = prediction.get('ensemble', {})
            pred_price = ensemble.get('predicted_price')

            if pred_price:
                predictions = [pred_price]  # Simplified - just endpoint
                fig = create_prediction_chart(
                    df['Close'],
                    predictions,
                    data.get('pred_horizon', 7)
                )
                st.plotly_chart(fig, use_container_width=True)

            # Model details
            with st.expander("Individual Model Results"):
                for model, result in prediction.get('individual_models', {}).items():
                    if 'error' not in result:
                        st.write(f"**{model.upper()}:**")
                        st.json(result)
        else:
            render_info("Enable ML models in settings to see predictions")

        # Backtesting section
        st.subheader("Backtest (Quick)")
        if st.button("Run Quick Backtest"):
            with st.spinner("Running backtest..."):
                backtester = Backtester()
                backtest_result = backtester.run_quick_backtest(
                    data['df'],
                    horizon=data.get('pred_horizon', 7)
                )

                if 'error' not in backtest_result:
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Tests", backtest_result['tests'])
                    with col2:
                        st.metric("Direction Accuracy", f"{backtest_result['direction_accuracy']*100:.1f}%")
                    with col3:
                        st.metric("MAE", f"{backtest_result['mean_absolute_error']:.2f}%")
                else:
                    render_error(backtest_result['error'])

    # Peers Tab
    with tabs[5]:
        peer_data = data.get('peer_data', {})
        sector = data.get('sector')

        st.subheader(f"Peer Comparison - {sector or 'Unknown Sector'}")

        if peer_data:
            render_peer_comparison_table(peer_data, symbol)

            # Comparison charts
            metrics_to_compare = ['pe_ratio', 'pb_ratio', 'roe', 'profit_margin']
            cols = st.columns(2)

            for i, metric in enumerate(metrics_to_compare):
                with cols[i % 2]:
                    fig = create_peer_comparison_chart(peer_data, metric)
                    st.plotly_chart(fig, use_container_width=True)
        else:
            render_info("No peer data available for this stock")

    # News Tab
    with tabs[6]:
        news = data.get('news', [])
        news_sentiment = data.get('news_sentiment', {})

        st.subheader("News Sentiment")

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Overall Sentiment", news_sentiment.get('sentiment', 'N/A'))
        with col2:
            st.metric("Sentiment Score", f"{news_sentiment.get('score', 0.5)*100:.0f}%")
        with col3:
            st.metric("Positive Articles", news_sentiment.get('positive_count', 0))
        with col4:
            st.metric("Negative Articles", news_sentiment.get('negative_count', 0))

        st.divider()
        render_news_feed(news)

    # Report Tab
    with tabs[7]:
        st.subheader("Generate Report")

        st.write("Generate a comprehensive HTML report that you can save or print.")

        if st.button("Generate Report", type="primary"):
            with st.spinner("Generating report..."):
                generator = ReportGenerator()

                html_report = generator.generate_report(
                    symbol=symbol,
                    metrics=metrics,
                    fundamental_analysis=data['fundamental'],
                    technical_analysis=data['technical'],
                    valuation_analysis=data['valuation'],
                    prediction_result=data.get('prediction'),
                    score_breakdown=score_breakdown,
                    news_articles=data.get('news')
                )

                # Download button
                st.download_button(
                    label="Download Report (HTML)",
                    data=html_report,
                    file_name=f"{symbol}_analysis_{datetime.now().strftime('%Y%m%d')}.html",
                    mime="text/html"
                )

                # Preview
                with st.expander("Preview Report"):
                    st.components.v1.html(html_report, height=800, scrolling=True)


if __name__ == "__main__":
    main()
