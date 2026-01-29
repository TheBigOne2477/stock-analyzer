# Indian Stock Market Analyzer

A comprehensive Streamlit-based web application for analyzing Indian public companies (NSE/BSE) with fundamental analysis, technical analysis, valuation models, ML-based price prediction, and investment recommendations.

## Features

- **Fundamental Analysis**: P/E, P/B, ROE, ROA, debt ratios, growth metrics
- **Technical Analysis**: Moving averages, RSI, MACD, Bollinger Bands, support/resistance
- **Valuation Models**: DCF analysis, comparable company analysis, Graham Number
- **ML Price Prediction**: LSTM and XGBoost models with ensemble predictions
- **Risk Scoring**: Multi-factor scoring system with investment recommendations
- **News Sentiment**: Real-time news aggregation with sentiment analysis
- **Peer Comparison**: Sector-based comparison with peers
- **Report Generation**: Comprehensive HTML reports

## Installation

### Prerequisites

- Python 3.9 or higher
- pip (Python package installer)

### Setup

1. **Clone or download the project**

```bash
cd stock-analyzer
```

2. **Create a virtual environment (recommended)**

```bash
python -m venv venv

# On Windows
venv\Scripts\activate

# On macOS/Linux
source venv/bin/activate
```

3. **Install dependencies**

```bash
pip install -r requirements.txt
```

Note: TensorFlow installation can be large. If you don't need LSTM predictions, you can skip it:
```bash
pip install streamlit pandas numpy yfinance ta scikit-learn xgboost plotly requests feedparser sqlalchemy
```

## Usage

1. **Start the application**

```bash
streamlit run app.py
```

2. **Open your browser** at http://localhost:8501

3. **Enter a stock symbol** (e.g., RELIANCE, TCS, INFY) and click "Analyze Stock"

## Stock Symbols

Use NSE stock symbols without the `.NS` suffix:
- RELIANCE (Reliance Industries)
- TCS (Tata Consultancy Services)
- INFY (Infosys)
- HDFCBANK (HDFC Bank)
- ICICIBANK (ICICI Bank)

The application automatically appends `.NS` for Yahoo Finance compatibility.

## Project Structure

```
stock-analyzer/
├── app.py                      # Main Streamlit application
├── requirements.txt            # Python dependencies
├── config.py                   # Configuration and constants
├── README.md                   # This file
│
├── data/
│   ├── __init__.py
│   ├── fetcher.py              # Stock data fetching (Yahoo Finance)
│   ├── news_fetcher.py         # News aggregation
│   └── cache.py                # Local caching with SQLite
│
├── analysis/
│   ├── __init__.py
│   ├── fundamental.py          # Fundamental analysis
│   ├── technical.py            # Technical indicators
│   └── valuation.py            # DCF and valuation models
│
├── prediction/
│   ├── __init__.py
│   ├── models.py               # ML models (LSTM, XGBoost)
│   ├── ensemble.py             # Ensemble prediction
│   └── backtester.py           # Backtesting engine
│
├── scoring/
│   ├── __init__.py
│   └── risk_scorer.py          # Risk-adjusted scoring
│
├── ui/
│   ├── __init__.py
│   ├── components.py           # Streamlit components
│   ├── charts.py               # Plotly charts
│   └── report.py               # Report generation
│
└── utils/
    ├── __init__.py
    └── helpers.py              # Utility functions
```

## Scoring System

The investment score (0-100) is calculated from:

| Category | Weight |
|----------|--------|
| Financial Health | 25% |
| Growth | 20% |
| Valuation | 25% |
| Technical Momentum | 15% |
| ML Prediction | 15% |

### Recommendation Thresholds

| Score | Recommendation |
|-------|----------------|
| 80-100 | Strong Buy |
| 65-79 | Buy |
| 45-64 | Hold |
| 30-44 | Sell |
| 0-29 | Strong Sell |

## Technical Indicators

- **Moving Averages**: SMA (20, 50, 200), EMA (12, 26)
- **Momentum**: RSI (14-period), Stochastic
- **Trend**: MACD (12, 26, 9), ADX
- **Volatility**: Bollinger Bands (20, 2), ATR
- **Volume**: Volume ratio, OBV

## ML Models

### LSTM (Long Short-Term Memory)
- 60-day lookback period
- 2 LSTM layers with dropout
- Best for capturing long-term patterns

### XGBoost
- Feature-based prediction
- Uses technical indicators as features
- Direction prediction accuracy typically 55-65%

### Ensemble
- Combines multiple model predictions
- Weighted by model confidence
- Provides consensus signal

## Limitations

- **Data Source**: Yahoo Finance (free tier) - may have delays
- **No Real-Time Data**: Uses daily closing prices
- **Fundamental Data**: Limited to what Yahoo Finance provides
- **Predictions**: ML predictions are probabilistic, not guaranteed

## Verification

Test the application with known stocks:
1. RELIANCE.NS - Large cap, Oil & Gas
2. TCS.NS - Large cap, IT sector
3. INFY.NS - Large cap, IT sector

Compare metrics with:
- [Screener.in](https://www.screener.in/)
- [MoneyControl](https://www.moneycontrol.com/)

## Disclaimer

**DISCLAIMER**: This tool is for educational and informational purposes only. It does not constitute financial advice. Stock market investments are subject to market risks. Past performance does not guarantee future results. Always consult a qualified financial advisor before making investment decisions.

## License

MIT License - See LICENSE file for details.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Support

If you encounter any issues, please open an issue on the repository.
