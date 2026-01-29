"""
News fetching module using Google News RSS.
"""

import feedparser
import requests
from datetime import datetime, timedelta
from typing import List, Dict, Optional
import re
import logging
from urllib.parse import quote

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class NewsFetcher:
    """
    Fetches news articles related to stocks using Google News RSS.
    """

    def __init__(self, max_articles: int = 10):
        """
        Initialize news fetcher.

        Args:
            max_articles: Maximum number of articles to fetch
        """
        self.max_articles = max_articles
        self.base_url = "https://news.google.com/rss/search"

    def _clean_symbol(self, symbol: str) -> str:
        """Remove exchange suffix from symbol."""
        return symbol.replace('.NS', '').replace('.BO', '')

    def _parse_date(self, date_str: str) -> Optional[datetime]:
        """Parse date from RSS feed."""
        try:
            # Try common date formats
            formats = [
                '%a, %d %b %Y %H:%M:%S %Z',
                '%a, %d %b %Y %H:%M:%S %z',
                '%Y-%m-%dT%H:%M:%SZ',
            ]
            for fmt in formats:
                try:
                    return datetime.strptime(date_str, fmt)
                except ValueError:
                    continue
            return None
        except Exception:
            return None

    def _clean_html(self, text: str) -> str:
        """Remove HTML tags from text."""
        clean = re.sub(r'<[^>]+>', '', text)
        return clean.strip()

    def get_stock_news(
        self,
        symbol: str,
        company_name: Optional[str] = None
    ) -> List[Dict[str, str]]:
        """
        Get news articles for a stock.

        Args:
            symbol: Stock symbol
            company_name: Optional company name for better search

        Returns:
            List of news articles with title, link, date, source, summary
        """
        clean_symbol = self._clean_symbol(symbol)

        # Build search query
        if company_name:
            search_query = f"{company_name} stock India"
        else:
            search_query = f"{clean_symbol} stock India NSE"

        # URL encode the query
        encoded_query = quote(search_query)
        url = f"{self.base_url}?q={encoded_query}&hl=en-IN&gl=IN&ceid=IN:en"

        try:
            feed = feedparser.parse(url)

            if not feed.entries:
                logger.warning(f"No news found for {symbol}")
                return []

            articles = []
            for entry in feed.entries[:self.max_articles]:
                # Extract source from title
                title_parts = entry.title.split(' - ')
                title = ' - '.join(title_parts[:-1]) if len(title_parts) > 1 else entry.title
                source = title_parts[-1] if len(title_parts) > 1 else 'Unknown'

                # Parse date
                published = entry.get('published', '')
                date = self._parse_date(published)

                # Get summary
                summary = self._clean_html(entry.get('summary', ''))

                articles.append({
                    'title': title,
                    'link': entry.link,
                    'date': date.isoformat() if date else published,
                    'source': source,
                    'summary': summary[:300] + '...' if len(summary) > 300 else summary
                })

            return articles

        except Exception as e:
            logger.error(f"Error fetching news for {symbol}: {e}")
            return []

    def get_market_news(self) -> List[Dict[str, str]]:
        """
        Get general Indian stock market news.

        Returns:
            List of news articles
        """
        search_query = "Indian stock market NSE BSE Sensex Nifty"
        encoded_query = quote(search_query)
        url = f"{self.base_url}?q={encoded_query}&hl=en-IN&gl=IN&ceid=IN:en"

        try:
            feed = feedparser.parse(url)

            if not feed.entries:
                return []

            articles = []
            for entry in feed.entries[:self.max_articles]:
                title_parts = entry.title.split(' - ')
                title = ' - '.join(title_parts[:-1]) if len(title_parts) > 1 else entry.title
                source = title_parts[-1] if len(title_parts) > 1 else 'Unknown'

                published = entry.get('published', '')
                date = self._parse_date(published)
                summary = self._clean_html(entry.get('summary', ''))

                articles.append({
                    'title': title,
                    'link': entry.link,
                    'date': date.isoformat() if date else published,
                    'source': source,
                    'summary': summary[:300] + '...' if len(summary) > 300 else summary
                })

            return articles

        except Exception as e:
            logger.error(f"Error fetching market news: {e}")
            return []

    def get_sector_news(self, sector: str) -> List[Dict[str, str]]:
        """
        Get news for a specific sector.

        Args:
            sector: Sector name (e.g., 'IT', 'Banking', 'Pharma')

        Returns:
            List of news articles
        """
        search_query = f"India {sector} sector stocks"
        encoded_query = quote(search_query)
        url = f"{self.base_url}?q={encoded_query}&hl=en-IN&gl=IN&ceid=IN:en"

        try:
            feed = feedparser.parse(url)

            if not feed.entries:
                return []

            articles = []
            for entry in feed.entries[:self.max_articles]:
                title_parts = entry.title.split(' - ')
                title = ' - '.join(title_parts[:-1]) if len(title_parts) > 1 else entry.title
                source = title_parts[-1] if len(title_parts) > 1 else 'Unknown'

                published = entry.get('published', '')
                date = self._parse_date(published)
                summary = self._clean_html(entry.get('summary', ''))

                articles.append({
                    'title': title,
                    'link': entry.link,
                    'date': date.isoformat() if date else published,
                    'source': source,
                    'summary': summary[:300] + '...' if len(summary) > 300 else summary
                })

            return articles

        except Exception as e:
            logger.error(f"Error fetching sector news for {sector}: {e}")
            return []

    def analyze_sentiment_simple(self, text: str) -> Dict[str, float]:
        """
        Simple sentiment analysis using keyword matching.

        Args:
            text: Text to analyze

        Returns:
            Dictionary with sentiment scores
        """
        text_lower = text.lower()

        # Positive keywords
        positive_words = [
            'surge', 'jump', 'rally', 'gain', 'rise', 'up', 'high', 'growth',
            'profit', 'bullish', 'outperform', 'beat', 'strong', 'positive',
            'upgrade', 'buy', 'boom', 'soar', 'record', 'best', 'optimistic'
        ]

        # Negative keywords
        negative_words = [
            'fall', 'drop', 'decline', 'down', 'low', 'loss', 'bearish',
            'underperform', 'miss', 'weak', 'negative', 'downgrade', 'sell',
            'crash', 'plunge', 'worst', 'pessimistic', 'concern', 'fear', 'risk'
        ]

        positive_count = sum(1 for word in positive_words if word in text_lower)
        negative_count = sum(1 for word in negative_words if word in text_lower)

        total = positive_count + negative_count
        if total == 0:
            return {'positive': 0.5, 'negative': 0.5, 'sentiment': 'Neutral'}

        positive_score = positive_count / total
        negative_score = negative_count / total

        if positive_score > 0.6:
            sentiment = 'Positive'
        elif negative_score > 0.6:
            sentiment = 'Negative'
        else:
            sentiment = 'Neutral'

        return {
            'positive': positive_score,
            'negative': negative_score,
            'sentiment': sentiment
        }

    def get_news_with_sentiment(
        self,
        symbol: str,
        company_name: Optional[str] = None
    ) -> List[Dict[str, any]]:
        """
        Get news articles with sentiment analysis.

        Args:
            symbol: Stock symbol
            company_name: Optional company name

        Returns:
            List of articles with sentiment
        """
        articles = self.get_stock_news(symbol, company_name)

        for article in articles:
            # Analyze sentiment of title and summary
            combined_text = f"{article['title']} {article['summary']}"
            sentiment = self.analyze_sentiment_simple(combined_text)
            article['sentiment'] = sentiment

        return articles

    def get_overall_sentiment(
        self,
        symbol: str,
        company_name: Optional[str] = None
    ) -> Dict[str, any]:
        """
        Get overall sentiment from recent news.

        Args:
            symbol: Stock symbol
            company_name: Optional company name

        Returns:
            Dictionary with overall sentiment metrics
        """
        articles = self.get_news_with_sentiment(symbol, company_name)

        if not articles:
            return {
                'sentiment': 'Neutral',
                'score': 0.5,
                'article_count': 0,
                'positive_count': 0,
                'negative_count': 0,
                'neutral_count': 0
            }

        positive_count = sum(1 for a in articles if a['sentiment']['sentiment'] == 'Positive')
        negative_count = sum(1 for a in articles if a['sentiment']['sentiment'] == 'Negative')
        neutral_count = len(articles) - positive_count - negative_count

        # Calculate weighted score
        total = len(articles)
        score = (positive_count - negative_count + total) / (2 * total)

        if score > 0.6:
            overall = 'Positive'
        elif score < 0.4:
            overall = 'Negative'
        else:
            overall = 'Neutral'

        return {
            'sentiment': overall,
            'score': score,
            'article_count': total,
            'positive_count': positive_count,
            'negative_count': negative_count,
            'neutral_count': neutral_count
        }
