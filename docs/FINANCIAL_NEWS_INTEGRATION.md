# 💹 Financial News API Integration - Technical Specification

## 📋 Overview

This document outlines the technical implementation plan for integrating real-time financial news APIs into Nemotecas AI content generator.

## 🎯 Objectives

- Provide up-to-date market information for financial content generation
- Enhance RAG context with real-time news data
- Maintain system reliability with proper error handling and caching
- Support multiple financial data sources with fallback mechanisms

## 🏗️ Architecture

```
┌──────────────────────────────────────────────────────────┐
│                    Streamlit UI                          │
│  [x] Include real-time financial news                    │
└──────────────────────────────────────────────────────────┘
                          ↓
┌──────────────────────────────────────────────────────────┐
│              LangChain Orchestrator                      │
│  • Topic analysis                                        │
│  • Context enrichment decision                           │
└──────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────┬────────────────┬──────────────────────┐
│  RAG Engine     │  News Fetcher  │  LLM Generation      │
│  (arXiv Papers) │  (Financial)   │  (Groq/Gemini)       │
│                 │                │                      │
│  • ChromaDB     │  • News API    │  • Context from      │
│  • Scientific   │  • Finnhub     │    both sources      │
│    papers       │  • Alpha Vant. │  • Timestamped       │
└─────────────────┴────────────────┴──────────────────────┘
```

## 📡 Proposed APIs

### 1. News API (Primary)
- **URL**: https://newsapi.org
- **Free Tier**: 100 requests/day
- **Use Case**: General financial news articles
- **Endpoint**: `/v2/everything?q=finance&category=business`

### 2. Finnhub (Secondary)
- **URL**: https://finnhub.io
- **Free Tier**: 60 calls/minute
- **Use Case**: Stock market news, company news
- **Endpoint**: `/api/v1/news?category=general`

### 3. Alpha Vantage (Tertiary)
- **URL**: https://www.alphavantage.co
- **Free Tier**: 25 requests/day
- **Use Case**: Market data, news sentiment
- **Endpoint**: `/query?function=NEWS_SENTIMENT`

### 4. Yahoo Finance (Fallback)
- **URL**: Unofficial API via `yfinance` library
- **Free Tier**: No official limits (use responsibly)
- **Use Case**: Backup when other APIs fail

## 🔧 Implementation Plan

### Phase 1: Core Module (Week 1)

#### File: `src/models/financial_news_fetcher.py`

```python
"""
Financial news fetcher module.

This module integrates multiple financial news APIs to provide
real-time market data and news for content generation.

Supported APIs:
- News API (newsapi.org)
- Finnhub (finnhub.io)
- Alpha Vantage (alphavantage.co)
- Yahoo Finance (yfinance library)
"""

from datetime import datetime, timedelta
from typing import List, Dict, Optional
import requests
from functools import lru_cache
import os

class FinancialNewsFetcher:
    """
    Fetches financial news from multiple sources with caching and fallback.
    
    Attributes:
        cache_ttl (int): Time-to-live for cached results in seconds
        max_articles (int): Maximum number of articles to fetch per query
    """
    
    def __init__(self, cache_ttl: int = 3600, max_articles: int = 5):
        """
        Initialize the financial news fetcher.
        
        Args:
            cache_ttl: Cache expiration time in seconds (default: 1 hour)
            max_articles: Maximum articles to retrieve (default: 5)
        """
        self.news_api_key = os.getenv("NEWS_API_KEY")
        self.finnhub_key = os.getenv("FINNHUB_API_KEY")
        self.alphavantage_key = os.getenv("ALPHAVANTAGE_API_KEY")
        self.cache_ttl = cache_ttl
        self.max_articles = max_articles
    
    def fetch_news(self, query: str, days_back: int = 7) -> List[Dict]:
        """
        Fetch financial news from all available sources.
        
        Args:
            query: Search query (e.g., "bitcoin", "tesla", "inflation")
            days_back: How many days back to search (default: 7)
        
        Returns:
            List of news articles with metadata
        """
        # Implementation will try APIs in order until successful
        pass
    
    def _fetch_from_newsapi(self, query: str, from_date: str) -> List[Dict]:
        """Fetch news from News API."""
        pass
    
    def _fetch_from_finnhub(self, query: str) -> List[Dict]:
        """Fetch news from Finnhub API."""
        pass
    
    def _fetch_from_alphavantage(self, query: str) -> List[Dict]:
        """Fetch news from Alpha Vantage API."""
        pass
    
    def _fetch_from_yahoo(self, query: str) -> List[Dict]:
        """Fetch news from Yahoo Finance (fallback)."""
        pass
    
    @lru_cache(maxsize=100)
    def _cached_fetch(self, query: str, timestamp: str) -> str:
        """
        Cache mechanism to avoid hitting rate limits.
        
        Args:
            query: Search query
            timestamp: Hour-based timestamp for cache invalidation
        
        Returns:
            Cached results as JSON string
        """
        pass
```

### Phase 2: RAG Integration (Week 1)

#### File: `src/core/rag_engine.py` (modifications)

```python
# Add new method to ScienceRAG class

def get_context_with_news(
    self, 
    user_query: str, 
    include_news: bool = False
) -> str:
    """
    Retrieve context from both scientific papers and financial news.
    
    Args:
        user_query: User's topic query
        include_news: Whether to include real-time financial news
    
    Returns:
        Combined context from papers and news sources
    """
    # Get scientific context from arXiv papers
    scientific_context = self.get_context(user_query)
    
    if not include_news:
        return scientific_context
    
    # Fetch financial news
    from src.models.financial_news_fetcher import FinancialNewsFetcher
    news_fetcher = FinancialNewsFetcher()
    news_articles = news_fetcher.fetch_news(user_query)
    
    # Format news context
    news_context = self._format_news_context(news_articles)
    
    # Combine both contexts
    combined_context = f"""
SCIENTIFIC CONTEXT (from arXiv):
{scientific_context}

REAL-TIME NEWS CONTEXT (last 7 days):
{news_context}
"""
    
    return combined_context

def _format_news_context(self, articles: List[Dict]) -> str:
    """
    Format news articles into readable context.
    
    Args:
        articles: List of news article dictionaries
    
    Returns:
        Formatted string with news summaries
    """
    pass
```

### Phase 3: Configuration Updates (Week 1)

#### File: `config/settings.py`

```python
# Add financial news API configurations

# Financial News APIs
NEWS_API_KEY = os.getenv("NEWS_API_KEY")
FINNHUB_API_KEY = os.getenv("FINNHUB_API_KEY")
ALPHAVANTAGE_API_KEY = os.getenv("ALPHAVANTAGE_API_KEY")

# Financial news settings
FINANCIAL_NEWS_CACHE_TTL = int(os.getenv("FINANCIAL_NEWS_CACHE_TTL", "3600"))  # 1 hour
FINANCIAL_NEWS_MAX_ARTICLES = int(os.getenv("FINANCIAL_NEWS_MAX_ARTICLES", "5"))
```

#### File: `.env.example`

```bash
# ----------------------------------------------------------------------------
# FINANCIAL NEWS APIs
# ----------------------------------------------------------------------------
# News API key for general financial news
# Get your key from: https://newsapi.org/register
NEWS_API_KEY=<YOUR_NEWS_API_KEY>

# Finnhub API key for stock market data
# Get your key from: https://finnhub.io/register
FINNHUB_API_KEY=<YOUR_FINNHUB_API_KEY>

# Alpha Vantage API key for market data and sentiment
# Get your key from: https://www.alphavantage.co/support/#api-key
ALPHAVANTAGE_API_KEY=<YOUR_ALPHAVANTAGE_API_KEY>

# Financial news settings
FINANCIAL_NEWS_CACHE_TTL=3600
FINANCIAL_NEWS_MAX_ARTICLES=5
```

### Phase 4: UI Updates (Week 2)

#### File: `app.py` (modifications in render_sidebar)

```python
# Add financial news toggle in sidebar

with st.sidebar:
    # ... existing code ...
    
    # Financial news integration
    st.divider()
    st.subheader("💹 Financial Market Data")
    include_financial_news = st.checkbox(
        "Include real-time market news",
        value=False,
        help="Enhance content with up-to-date financial news from multiple sources"
    )
    
    if include_financial_news:
        st.info("📊 News sources: News API, Finnhub, Alpha Vantage")
        st.caption("⚠️ Rate limits: Data cached for 1 hour")
```

### Phase 5: Prompt Engineering (Week 2)

#### File: `config/prompts.py`

```python
# Add financial content generation template

FINANCIAL_CONTENT_TEMPLATE = """
[INST] <<SYS>>
You are a financial content creator with expertise in market analysis.
IMPORTANT: Include a disclaimer that this is not financial advice.
<</SYS>>

BRAND PROFILE (Context):
{brand_bio}

SCIENTIFIC CONTEXT (Research Background):
{scientific_context}

REAL-TIME MARKET NEWS (Last 7 days):
{news_context}

TASK DETAILS:
- TOPIC: {topic}
- AUDIENCE: {audience}
- OUTPUT LANGUAGE: {target_language}

INSTRUCTIONS:
1. Write a professional financial article (600-1000 words) in {target_language}
2. Combine scientific research with recent market news
3. Cite news sources with dates
4. Include this disclaimer at the end:
   "⚠️ Disclaimer: This content is for informational purposes only and 
   does not constitute financial advice. Always consult with a qualified 
   financial advisor before making investment decisions."
5. Use Markdown formatting

FINANCIAL ARTICLE IN {target_language}:
[/INST]
"""
```

## 🧪 Testing Strategy

### Unit Tests

#### File: `tests/test_financial_news_fetcher.py`

```python
"""
Unit tests for financial news fetcher module.
"""

import unittest
from unittest.mock import patch, MagicMock
from src.models.financial_news_fetcher import FinancialNewsFetcher

class TestFinancialNewsFetcher(unittest.TestCase):
    """Test cases for FinancialNewsFetcher class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.fetcher = FinancialNewsFetcher()
    
    @patch('requests.get')
    def test_fetch_from_newsapi_success(self, mock_get):
        """Test successful news fetch from News API."""
        # Mock response
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            'articles': [
                {
                    'title': 'Bitcoin hits new high',
                    'description': 'BTC reaches $100k',
                    'publishedAt': '2026-01-20',
                    'source': {'name': 'Bloomberg'}
                }
            ]
        }
        mock_get.return_value = mock_response
        
        # Test
        articles = self.fetcher._fetch_from_newsapi('bitcoin', '2026-01-14')
        
        # Assert
        self.assertEqual(len(articles), 1)
        self.assertEqual(articles[0]['title'], 'Bitcoin hits new high')
    
    @patch('requests.get')
    def test_fetch_news_with_fallback(self, mock_get):
        """Test fallback mechanism when primary API fails."""
        # Mock News API failure, Finnhub success
        mock_get.side_effect = [
            Exception("News API rate limit"),  # First call fails
            MagicMock(status_code=200, json=lambda: {'news': []})  # Second succeeds
        ]
        
        # Test
        articles = self.fetcher.fetch_news('tesla')
        
        # Assert fallback was used
        self.assertIsNotNone(articles)
    
    def test_cache_mechanism(self):
        """Test that caching prevents redundant API calls."""
        # Implementation test for cache
        pass

if __name__ == '__main__':
    unittest.main()
```

## 📊 Success Metrics

- **API Response Time**: < 2 seconds average
- **Cache Hit Rate**: > 70%
- **Error Rate**: < 5%
- **Cost**: Stay within free tier limits
- **User Adoption**: Track checkbox usage in analytics

## ⚠️ Risk Mitigation

| Risk | Impact | Mitigation |
|------|--------|------------|
| API rate limits exceeded | High | Aggressive caching, fallback sources |
| API downtime | Medium | Multi-provider strategy, graceful degradation |
| Stale data | Low | Timestamp validation, cache TTL = 1 hour |
| Incorrect financial info | High | Clear disclaimers, cite sources |
| Cost overruns | Medium | Monitor usage, set hard limits |

## 🚀 Deployment Checklist

- [ ] Environment variables configured in production
- [ ] Rate limiting tested under load
- [ ] Cache mechanism validated
- [ ] Error logging implemented
- [ ] API keys secured (not in code)
- [ ] Fallback sources tested
- [ ] Financial disclaimer present in all outputs
- [ ] Documentation updated
- [ ] Demo video created
- [ ] Feature flag enabled

## 📚 References

- [News API Documentation](https://newsapi.org/docs)
- [Finnhub API Documentation](https://finnhub.io/docs/api)
- [Alpha Vantage Documentation](https://www.alphavantage.co/documentation/)
- [LangChain External Data Integration](https://python.langchain.com/docs/use_cases/web_scraping)

## 👥 Contributors

- **Feature Owner**: [Assigned team member]
- **Technical Lead**: [Assigned team member]
- **QA**: [Assigned team member]

## 📅 Timeline

- **Week 1**: Core module + RAG integration + Configuration
- **Week 2**: UI updates + Prompt engineering + Testing
- **Week 3**: Documentation + Demo + Code review
- **Week 4**: Deployment + Monitoring

---

**Last Updated**: January 21, 2026
**Status**: Planning Phase
**Branch**: `feature/financial-news-api-integration`
