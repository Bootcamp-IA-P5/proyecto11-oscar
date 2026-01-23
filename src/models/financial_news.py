"""
Financial news module for Alpha Vantage and NewsAPI integration.

This module provides functionality to fetch real-time financial news
from multiple providers and format it as context for LLM content generation.
"""

import os
import requests
from typing import List, Dict
from datetime import datetime

from src.core.logger.logger import Logger
from src.core.logger.log_setup import log_setup

log_setup()
log = Logger().log


# ---------------------------------------------------------------------
# Alpha Vantage (your original implementation, unchanged)
# ---------------------------------------------------------------------

def load_financial_news(query: str, max_articles: int = 5) -> List[Dict]:
    """
    Fetch financial news from Alpha Vantage API.
    
    Args:
        query: Search query (e.g., "Tesla", "inflation", "interest rates")
        max_articles: Maximum number of articles to retrieve (default: 5)
    
    Returns:
        List of news articles with structured data:
        - title: Article headline
        - summary: Brief description
        - url: Link to full article
        - source: News source name
        - published_at: Publication timestamp
    """
    api_key = os.getenv("FINANCE_ALPHA_VANTAGE_KEY")
    
    if not api_key:
        log.warning("FINANCE_ALPHA_VANTAGE_KEY not configured. Skipping financial news.")
        return []
    
    try:
        url = "https://www.alphavantage.co/query"
        
        # Normalize and transliterate query to avoid mojibake/accents issues
        import re
        import unicodedata

        def _normalize(q: str) -> str:
            q = (q or "").strip()
            q_norm = unicodedata.normalize('NFKD', q)
            q_ascii = q_norm.encode('ascii', 'ignore').decode('ascii')
            return q_ascii or q

        query_norm = _normalize(query)

        # Add country/language keywords for better filtering if not present
        if "espa" in query_norm.lower() or "spain" in query_norm.lower():
            if "spain" not in query_norm.lower():
                query_norm += " Spain"
            if "spanish" not in query_norm.lower():
                query_norm += " Spanish"
        elif any(word in query_norm.lower() for word in ["inflacion", "precios", "cesta", "euros", "espanol"]):
            query_norm += " Spain Spanish"

        # Check if query looks like a ticker
        if re.match(r'^[A-Z0-9:_\-]+$', query_norm.upper()):
            params = {
                "function": "NEWS_SENTIMENT",
                "tickers": query_norm,
                "apikey": api_key,
                "limit": max_articles
            }
        else:
            params = {
                "function": "NEWS_SENTIMENT",
                "topics": query_norm,
                "apikey": api_key,
                "limit": max_articles
            }

        log.info(f"Fetching financial news for query: {query!r} (normalized: {query_norm!r})")
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        
        data = response.json()

        if "Error Message" in data:
            log.error(f"Alpha Vantage API error: {data['Error Message']}")
            return []
        
        if "Note" in data:
            log.warning(f"Alpha Vantage API rate limit: {data['Note']}")
            return []
        
        articles = []
        feed = data.get("feed", [])
        
        for item in feed[:max_articles]:
            article = {
                "title": item.get("title", "No title"),
                "summary": item.get("summary", "No summary available"),
                "url": item.get("url", ""),
                "source": item.get("source", "Unknown source"),
                "published_at": item.get("time_published", ""),
                "provider": "alpha"
            }
            articles.append(article)
        
        log.info(f"Successfully fetched {len(articles)} Alpha Vantage articles")

        if not articles:
            log.warning(f"No articles found in Alpha Vantage response. Keys: {list(data.keys())}")
        
        return articles
        
    except Exception as e:
        log.error(f"Unexpected error in load_financial_news: {e}")
        return []


# ---------------------------------------------------------------------
# NewsAPI integration (new)
# ---------------------------------------------------------------------

def load_financial_news_newsapi(query: str, max_articles: int = 5) -> List[Dict]:
    """
    Fetch financial and business-related news using NewsAPI.

    Args:
        query: Search query (e.g., "inflation", "markets", "Tesla")
        max_articles: Maximum number of articles to retrieve

    Returns:
        List of articles with normalized structure
    """
    api_key = os.getenv("FINANCE_NEWSAPI_KEY")

    if not api_key:
        log.warning("FINANCE_NEWSAPI_KEY not configured. Skipping NewsAPI.")
        return []

    try:
        url = "https://newsapi.org/v2/everything"
        params = {
            "q": query,
            "sortBy": "publishedAt",
            "language": "en",
            "pageSize": max_articles,
            "apiKey": api_key,
        }

        log.info(f"Fetching NewsAPI articles for query: {query!r}")
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()

        data = response.json()

        articles = []
        for item in data.get("articles", []):
            article = {
                "title": item.get("title", "No title"),
                "summary": item.get("description") or "",
                "url": item.get("url", ""),
                "source": item.get("source", {}).get("name", "NewsAPI"),
                "published_at": item.get("publishedAt", ""),
                "provider": "newsapi"
            }
            articles.append(article)

        log.info(f"Successfully fetched {len(articles)} NewsAPI articles")
        return articles

    except Exception as e:
        log.error(f"NewsAPI error: {e}")
        return []


# ---------------------------------------------------------------------
# Multi-source orchestrator (new)
# ---------------------------------------------------------------------

def load_financial_news_multi(query: str, max_articles: int = 5) -> List[Dict]:
    """
    Fetch financial news from multiple providers (Alpha Vantage + NewsAPI).

    Strategy:
    - Query both sources
    - Merge results
    - Remove duplicates by URL
    - Return top N results

    Args:
        query: Search query
        max_articles: Maximum total articles to return

    Returns:
        List of merged financial news articles
    """
    log.info(f"Multi-source financial news retrieval for query: {query!r}")

    alpha_articles = load_financial_news(query, max_articles)
    newsapi_articles = load_financial_news_newsapi(query, max_articles)

    combined = alpha_articles + newsapi_articles

    # Deduplicate by URL (professional-grade hygiene)
    seen = set()
    unique = []
    for a in combined:
        url = a.get("url")
        if url and url not in seen:
            seen.add(url)
            unique.append(a)

    log.info(f"Total merged financial articles: {len(unique)}")
    return unique[:max_articles]


# ---------------------------------------------------------------------
# Context formatting (your original, unchanged)
# ---------------------------------------------------------------------

def build_finance_context(news: List[Dict]) -> str:
    """
    Format financial news articles into structured context for LLM.
    
    Args:
        news: List of news article dictionaries
    
    Returns:
        Formatted context string ready for prompt injection
    """
    if not news:
        return ""
    
    context_lines = ["📈 FINANCIAL MARKET CONTEXT (real-time news):\n"]
    
    for idx, article in enumerate(news, 1):
        title = article.get("title", "No title")
        summary = article.get("summary", "")[:150]
        source = article.get("source", "Unknown")
        provider = article.get("provider", "")

        line = f"{idx}. {title} - {summary}... ({source} / {provider})"
        context_lines.append(line)
    
    context_lines.append("\nUse this information as factual grounding for financial content.\n")
    
    return "\n".join(context_lines)
