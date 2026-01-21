"""
Financial news module for Alpha Vantage API integration.

This module provides functionality to fetch real-time financial news
and format it as context for LLM content generation.
"""

import os
import requests
from typing import List, Dict, Optional
from datetime import datetime

from src.core.logger.logger import Logger
from src.core.logger.log_setup import log_setup

log_setup()
log = Logger().log


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
    
    Raises:
        Exception: If API call fails or key is missing
    """
    api_key = os.getenv("FINANCE_ALPHA_VANTAGE_KEY")
    
    if not api_key:
        log.warning("FINANCE_ALPHA_VANTAGE_KEY not configured. Skipping financial news.")
        return []
    
    try:
        url = "https://www.alphavantage.co/query"
        params = {
            "function": "NEWS_SENTIMENT",
            "tickers": query,
            "apikey": api_key,
            "limit": max_articles
        }
        
        log.info(f"Fetching financial news for query: {query}")
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        
        data = response.json()
        
        # Check for API errors
        if "Error Message" in data:
            log.error(f"Alpha Vantage API error: {data['Error Message']}")
            return []
        
        if "Note" in data:
            log.warning(f"Alpha Vantage API rate limit: {data['Note']}")
            return []
        
        # Extract and format articles
        articles = []
        feed = data.get("feed", [])
        
        for item in feed[:max_articles]:
            article = {
                "title": item.get("title", "No title"),
                "summary": item.get("summary", "No summary available"),
                "url": item.get("url", ""),
                "source": item.get("source", "Unknown source"),
                "published_at": item.get("time_published", "")
            }
            articles.append(article)
        
        log.info(f"Successfully fetched {len(articles)} financial news articles")
        return articles
        
    except requests.exceptions.Timeout:
        log.error("Alpha Vantage API request timed out")
        return []
    except requests.exceptions.RequestException as e:
        log.error(f"Error fetching financial news: {e}")
        return []
    except Exception as e:
        log.error(f"Unexpected error in load_financial_news: {e}")
        return []


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
        summary = article.get("summary", "")[:150]  # Limit summary length
        source = article.get("source", "Unknown")
        
        # Format: "1. Title - short summary (source)"
        line = f"{idx}. {title} - {summary}... ({source})"
        context_lines.append(line)
    
    context_lines.append("\nUse this information as factual grounding for financial content.\n")
    
    return "\n".join(context_lines)
