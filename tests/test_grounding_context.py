"""
Tests for grounding context behavior.

This module tests the multi-source grounding logic to ensure:
- Pure LLM generation (no RAG, no Financial)
- RAG only context
- Financial only context  
- Combined RAG + Financial context

Run with: pytest tests/test_grounding_context.py -v
"""
import pytest
from unittest.mock import patch, MagicMock

from src.core.content_chains import (
    _prepare_financial_context,
    assemble_grounding_context,
    get_grounding_summary
)


class TestPrepareFinancialContext:
    """Tests for _prepare_financial_context function."""
    
    def test_disabled_returns_empty(self):
        """When finance is disabled, should return empty context and list."""
        context, articles = _prepare_financial_context(use_finance=False)
        assert context == ""
        assert articles == []
    
    def test_no_query_returns_empty(self):
        """When no query provided, should return empty."""
        context, articles = _prepare_financial_context(
            use_finance=True, 
            finance_query=None
        )
        assert context == ""
        assert articles == []
    
    @patch('src.core.content_chains.load_financial_news')
    @patch('src.core.content_chains.build_finance_context')
    def test_enabled_with_articles(self, mock_build, mock_load):
        """When enabled and articles found, should return context."""
        mock_articles = [
            {"title": "Test News", "summary": "Summary", "source": "Test"}
        ]
        mock_load.return_value = mock_articles
        mock_build.return_value = "📈 FINANCIAL CONTEXT"
        
        context, articles = _prepare_financial_context(
            use_finance=True,
            finance_query="Tesla",
            finance_max_articles=5
        )
        
        assert context == "📈 FINANCIAL CONTEXT"
        assert articles == mock_articles
        mock_load.assert_called_once_with("Tesla", 5)


class TestAssembleGroundingContext:
    """Tests for assemble_grounding_context function."""
    
    def test_no_context_returns_empty(self):
        """No RAG, no Financial -> empty string."""
        result = assemble_grounding_context(
            rag_context="",
            financial_context=""
        )
        assert result == ""
    
    def test_rag_only_context(self):
        """RAG enabled only -> includes scientific label."""
        rag_content = "This is content from arXiv papers about LLMs."
        result = assemble_grounding_context(
            rag_context=rag_content,
            financial_context=""
        )
        
        assert "SCIENTIFIC CONTEXT" in result
        assert "arXiv" in result
        assert rag_content in result
    
    def test_financial_only_context(self):
        """Financial enabled only -> includes financial content."""
        financial_content = "📈 FINANCIAL MARKET CONTEXT"
        result = assemble_grounding_context(
            rag_context="",
            financial_context=financial_content
        )
        
        assert financial_content in result
        assert "SCIENTIFIC CONTEXT" not in result
    
    def test_combined_context(self):
        """Both enabled -> includes both with proper labels."""
        rag_content = "Scientific paper content"
        financial_content = "📈 Financial news content"
        
        result = assemble_grounding_context(
            rag_context=rag_content,
            financial_context=financial_content
        )
        
        assert "SCIENTIFIC CONTEXT" in result
        assert rag_content in result
        assert financial_content in result
    
    def test_debug_mode_logs(self, caplog):
        """Debug mode should log context info."""
        import logging
        caplog.set_level(logging.INFO)
        
        assemble_grounding_context(
            rag_context="Test content",
            financial_context="",
            debug=True
        )
        
        # Check that debug logging occurred
        assert "DEBUG: ASSEMBLED GROUNDING CONTEXT" in caplog.text or len(caplog.records) > 0


class TestGetGroundingSummary:
    """Tests for get_grounding_summary function."""
    
    def test_no_sources(self):
        """No sources -> is_grounded False."""
        summary = get_grounding_summary(
            rag_documents="",
            financial_articles=None
        )
        
        assert summary["is_grounded"] is False
        assert summary["rag_enabled"] is False
        assert summary["financial_enabled"] is False
        assert summary["sources_used"] == []
    
    def test_rag_only(self):
        """RAG documents present."""
        summary = get_grounding_summary(
            rag_documents="Chunk 1\n\nChunk 2\n\nChunk 3",
            financial_articles=None
        )
        
        assert summary["is_grounded"] is True
        assert summary["rag_enabled"] is True
        assert summary["financial_enabled"] is False
        assert summary["rag_doc_count"] == 3
        assert "arXiv" in summary["sources_used"][0]
    
    def test_financial_only(self):
        """Financial articles present."""
        articles = [
            {"title": "News 1", "source": "Bloomberg"},
            {"title": "News 2", "source": "Reuters"}
        ]
        summary = get_grounding_summary(
            rag_documents="",
            financial_articles=articles
        )
        
        assert summary["is_grounded"] is True
        assert summary["rag_enabled"] is False
        assert summary["financial_enabled"] is True
        assert summary["financial_article_count"] == 2
        assert "Financial News" in summary["sources_used"][0]
    
    def test_combined_sources(self):
        """Both sources present."""
        articles = [{"title": "News", "source": "Source"}]
        summary = get_grounding_summary(
            rag_documents="Paper content chunk",
            financial_articles=articles
        )
        
        assert summary["is_grounded"] is True
        assert summary["rag_enabled"] is True
        assert summary["financial_enabled"] is True
        assert len(summary["sources_used"]) == 2


class TestGroundingIntegration:
    """Integration tests for the grounding behavior matrix."""
    
    def test_case_1_no_rag_no_financial(self):
        """Case 1: Neither RAG nor Financial enabled."""
        context = assemble_grounding_context(
            rag_context="",
            financial_context=""
        )
        summary = get_grounding_summary(
            rag_documents="",
            financial_articles=None
        )
        
        assert context == ""
        assert summary["is_grounded"] is False
        # Pure LLM generation expected
    
    def test_case_2_rag_only(self):
        """Case 2: Only RAG enabled."""
        rag_docs = "Content from arXiv paper about neural networks"
        
        context = assemble_grounding_context(
            rag_context=rag_docs,
            financial_context=""
        )
        summary = get_grounding_summary(
            rag_documents=rag_docs,
            financial_articles=None
        )
        
        assert "SCIENTIFIC CONTEXT" in context
        assert summary["is_grounded"] is True
        assert summary["rag_enabled"] is True
        assert summary["financial_enabled"] is False
    
    def test_case_3_financial_only(self):
        """Case 3: Only Financial enabled."""
        financial_ctx = "📈 FINANCIAL MARKET CONTEXT\n1. Tesla news..."
        articles = [{"title": "Tesla Q4", "source": "Bloomberg"}]
        
        context = assemble_grounding_context(
            rag_context="",
            financial_context=financial_ctx
        )
        summary = get_grounding_summary(
            rag_documents="",
            financial_articles=articles
        )
        
        assert "FINANCIAL" in context
        assert "SCIENTIFIC" not in context
        assert summary["is_grounded"] is True
        assert summary["rag_enabled"] is False
        assert summary["financial_enabled"] is True
    
    def test_case_4_both_enabled(self):
        """Case 4: Both RAG and Financial enabled."""
        rag_docs = "Scientific paper content"
        financial_ctx = "📈 Market news content"
        articles = [{"title": "News", "source": "Source"}]
        
        context = assemble_grounding_context(
            rag_context=rag_docs,
            financial_context=financial_ctx
        )
        summary = get_grounding_summary(
            rag_documents=rag_docs,
            financial_articles=articles
        )
        
        # Both contexts should be present
        assert "SCIENTIFIC CONTEXT" in context
        assert "FINANCIAL" in context or "Market" in context
        
        # Summary should reflect both
        assert summary["is_grounded"] is True
        assert summary["rag_enabled"] is True
        assert summary["financial_enabled"] is True
        assert len(summary["sources_used"]) == 2
