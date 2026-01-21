import pytest
import os
from unittest.mock import patch, MagicMock
from src.core.rag_engine import ScienceRAG

@pytest.fixture
def mock_rag():
    with patch("src.core.rag_engine.HuggingFaceEmbeddings") as mock_embeddings:
        rag = ScienceRAG(topic="Test Topic")
        yield rag, mock_embeddings

@patch("src.core.rag_engine.ArxivLoader")
@patch("src.core.rag_engine.Chroma")
def test_ingest_papers_success(mock_chroma, mock_arxiv, mock_rag):
    rag, _ = mock_rag
    
    # Mock arXiv returns
    mock_doc = MagicMock()
    mock_doc.page_content = "Test content"
    mock_doc.metadata = {"Title": "Test Paper"}
    mock_arxiv.return_value.load.return_value = [mock_doc]
    
    result = rag.ingest_papers("AI", max_results=1)
    
    assert "1 paper(s) sobre 'AI' han sido indexados" in result
    mock_chroma.from_documents.assert_called_once()

@patch("src.core.rag_engine.ArxivLoader")
def test_ingest_papers_no_results(mock_arxiv, mock_rag):
    rag, _ = mock_rag
    mock_arxiv.return_value.load.return_value = []
    
    result = rag.ingest_papers("Unknown Topic")
    
    assert "No se han encontrado papers" in result

@patch("src.core.rag_engine.Chroma")
def test_get_context_success(mock_chroma, mock_rag):
    rag, _ = mock_rag
    
    # Mock retriever
    mock_retriever = MagicMock()
    mock_doc = MagicMock()
    mock_doc.page_content = "Context content"
    mock_retriever.invoke.return_value = [mock_doc]
    
    # Mock vector store as retriever
    mock_instance = mock_chroma.return_value
    mock_instance.as_retriever.return_value = mock_retriever
    
    # Set rag.vector_store to mock_instance
    rag.vector_store = mock_instance
    
    context = rag.get_context("test query")
    
    assert context == "Context content"
    mock_retriever.invoke.assert_called_once_with("test query")

def test_list_indexed_papers_empty(mock_rag):
    rag, _ = mock_rag
    with patch("os.path.exists", return_value=False):
        titles = rag.list_indexed_papers()
        assert titles == []

@patch("src.core.rag_engine.Chroma")
def test_list_indexed_papers_with_data(mock_chroma, mock_rag):
    rag, _ = mock_rag
    mock_instance = mock_chroma.return_value
    mock_instance.get.return_value = {'metadatas': [{'Title': 'Paper 1'}, {'title': 'Paper 2'}]}
    
    with patch("os.path.exists", return_value=True):
        titles = rag.list_indexed_papers()
        assert set(titles) == {"Paper 1", "Paper 2"}

@patch("shutil.rmtree")
@patch("os.path.exists", return_value=True)
def test_reset_database(mock_exists, mock_rmtree, mock_rag):
    rag, _ = mock_rag
    rag.vector_store = MagicMock()
    
    rag.reset_database()
    
    mock_rmtree.assert_called_once_with(rag.persist_directory)
    assert rag.vector_store is None
