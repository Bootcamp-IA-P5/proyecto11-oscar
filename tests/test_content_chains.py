import pytest
from unittest.mock import patch, MagicMock
from langchain_core.runnables import RunnableSequence
from src.core.content_chains import (
    create_chain, 
    create_blog_chain, 
    create_twitter_adaptor_chain,
    create_image_prompt_chain
)

def test_create_chain():
    mock_llm = MagicMock()
    template = "Prompt: {topic}"
    chain = create_chain(mock_llm, template)
    
    # Check if it's a Runnable (LangChain pipe)
    assert isinstance(chain, RunnableSequence)

@patch("src.core.content_chains.get_llm")
def test_create_blog_chain(mock_get_llm):
    mock_llm = MagicMock()
    mock_get_llm.return_value = mock_llm
    
    chain = create_blog_chain("Gemini")
    
    mock_get_llm.assert_called_once_with("Gemini")
    assert isinstance(chain, RunnableSequence)

@patch("src.core.content_chains.get_llm")
def test_create_twitter_chain(mock_get_llm):
    mock_llm = MagicMock()
    mock_get_llm.return_value = mock_llm
    
    chain = create_twitter_adaptor_chain("Groq")
    
    mock_get_llm.assert_called_once_with("Groq")
    assert isinstance(chain, RunnableSequence)

@patch("src.core.content_chains.get_llm")
def test_create_image_prompt_chain(mock_get_llm):
    mock_llm = MagicMock()
    mock_get_llm.return_value = mock_llm
    
    chain = create_image_prompt_chain("Ollama")
    
    mock_get_llm.assert_called_once_with("Ollama")
    assert isinstance(chain, RunnableSequence)
