import pytest
from unittest.mock import patch, MagicMock
from src.models.llm_factory import get_llm, get_llm_gemini, get_llm_groq, get_llm_ollama

@patch("src.models.llm_factory.ChatGoogleGenerativeAI")
@patch("src.models.llm_factory.GEMINI_API_KEY", "fake_key")
@patch("src.models.llm_factory.GEMINI_MODEL", "gemini-pro")
def test_get_llm_gemini_success(mock_gemini):
    llm = get_llm_gemini()
    mock_gemini.assert_called_once()
    assert llm is not None

@patch("src.models.llm_factory.GEMINI_API_KEY", None)
def test_get_llm_gemini_missing_key():
    with pytest.raises(ValueError, match="Gemini API Key is not set"):
        get_llm_gemini()

@patch("src.models.llm_factory.ChatGroq")
@patch("src.models.llm_factory.GROQ_API_KEY", "fake_key")
@patch("src.models.llm_factory.GROQ_MODEL", "llama3-8b")
def test_get_llm_groq_success(mock_groq):
    llm = get_llm_groq()
    mock_groq.assert_called_once()
    assert llm is not None

@patch("src.models.llm_factory.GROQ_API_KEY", None)
def test_get_llm_groq_missing_key():
    with pytest.raises(ValueError, match="Groq API Key is not set"):
        get_llm_groq()

@patch("src.models.llm_factory.OllamaLLM")
def test_get_llm_ollama_success(mock_ollama):
    llm = get_llm_ollama("mistral")
    mock_ollama.assert_called_once_with(model="mistral", temperature=0.3)
    assert llm is not None

def test_get_llm_invalid_choice():
    with pytest.raises(ValueError, match="LLM option not recognized"):
        get_llm("InvalidProvider")

@patch("src.models.llm_factory.get_llm_gemini")
def test_get_llm_factory_gemini(mock_get_gemini):
    get_llm("Gemini")
    mock_get_gemini.assert_called_once()

@patch("src.models.llm_factory.get_llm_groq")
def test_get_llm_factory_groq(mock_get_groq):
    get_llm("Groq")
    mock_get_groq.assert_called_once()

@patch("src.models.llm_factory.get_llm_ollama")
def test_get_llm_factory_ollama(mock_get_ollama):
    get_llm("Ollama")
    mock_get_ollama.assert_called_once()
