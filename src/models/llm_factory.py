"""
This module provides a factory for creating instances of language models.

It supports creating models from different providers such as Ollama (local),
Groq (cloud), and Google Gemini (cloud). The main function `get_llm` allows
selecting the desired language model provider based on a string identifier.
"""
from langchain_ollama import OllamaLLM
from langchain_groq import ChatGroq
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.language_models import BaseChatModel

from src.core.logger.logger import Logger
from src.core.logger.log_setup import log_setup
from config.settings import GROQ_API_KEY, GROQ_MODEL, GEMINI_API_KEY, GEMINI_MODEL

log_setup()
log = Logger().log

def get_llm_ollama(model_name: str = "mistral") -> BaseChatModel:
    """
    Initializes and returns a local Ollama language model.

    Args:
        model_name (str, optional): The name of the Ollama model to use.
                                    Defaults to "mistral".

    Returns:
        An instance of the Ollama language model.

    Raises:
        Exception: If there is an error initializing the language model.
    """
    try:
        # Note: By default Ollama can be found in http://localhost:11434
        llm = OllamaLLM(model=model_name)
        log.info(f"✅ LLM local '{model_name}' successfully initialized.")
        return llm
    except Exception as e:
        log.error(f"❌ Error initializing LLM local '{model_name}': {e}")
        raise e
    
def get_llm_groq() -> BaseChatModel:
    """
    Initializes and returns a Groq language model.

    This function retrieves the Groq API key and model name from the application
    settings. It raises a `ValueError` if the API key is not configured.

    Returns:
        An instance of the `ChatGroq` language model.

    Raises:
        ValueError: If the `GROQ_API_KEY` is not set in the environment.
    """
    try:
        api_key = GROQ_API_KEY
        model_name = GROQ_MODEL
        
        if not api_key:
            raise ValueError("Groq API Key is not set. Please set the GROQ_API_KEY environment variable.")
        
        llm = ChatGroq(
            api_key=api_key,
            model_name=model_name
        )
        log.info(f"✅ LLM '{model_name}' (Groq) successfully initialized.")
        return llm
        
    except Exception as e:
        log.error(f"⚠️ Error initializing LLM basado en API '{model_name}' (Groq): {e}.")
        
def get_llm_gemini() -> BaseChatModel:
    """
    Initializes and returns a Google Gemini language model.

    This function retrieves the Gemini API key and model name from the application
    settings. It raises a `ValueError` if the API key is not configured.

    Returns:
        An instance of the `ChatGoogleGenerativeAI` language model.

    Raises:
        ValueError: If the `GEMINI_API_KEY` is not set in the environment.
        Exception: If there is an error during the language model initialization.
    """
    api_key = GEMINI_API_KEY
    model_name = GEMINI_MODEL

    if not api_key:
        raise ValueError("Gemini API Key is not set. Please set the GEMINI_API_KEY environment variable.")
    
    try:
        llm = ChatGoogleGenerativeAI(
            model=model_name,
            api_key=api_key,
        )
        log.info(f"✅ LLM  '{model_name}' (Gemini) successfully initialized.")
        return llm
    except Exception as e:
        log.error(f"❌ Error initializing LLM '{model_name}' (Gemini): {e}")
        raise e
    
def get_llm(llm_choice: str) -> BaseChatModel:
    """
    Factory function to get a language model instance based on the provider choice.

    Args:
        llm_choice (str): The desired language model provider.
                          Supported values are "Gemini", "Groq", "Ollama".

    Returns:
        An instance of the selected language model.

    Raises:
        RuntimeError: If the selected language model fails to initialize.
        ValueError: If the `llm_choice` is not a recognized provider.
    """
    if llm_choice == "Gemini":
        try:
            return get_llm_gemini()
        except Exception as e:
            raise RuntimeError(f"Error initialing Gemini. Revise settings.py. Error: {e}")

    elif llm_choice == "Groq":
        try:
            return get_llm_groq()
        except Exception as e:
            raise RuntimeError(f"Error initialing Groq. Revise settings.py. Error: {e}")

    elif llm_choice == "Ollama":
        try:
            return get_llm_ollama()
        except Exception as e:
            raise RuntimeError(f"Error initialing Ollama. Revise settings.py. Error: {e}")
            
    else:
        raise ValueError(f"LLM option not recognized: {llm_choice}")