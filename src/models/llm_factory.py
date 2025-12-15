from langchain_ollama import OllamaLLM
from langchain_groq import ChatGroq
from langchain_core.language_models import BaseChatModel

from src.core.logger.logger import Logger
from src.core.logger.log_setup import log_setup
from config.settings import GROQ_API_KEY, GROQ_MODEL

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
    Initializes and returns a GROQ language model.

    Args:
        model_name (str, optional): The name of the GROQ model to use.
                                    Defaults to the value of the GROQ_MODEL environment variable.

    Returns:
        An instance of the GROQ language
    """
    try:
        api_key = GROQ_API_KEY
        model_name = GROQ_MODEL
        
        if not api_key:
            raise ValueError("Groq API Key is not set. Please set the GROQ_API_KEY environment variable")
        
        llm = ChatGroq(
            api_key=api_key,
            model_name=model_name
        )
        log.info(f"✅ LLM basado en API '{model_name}' (Groq) successfully initialized.")
        return llm
        
    except Exception as e:
        log.error(f"⚠️ Error initializing LLM basado en API '{model_name}' (Groq): {e}.")