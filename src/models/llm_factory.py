# from langchain_community.llms import Ollama
from langchain_ollama import OllamaLLM
from langchain_core.language_models import BaseChatModel

from src.core.logger.logger import Logger
from src.core.logger.log_setup import log_setup

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
        # Nota: By default Ollama can be found in http://localhost:11434
        llm = OllamaLLM(model=model_name)
        log.info(f"✅ LLM local '{model_name}' successfully initialized.")
        return llm
    except Exception as e:
        log.error(f"❌ Error initializing LLM local '{model_name}': {e}")
        raise e