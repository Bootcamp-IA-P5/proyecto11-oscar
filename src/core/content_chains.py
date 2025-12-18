from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.language_models import BaseChatModel
from src.models.llm_factory import get_llm_ollama, get_llm_groq, get_llm_gemini, get_llm
from config.prompts import (
    BLOG_GENERATION_TEMPLATE,
    TWITTER_ADAPTOR_TEMPLATE,
    INSTAGRAM_ADAPTOR_TEMPLATE,
    LINKEDIN_ADAPTOR_TEMPLATE,
    IMAGE_PROMPT_GENERATION_TEMPLATE,
    SCIENCE_DIVULGATION_TEMPLATE
)

def create_chain(llm: BaseChatModel, template: str):
    """
    Creates a generic content generation chain.

    Args:
        llm (BaseChatModel): The language model to use.
        template (str): The prompt template for the chain.

    Returns:
        A chain of transformations for content generation.
    """
    prompt = ChatPromptTemplate.from_template(template)
    return prompt | llm | StrOutputParser()



def create_blog_chain(llm_choice: str):
    """
    Creates a blog generation chain using the specified LLM.

    Args:
        llm_choice (str):   The choice of the language model provider 
                            (e.g., "Gemini", "Groq", "Ollama").

    Returns:
        A chain of transformations for blog generation.
    """
    llm = get_llm(llm_choice)
    return create_chain(llm, BLOG_GENERATION_TEMPLATE)

def create_blog_chain_ollama(model_name: str = "mistral"):
    """
    Creates a blog generation chain using an Ollama model.

    Args:
        model_name (str, optional): The name of the Ollama model to use. 
                                    Defaults to "mistral".

    Returns:
        A chain of transformations for blog generation.
    """
    llm = get_llm_ollama(model_name)
    return create_chain(llm, BLOG_GENERATION_TEMPLATE)

def create_blog_chain_groq():
    """
    Creates a blog generation chain using a Groq model.

    Returns:
        A chain of transformations for blog generation.
    """
    llm = get_llm_groq() 
    return create_chain(llm, BLOG_GENERATION_TEMPLATE)

def create_blog_chain_gemini():
    """
    Creates a blog generation chain using a Gemini model.

    Returns:
        A chain of transformations for blog generation.
    """
    llm = get_llm_gemini() 
    return create_chain(llm, BLOG_GENERATION_TEMPLATE)

def create_twitter_adaptor_chain(llm_choice: str):
    """
    Creates a chain to adapt content for Twitter using the specified LLM.

    Args:
        llm_choice (str):   The choice of the language model provider 
                            (e.g., "Gemini", "Groq", "Ollama").

    Returns:
        A chain of transformations for adapting content to Twitter format.
    """
    llm = get_llm(llm_choice)
    return create_chain(llm, TWITTER_ADAPTOR_TEMPLATE)

def create_instagram_adaptor_chain(llm_choice: str):
    """
    Creates a chain to adapt content for Instagram using the specified LLM.

    Args:
        llm_choice (str):   The choice of the language model provider 
                            (e.g., "Gemini", "Groq", "Ollama").

    Returns:
        A chain of transformations for adapting content to Instagram format.
    """
    llm = get_llm(llm_choice)
    return create_chain(llm, INSTAGRAM_ADAPTOR_TEMPLATE)

def create_linkedin_adaptor_chain(llm_choice: str):
    """
    Creates a chain to adapt content for LinkedIn using the specified LLM.

    Args:
        llm_choice (str):   The choice of the language model provider 
                            (e.g., "Gemini", "Groq", "Ollama").

    Returns:
        A chain of transformations for adapting content to LinkedIn format.
    """
    llm = get_llm(llm_choice)
    return create_chain(llm, LINKEDIN_ADAPTOR_TEMPLATE)

def create_image_prompt_chain(llm_choice: str):
    """
    Creates a chain to generate an image prompt using the specified LLM.

    Args:
        llm_choice (str):   The choice of the language model provider 
                            (e.g., "Gemini", "Groq", "Ollama").

    Returns:
        A chain of transformations for generating an image prompt.
    """
    llm = get_llm(llm_choice)
    return create_chain(llm, IMAGE_PROMPT_GENERATION_TEMPLATE)

def generate_science_post_chain(llm_choice: str):
    llm = get_llm(llm_choice)
    return create_chain(llm, SCIENCE_DIVULGATION_TEMPLATE)