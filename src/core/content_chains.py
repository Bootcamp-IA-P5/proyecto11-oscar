from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from src.models.llm_factory import get_llm_ollama, get_llm_groq
from config.prompts import (
    BLOG_GENERATION_TEMPLATE,
    TWITTER_ADAPTOR_TEMPLATE,
    IMAGE_PROMPT_GENERATION_TEMPLATE
)

def create_blog_chain_groq(model_name: str = "mistral"):
    """
    Creates a blog generation chain using an Ollama model.

    Args:
        model_name (str, optional): The name of the Ollama model to use. 
                                    Defaults to "mistral".

    Returns:
        A chain of transformations for blog generation.
    """
    llm = get_llm_ollama(model_name)

    prompt = ChatPromptTemplate.from_template(BLOG_GENERATION_TEMPLATE)

    chain = (
        prompt
        | llm
        | StrOutputParser()
    )
    
    return chain

def create_blog_chain_groq():
    llm = get_llm_groq() 
    
    prompt = ChatPromptTemplate.from_template(BLOG_GENERATION_TEMPLATE)

    chain = (
        prompt
        | llm
        | StrOutputParser()
    )
    
    return chain

def create_twitter_adaptor_chain():
    llm = get_llm_groq()
    
    prompt = ChatPromptTemplate.from_template(TWITTER_ADAPTOR_TEMPLATE)

    chain = (
        prompt
        | llm
        | StrOutputParser()
    )
    
    return chain

def create_image_prompt_chain():
    llm = get_llm_groq()
    
    prompt = ChatPromptTemplate.from_template(IMAGE_PROMPT_GENERATION_TEMPLATE)

    chain = (
        prompt
        | llm
        | StrOutputParser()
    )
    
    return chain