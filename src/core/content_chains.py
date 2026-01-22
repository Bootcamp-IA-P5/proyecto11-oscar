from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.language_models import BaseChatModel
from typing import Optional
from src.models.llm_factory import get_llm_ollama, get_llm_groq, get_llm_gemini, get_llm
from src.models.financial_news import load_financial_news, build_finance_context
from src.core.logger.logger import Logger
from src.core.logger.log_setup import log_setup
from config.prompts import (
    BLOG_GENERATION_TEMPLATE,
    TWITTER_ADAPTOR_TEMPLATE,
    INSTAGRAM_ADAPTOR_TEMPLATE,
    LINKEDIN_ADAPTOR_TEMPLATE,
    IMAGE_PROMPT_GENERATION_TEMPLATE,
    SCIENCE_DIVULGATION_TEMPLATE
)

log_setup()
log = Logger().log

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


def _prepare_financial_context(
    use_finance: bool = False,
    finance_query: Optional[str] = None,
    finance_max_articles: int = 5
) -> tuple[str, list]:
    """
    Prepare financial news context if enabled.
    
    Args:
        use_finance: Whether to include financial news
        finance_query: Query for financial news (e.g., "Tesla", "inflation")
        finance_max_articles: Maximum number of articles to fetch
    
    Returns:
        Tuple of (formatted financial context string, list of raw articles)
    """
    if not use_finance or not finance_query:
        return "", []
    
    try:
        log.info(f"Fetching financial news for: {finance_query}")
        news_articles = load_financial_news(finance_query, finance_max_articles)
        
        if not news_articles:
            log.warning("No financial news articles retrieved")
            return "", []
        
        context = build_finance_context(news_articles)
        log.info(f"Financial context prepared with {len(news_articles)} articles")
        return context, news_articles
        
    except Exception as e:
        log.error(f"Error preparing financial context: {e}")
        return "", []


def assemble_grounding_context(
    rag_context: str = "",
    financial_context: str = "",
    debug: bool = False
) -> str:
    """
    Assemble the final grounding context from multiple sources.
    
    This function combines RAG (scientific) and financial contexts
    into a single, well-labeled context block for the LLM.
    
    Args:
        rag_context: Scientific context from arXiv papers
        financial_context: Financial news context from Alpha Vantage
        debug: If True, logs the assembled context for validation
    
    Returns:
        Combined context string with clear source labels
    """
    context_blocks = []
    sources_used = []
    
    if rag_context:
        labeled_rag = f"""
🔬 SCIENTIFIC CONTEXT (from arXiv papers):
-------------------------------------------
{rag_context}
-------------------------------------------
"""
        context_blocks.append(labeled_rag)
        sources_used.append("arXiv RAG")
    
    if financial_context:
        # Financial context already has labels from build_finance_context
        context_blocks.append(financial_context)
        sources_used.append("Financial News")
    
    combined_context = "\n".join(context_blocks)
    
    if debug:
        log.info("=" * 60)
        log.info("DEBUG: ASSEMBLED GROUNDING CONTEXT")
        log.info(f"Sources used: {sources_used if sources_used else 'None (pure LLM)'}")
        log.info("-" * 60)
        if combined_context:
            # Log first 500 chars to avoid flooding logs
            log.info(f"Context preview:\n{combined_context[:500]}...")
        else:
            log.info("No external context - using pure LLM generation")
        log.info("=" * 60)
    
    return combined_context


def get_grounding_summary(
    rag_documents: str = "",
    financial_articles: list = None
) -> dict:
    """
    Generate a summary of grounding sources for UI transparency.
    
    Args:
        rag_documents: Raw RAG context string
        financial_articles: List of financial article dicts
    
    Returns:
        Dict with source counts and details
    """
    summary = {
        "sources_used": [],
        "rag_enabled": bool(rag_documents),
        "rag_doc_count": 0,
        "financial_enabled": bool(financial_articles),
        "financial_article_count": 0,
        "financial_articles": [],
        "is_grounded": False
    }
    
    if rag_documents:
        # Count approximate number of document chunks
        chunks = rag_documents.split("\n\n")
        summary["rag_doc_count"] = len([c for c in chunks if c.strip()])
        summary["sources_used"].append(f"arXiv ({summary['rag_doc_count']} chunks)")
    
    if financial_articles:
        summary["financial_article_count"] = len(financial_articles)
        summary["financial_articles"] = [
            {"title": a.get("title", ""), "source": a.get("source", "")}
            for a in financial_articles[:5]  # Limit for UI
        ]
        summary["sources_used"].append(f"Financial News ({summary['financial_article_count']} articles)")
    
    summary["is_grounded"] = summary["rag_enabled"] or summary["financial_enabled"]
    
    return summary



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