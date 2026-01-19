import os
import pysqlite3
import sys
sys.modules["sqlite3"] = sys.modules.pop("pysqlite3") # Force the use of sqlite binary.

import arxiv
import time
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
import shutil

from src.core.logger.log_setup import log_setup
from src.core.logger.logger import Logger

log_setup()

class ScienceRAG:
    """
    A Retrieval-Augmented Generation (RAG) engine focused on scientific papers from arXiv.

    This class handles the ingestion of academic papers, creation of a vector store,
    and retrieval of relevant context based on a user's query. It uses ChromaDB
    for vector storage and HuggingFace embeddings.
    """
    def __init__(self, topic="Artificial Intelligence"):
        """
        Initializes the ScienceRAG engine.

        Args:
            topic (str, optional):  A default topic, currently not used but can be
                                    leveraged for pre-seeding the database.
                                    Defaults to "Artificial Intelligence".
        """
        self.topic = topic
        
        self.log = Logger().log
        self.embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        self.persist_directory = "./chroma_db"
        self.vector_store = None

    def ingest_papers(self, query: str, max_results: int = 7) -> str:
        """
        Searches for papers on arXiv, processes them, and ingests them into the vector store.

        Args:
            query (str): The search query for arXiv.
            max_results (int, optional): The maximum number of papers to download. Defaults to 7.

        Returns:
            str: A status message indicating the result of the ingestion.
        """
        try:
            self.log.debug(f"Searching for papers related to: {query}...")
            
            # Add delay to avoid rate limiting (429 errors)
            time.sleep(3)
            
            # Use arxiv client directly instead of ArxivLoader to avoid PyMuPDF issues
            search = arxiv.Search(
                query=query,
                max_results=max_results,
                sort_by=arxiv.SortCriterion.Relevance
            )
            
            documents = []
            for result in search.results():
                # Create document from metadata and summary (no PDF processing)
                content = f"Title: {result.title}\n\n"
                content += f"Authors: {', '.join([author.name for author in result.authors])}\n\n"
                content += f"Published: {result.published.strftime('%Y-%m-%d')}\n\n"
                content += f"Summary: {result.summary}\n\n"
                
                metadata = {
                    'Title': result.title,
                    'entry_id': result.entry_id,
                    'Published': result.published.strftime('%Y-%m-%d'),
                    'Authors': ', '.join([author.name for author in result.authors])
                }
                
                doc = Document(page_content=content, metadata=metadata)
                documents.append(doc)
            
            if not documents:
                self.log.warning("No papers found for the given query.")
                return "No se han encontrado papers para la búsqueda."
            
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1400, 
                chunk_overlap=175
            )
            splits = text_splitter.split_documents(documents)
            
            self.vector_store = Chroma.from_documents(
                documents=splits, 
                embedding=self.embeddings,
                persist_directory=self.persist_directory
            )
            self.log.debug("Papers indexed successfully.")
            return f"{len(documents)} paper(s) sobre '{query}' han sido indexados."
        except Exception as e:
            self.log.error(f"Error while indexing papers: {e}")
            return f"Error al indexar: {e}"


    def get_context(self, user_query: str) -> tuple[str, list[dict]]:
        """
        Retrieves relevant context from the vector store for a given query.

        Args:
            user_query (str): The user's query to find relevant document chunks.

        Returns:
            tuple[str, list[dict]]: A tuple containing:
                - str: A string containing the concatenated page content of relevant documents.
                - list[dict]: A list of source metadata dictionaries with 'title' and 'url' keys.
        """
        if not self.vector_store:
            self.vector_store = Chroma(persist_directory=self.persist_directory, embedding_function=self.embeddings)
            
        retriever = self.vector_store.as_retriever(
            search_type="similarity",
            search_kwargs={
                "k": 5
            }
        )
        
        try:
            relevant_docs = retriever.invoke(user_query)
        except Exception as e:
            self.log.warning(f"No relevant documents found for query: {user_query}")
            return "", []
        
        # Extract unique sources
        sources = []
        seen_ids = set()
        for doc in relevant_docs:
            metadata = doc.metadata
            title = metadata.get('Title', 'Unknown')
            entry_id = metadata.get('entry_id', '')
            authors = metadata.get('Authors', 'Unknown')
            published = metadata.get('Published', 'Unknown')
            
            if entry_id and entry_id not in seen_ids:
                sources.append({
                    'title': title,
                    'url': entry_id,
                    'authors': authors,
                    'published': published
                })
                seen_ids.add(entry_id)
        
        context = "\n\n".join([doc.page_content for doc in relevant_docs])
        return context, sources
    
    def list_indexed_papers(self) -> list[str]:
        """
        Lists the unique titles of all papers currently indexed in the vector store.

        Returns:
            list[str]: A list of unique paper titles.
        """
        if not os.path.exists(self.persist_directory):
            self.vector_store = None
            return []
        try:
            if not self.vector_store:
                self.vector_store = Chroma(
                    persist_directory=self.persist_directory, 
                    embedding_function=self.embeddings
                )
            
            data = self.vector_store.get()
            metadatos = data['metadatas']
            
            if not metadatos:
                return []

            
            titles = set()
            for m in metadatos:
                title = m.get('Title') or m.get('title') or "Unknown Title"
                titles.add(title)
                
            return list(titles)
        except Exception as e:
            self.log.error(f"Error while listing indexed papers: {e}")
            return []
    
    def reset_database(self) -> None:
        """
        Deletes the entire vector store from disk and resets the in-memory state.
        """
        try:
            if self.vector_store is not None:
                self.vector_store = None
            
            if os.path.exists(self.persist_directory):
                shutil.rmtree(self.persist_directory)
                self.log.debug(f"Directory {self.persist_directory} deleted successfully.")
            
            self.vector_store = None
        except Exception as e:
            self.log.error(f"Error while resetting database: {e}")