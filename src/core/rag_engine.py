import os
import pysqlite3
import sys
sys.modules["sqlite3"] = sys.modules.pop("pysqlite3") # Force the use of sqlite binary.

from langchain_community.document_loaders import ArxivLoader
from langchain_community.vectorstores.utils import filter_complex_metadata
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

    def ingest_papers(self, query: str, max_results: int = 2) -> str:
        """
        Searches for papers on arXiv, processes them, and ingests them into the vector store.

        Args:
            query (str): The search query for arXiv.
            max_results (int, optional): The maximum number of papers to download. Defaults to 2.

        Returns:
            str: A status message indicating the result of the ingestion.
        """
        try:
            self.log.debug(f"Searching for papers related to: {query}...")
            loader = ArxivLoader(
                query=query, 
                load_max_docs=max_results,
                load_all_available_meta=True
            )
            docs = loader.load()
            
            if not docs:
                self.log.warning("No papers found for the given query.")
                return "No se han encontrado papers para la búsqueda."
            
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=600, 
                chunk_overlap=50
            )
            splits = text_splitter.split_documents(docs)
            cleaned_splits = filter_complex_metadata(splits)
            
            self.vector_store = Chroma.from_documents(
                documents=cleaned_splits, 
                embedding=self.embeddings,
                persist_directory=self.persist_directory
            )
            self.log.debug("Papers indexed successfully.")
            return f"{len(docs)} paper(s) sobre '{query}' han sido indexados."
        except Exception as e:
            self.log.error(f"Error while indexing papers: {e}")
            return f"Error al indexar: {e}"


    def get_context(self, user_query: str) -> str:
        """
        Retrieves relevant context from the vector store for a given query.

        Args:
            user_query (str): The user's query to find relevant document chunks.

        Returns:
            str: A string containing the concatenated page content of relevant documents.
        """
        if not self.vector_store:
            self.vector_store = Chroma(persist_directory=self.persist_directory, embedding_function=self.embeddings)
            
        retriever = self.vector_store.as_retriever(
            search_type="similarity_score_threshold",
            search_kwargs={
                "k": 3,
                "score_threshold": 0.35
            }
        )
        
        try:
            relevant_docs = retriever.invoke(user_query)
        except Exception as e:
            self.log.warning(f"No relevant documents found for query: {user_query}")
            return ""
        
        return "\n\n".join([doc.page_content for doc in relevant_docs])
    
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