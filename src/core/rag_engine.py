import os
import pysqlite3
import sys
sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")  # Force the use of sqlite binary.

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
            topic (str, optional): A default topic, currently not used but can be
                                   leveraged for pre-seeding the database.
                                   Defaults to "Artificial Intelligence".
        """
        self.topic = topic

        self.log = Logger().log
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
        )

        self.persist_directory = "./chroma_db"
        self.collection_name = "science_rag"
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

            # Chunking strategy:
            # - chunk_size: size of each text fragment sent to the embedding model
            # - chunk_overlap: ensures semantic continuity between chunks
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=900,
                chunk_overlap=120
            )

            splits = text_splitter.split_documents(docs)
            cleaned_splits = filter_complex_metadata(splits)

            self.vector_store = Chroma.from_documents(
                documents=cleaned_splits,
                embedding=self.embeddings,
                persist_directory=self.persist_directory,
                collection_name=self.collection_name
            )

            self.vector_store.persist()

            self.log.debug("Papers indexed successfully.")
            return f"{len(docs)} paper(s) sobre '{query}' han sido indexados."

        except Exception as e:
            self.log.error(f"Error while indexing papers: {e}")
            return f"Error al indexar: {e}"

    def get_context(self, user_query: str, k_chunks: int = 5) -> dict:
        """
        Retrieves relevant context from the vector store for a given query.

        Args:
            user_query (str): The user's query to find relevant document chunks.

        Returns:
            dict:
                - context: concatenated text
                - sources: list of papers used with title, url, chunks used
                - coverage: approximate coverage percentage (0-100)
        """

        try:
            if not self.vector_store:
                if not os.path.exists(self.persist_directory):
                    return {"context": "", "sources": [], "coverage": 0}

                self.vector_store = Chroma(
                    persist_directory=self.persist_directory,
                    embedding_function=self.embeddings,
                    collection_name=self.collection_name
                )

            # MMR retrieval:
            # - k: number of chunks retrieved
            # - lambda_mult: balance between relevance and diversity (0.7 is a good tradeoff)
            retriever = self.vector_store.as_retriever(
                search_type="mmr",
                search_kwargs={
                    "k": k_chunks,
                    "lambda_mult": 0.7
                }
            )

            relevant_docs = retriever.invoke(user_query)

            if not relevant_docs:
                return {"context": "", "sources": [], "coverage": 0}

            # Build clean context
            context = "\n\n".join(doc.page_content for doc in relevant_docs)

            # Build structured sources (for UI evidence)
            sources = {}
            for doc in relevant_docs:
                meta = doc.metadata
                title = meta.get("Title") or meta.get("title") or "Unknown title"
                entry_id = meta.get("entry_id", "")
                url = entry_id if entry_id.startswith("http") else None

                if title not in sources:
                    sources[title] = {
                        "title": title,
                        "url": url,
                        "chunks": 1
                    }
                else:
                    sources[title]["chunks"] += 1

            # Simple heuristic for coverage (demo-friendly metric)
            coverage = min(100, int((len(relevant_docs) / 5) * 100))

            return {
                "context": context,
                "sources": list(sources.values()),
                "coverage": coverage
            }

        except Exception as e:
            self.log.error(f"Error retrieving context: {e}")
            return {"context": "", "sources": [], "coverage": 0}

    def list_indexed_papers(self) -> list[str]:
        """
        Lists the unique titles of all papers currently indexed in the vector store.

        Returns:
            list[str]: A list of unique paper titles.
        """
        if not os.path.exists(self.persist_directory):
            return []

        try:
            if not self.vector_store:
                self.vector_store = Chroma(
                    persist_directory=self.persist_directory,
                    embedding_function=self.embeddings,
                    collection_name=self.collection_name
                )

            data = self.vector_store.get()
            metadatos = data.get("metadatas", [])

            if not metadatos:
                return []

            titles = set()
            for m in metadatos:
                title = m.get("Title") or m.get("title") or "Unknown Title"
                titles.add(title)

            return sorted(titles)

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

        except PermissionError:
            self.log.error("Permission denied: The database file is likely still in use.")
        except Exception as e:
            self.log.error(f"Error while resetting database: {e}")
