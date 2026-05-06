import os
from typing import Callable, Dict, Type

from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    UnstructuredMarkdownLoader,
    Docx2txtLoader,
    UnstructuredHTMLLoader
)
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_ollama import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from shared.config import Config


class DocumentIndexer:
    """
    The RAG Principle (Retrieval-Augmented Generation) - RAG addresses the "knowledge cutoff" and hallucination issues
    of LLMs. By retrieving relevant document snippets before generation, we shift the LLM's role from a Knowledge Store
    to a Reasoning Engine. It no longer needs to know the ID number; it just needs to be able to read the provided context.

    Vector Database (ChromaDB) - Unlike relational databases (SQL) that match keywords, a Vector DB matches intent.

    Embeddings - Text is converted into a high-dimensional vector. Identical meanings (e.g., "National ID" and "Personal
    Identification Number") land near each other in vector space.Similarity

    Search: When you query, the DB calculates the "Cosine Similarity" to find the text chunks closest to your question.
    """

    # Mapping registry for polymorphic loading
    LOADER_MAPPING: Dict[str, Type] = {
        ".pdf": PyPDFLoader,
        ".txt": TextLoader,
        ".md": UnstructuredMarkdownLoader,
        ".docx": Docx2txtLoader,
        ".html": UnstructuredHTMLLoader,
    }

    def __init__(self):
        self.embeddings = OllamaEmbeddings(
            model=Config.EMBED_MODEL,
            base_url=Config.OLLAMA_BASE_URL
        )
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=Config.CHUNK_SIZE,
            chunk_overlap=Config.CHUNK_OVERLAP
        )

    def _load_single_file(self, file_path: str) -> list[Document]:
        """Selects the appropriate loader based on file extension."""
        ext = os.path.splitext(file_path)[1].lower()
        if ext in self.LOADER_MAPPING:
            loader_class = self.LOADER_MAPPING[ext]
            # Some loaders require specific args; here we handle standard instantiation
            loader = loader_class(file_path)
            return loader.load()
        return []

    def run_sync(self, progress_fn: Callable[[float, str], None]):
        """
        Universal ETL Pipeline:
        Supports PDF, TXT, MD, DOCX, and HTML.
        """
        progress_fn(0.1, "Initializing filesystem scan...")

        all_documents = []
        source_dir = Config.DOCUMENT_SOURCE_PATH

        # Walk through the directory (Recursion supported)
        files_to_process = []
        for root, _, files in os.walk(source_dir):
            for file in files:
                ext = os.path.splitext(file)[1].lower()
                if ext in self.LOADER_MAPPING:
                    files_to_process.append(os.path.join(root, file))

        if not files_to_process:
            raise FileNotFoundError(f"No supported documents found in {source_dir}")

        # Process with Progress Feedback
        total_files = len(files_to_process)
        for i, file_path in enumerate(files_to_process):
            progress_pct = 0.2 + (0.4 * (i / total_files))
            progress_fn(progress_pct, f"Parsing: {os.path.basename(file_path)}")

            try:
                all_documents.extend(self._load_single_file(file_path))
            except Exception as e:
                print(f"Error loading {file_path}: {e}")  # Log and continue

        # Transform
        progress_fn(0.7, f"Chunking {len(all_documents)} text segments...")
        chunks = self.splitter.split_documents(all_documents)

        # Load
        progress_fn(0.85, "Generating embeddings (GPU Sync)...")
        Chroma.from_documents(
            documents=chunks,
            embedding=self.embeddings,
            persist_directory=Config.VECTOR_DB_PATH
        )

        progress_fn(1.0, f"Sync complete. Indexed {total_files} files.")
