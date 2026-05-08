import logging
import os
from pathlib import Path
from typing import Callable, Dict, List, Type

from langchain_chroma import Chroma
from langchain_community.document_loaders import (
    Docx2txtLoader,
    PyPDFLoader,
    TextLoader,
    UnstructuredMarkdownLoader,
)
from langchain_core.documents import Document
from langchain_ollama import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from shared.config import Config

# -----------------------------------------------------------------------------
# Logging Configuration
# -----------------------------------------------------------------------------
# In real systems, logging replaces print statements.
# It provides observability, debugging, and operational monitoring.
#
# INFO  -> normal operational messages
# WARNING -> recoverable issues
# ERROR -> failures requiring attention
# -----------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DocumentIndexer:
    """
    Production-style filesystem indexer for local RAG systems.

    ----------------------------------------------------------------------------
    What this class does
    ----------------------------------------------------------------------------
    1. Scans a user directory recursively
    2. Ignores system/cache/runtime folders
    3. Loads supported documents into LangChain Document objects
    4. Splits documents into semantic chunks
    5. Generates embeddings via Ollama
    6. Persists vectors into ChromaDB

    ----------------------------------------------------------------------------
    Why this matters
    ----------------------------------------------------------------------------
    LLMs are not databases.

    RAG (Retrieval-Augmented Generation) solves this by:
        user query
            ↓
        embed query
            ↓
        similarity search
            ↓
        retrieve relevant chunks
            ↓
        inject chunks into LLM prompt

    This class is responsible for building the searchable vector database.
    """

    # -------------------------------------------------------------------------
    # Specialized parsers for structured file formats.
    #
    # Example:
    #   PDF -> extracts pages
    #   DOCX -> extracts paragraphs
    #   Markdown -> preserves markdown semantics
    # -------------------------------------------------------------------------
    LOADER_MAPPING: Dict[str, Type] = {
        ".pdf": PyPDFLoader,
        ".txt": TextLoader,
        ".md": UnstructuredMarkdownLoader,
        ".docx": Docx2txtLoader,
    }

    # -------------------------------------------------------------------------
    # Source code and config files can still contain valuable knowledge.
    #
    # These are loaded as plain text.
    # -------------------------------------------------------------------------
    SUPPORTED_CODE_EXTENSIONS = {
        ".py",
        ".java",
        ".js",
        ".ts",
        ".yaml",
        ".yml",
        ".xml",
        ".sql",
        ".sh",
    }

    # -------------------------------------------------------------------------
    # Directories that should NEVER be indexed.
    #
    # Why skip them?
    #
    # - huge
    # - low semantic value
    # - system noise
    # - package caches
    # - generated artifacts
    # -------------------------------------------------------------------------
    SKIP_DIRS = {
        ".cache",
        ".local",
        "sportradar",
        ".config",
        ".npm",
        ".cargo",
        ".rustup",
        ".codeium",
        ".conda",
        "miniconda3",
        "anaconda3",
        "snap",
        ".git",
        "node_modules",
        "__pycache__",
        ".venv",
        "venv",
        "env",
        ".idea",
        ".vscode",
        ".m2",
        ".gradle",
        "target",
        "build",
        "dist",
        "R",
    }

    # -------------------------------------------------------------------------
    # Safety valve.
    #
    # Large files can:
    # - explode memory
    # - slow indexing massively
    # - usually aren't useful in RAG
    # -------------------------------------------------------------------------
    MAX_FILE_SIZE_MB = 10

    def __init__(self):
        """
        Initializes embedding engine + text splitter.
        """

        # ---------------------------------------------------------------------
        # Embeddings convert text into vectors.
        #
        # Similar meaning -> vectors near each other.
        #
        # Example:
        # "passport number"
        # "ID document"
        #
        # become close in vector space.
        # ---------------------------------------------------------------------
        self.embeddings = OllamaEmbeddings(
            model=Config.EMBED_MODEL,
            base_url=Config.OLLAMA_BASE_URL,
        )

        # ---------------------------------------------------------------------
        # Split long documents into smaller chunks.
        #
        # Why?
        #
        # LLM context windows are finite.
        #
        # Chunk overlap preserves continuity across boundaries.
        # ---------------------------------------------------------------------
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=Config.CHUNK_SIZE,
            chunk_overlap=Config.CHUNK_OVERLAP,
        )

    def _is_supported_file(self, file_path: Path) -> bool:
        """
        Determines whether a file should be indexed.
        """
        suffix = file_path.suffix.lower()

        return (
                suffix in self.LOADER_MAPPING
                or suffix in self.SUPPORTED_CODE_EXTENSIONS
        )

    def _is_reasonable_size(self, file_path: Path) -> bool:
        """
        Reject very large files.

        Prevents:
        - memory explosions
        - accidental binary ingestion
        """
        try:
            size_mb = file_path.stat().st_size / (1024 * 1024)
            return size_mb <= self.MAX_FILE_SIZE_MB
        except Exception:
            return False

    def _load_single_file(self, file_path: Path) -> List[Document]:
        """
        Loads one file into LangChain Document objects.

        Strategy:
        - known structured formats -> specialized parser
        - code/text -> plain text loader
        """
        ext = file_path.suffix.lower()

        try:
            if ext in self.LOADER_MAPPING:
                loader = self.LOADER_MAPPING[ext](str(file_path))
            else:
                loader = TextLoader(
                    str(file_path),
                    encoding="utf-8",
                )

            docs = loader.load()

            # Add source metadata for citations in UI
            for doc in docs:
                doc.metadata["source"] = str(file_path)

            return docs

        except Exception as exc:
            logger.warning(f"Failed parsing {file_path}: {exc}")
            return []

    def _discover_files(self, source_dir: Path) -> List[Path]:
        """
        Recursively discovers files while pruning bad directories.

        Why os.walk instead of Path.rglob?

        os.walk allows directory pruning:
            dirs[:] = filtered_dirs

        This is MUCH faster for huge trees.
        """
        files_to_process: List[Path] = []

        for root, dirs, files in os.walk(source_dir):

            # prune unwanted directories in-place
            dirs[:] = [d for d in dirs if d not in self.SKIP_DIRS]

            root_path = Path(root)

            for filename in files:
                # Skip hidden files (.DS_Store, etc.)
                if filename.startswith("."):
                    continue

                file_path = root_path / filename

                if not self._is_supported_file(file_path):
                    continue

                if not self._is_reasonable_size(file_path):
                    continue

                files_to_process.append(file_path)

        return sorted(files_to_process)

    def run_sync(self, progress_fn: Callable[[float, str], None]):
        """
        Full ETL pipeline.

        Extract -> Transform -> Load
        """

        progress_fn(0.05, "Initializing filesystem scan...")

        source_dir = Path(Config.DOCUMENT_SOURCE_PATH).resolve()

        # ---------------------------------------------------------------------
        # Validation
        # ---------------------------------------------------------------------
        if not source_dir.exists():
            raise FileNotFoundError(
                f"Source path does not exist: {source_dir}"
            )

        if not os.access(source_dir, os.R_OK):
            raise PermissionError(
                f"Read permission denied: {source_dir}"
            )

        # ---------------------------------------------------------------------
        # Discover files
        # ---------------------------------------------------------------------
        files_to_process = self._discover_files(source_dir)

        if not files_to_process:
            raise FileNotFoundError(
                f"No supported files found in {source_dir}"
            )

        logger.info(f"Discovered {len(files_to_process)} files.")

        # ---------------------------------------------------------------------
        # Parse files
        # ---------------------------------------------------------------------
        all_documents: List[Document] = []
        total_files = len(files_to_process)

        for idx, file_path in enumerate(files_to_process):
            pct = 0.10 + (0.50 * (idx / total_files))

            progress_fn(
                pct,
                f"Indexing ({idx + 1}/{total_files}): {file_path.name}",
            )

            docs = self._load_single_file(file_path)
            all_documents.extend(docs)

        if not all_documents:
            raise ValueError(
                "No documents could be parsed successfully."
            )

        # ---------------------------------------------------------------------
        # Chunking
        # ---------------------------------------------------------------------
        progress_fn(0.70, "Chunking documents...")
        chunks = self.splitter.split_documents(all_documents)

        logger.info(f"Generated {len(chunks)} chunks.")

        # ---------------------------------------------------------------------
        # Vector DB initialization
        #
        # Modern Chroma auto-persists.
        # No db.persist() required.
        # ---------------------------------------------------------------------
        progress_fn(0.85, "Generating embeddings and storing vectors...")

        db = Chroma(
            persist_directory=Config.VECTOR_DB_PATH,
            embedding_function=self.embeddings,
        )

        # ---------------------------------------------------------------------
        # Batch inserts
        #
        # Important for:
        # - memory stability
        # - large corpora
        # - long-running jobs
        # ---------------------------------------------------------------------
        BATCH_SIZE = 500

        for i in range(0, len(chunks), BATCH_SIZE):
            batch = chunks[i:i + BATCH_SIZE]

            logger.info(
                f"Writing batch {i // BATCH_SIZE + 1} "
                f"({len(batch)} chunks)"
            )

            db.add_documents(batch)

        # ---------------------------------------------------------------------
        # IMPORTANT:
        #
        # No db.persist()
        #
        # Chroma persists automatically when persist_directory is set.
        # ---------------------------------------------------------------------

        progress_fn(
            1.0,
            f"Sync complete: indexed {total_files} files "
            f"into {Config.VECTOR_DB_PATH}",
        )

        logger.info("Vector store synchronization complete.")
