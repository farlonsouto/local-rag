"""
SHARED CONFIGURATION REGISTRY
=============================

DESIGN RATIONALE:

1. DRY (Don't Repeat Yourself): Prevents divergence between the 'filewalker' (Writer)
   and 'app' (Reader). If the embedding model changes in one, it must change in the
   other to maintain mathematical vector compatibility.

2. ENVIRONMENT AGNOSTICISM: Uses os.getenv to allow the Docker-Compose orchestration
   to override settings without modifying source code.

3. TYPE SAFETY: Centralizing these as constants reduces runtime errors caused by typos
   in string-based lookups.

TECHNICAL CONTEXT:

I. Mathematical Consistency (The Vector Problem)In RAG, the Embedding Model is the
"lens" through which the system sees text. If filewalker uses one lens to write the
data and app uses a different one to read it, the search will fail 100% of the time.
By centralizing EMBED_MODEL, we ensure that the "Librarian" and the "Reader" are literally
speaking the same mathematical language.

II. Volume Mapping AlignmentYour docker-compose.yml maps host directories to container
directories. If filewalker saves the database to /tmp/db but app looks in /app/chroma_db,
the system appears broken. This configuration file acts as the Internal Contract for the
container's file system, ensuring both modules point to the exact same mount point defined
in your orchestration layer.

III. Production ReadinessBy using os.getenv, we follow the Twelve-Factor App methodology. This
allows one to swap a lightweight model (Llama 3.2) for a more powerful one (Llama 3 8B) simply by
changing a single line in your .env or docker-compose.yml, without ever touching the logic
in src/app/core.py.
"""

import os


class Config:
    # --- MODEL CONFIGURATION ---
    # The 'Reasoning' Model (LLM)
    # Llama 3.2 is chosen for its superior performance-to-size ratio in
    # factual extraction tasks.
    LLM_MODEL = os.getenv("LLM_MODEL", "llama3.2")

    # The 'Librarian' Model (Embeddings)
    # CRITICAL: If this changes, the Vector DB must be deleted and re-indexed.
    # Vectors created by one model cannot be read by another.
    EMBED_MODEL = os.getenv("EMBED_MODEL", "nomic-embed-text")

    # --- INFRASTRUCTURE CONFIGURATION ---
    # Connection string for the Ollama API service within the Docker network.
    OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://ollama:11434")

    # --- PERSISTENCE LAYER ---
    # Path where the ChromaDB SQLite files and Parquet vectors are stored.
    # This must be consistent across the 'filewalker' and 'app' volumes.
    VECTOR_DB_PATH = "/app/chroma_db"

    # Source directory for raw document ingestion.
    DOCUMENT_SOURCE_PATH = "/app/data"

    # --- RAG PARAMETERS ---
    # Top-K: Number of document chunks to retrieve for context.
    # Lower value (3) reduces noise; higher value increases context but costs memory.
    RETRIEVAL_K = 3

    # Chunking: Small chunks (600) with overlap (100) are optimized for
    # dense factual data like National ID numbers.
    CHUNK_SIZE = 600
    CHUNK_OVERLAP = 100
