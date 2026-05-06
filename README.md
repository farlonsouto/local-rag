## Local-first RAG Architecture
A high-performance, containerized Retrieval-Augmented Generation (RAG) system built for secure, local document intelligence. This project leverages Ollama for GPU-accelerated inference and ChromaDB for vector persistence, entirely isolated from the public cloud.
## 🏗️ Architectural Overview
The system is engineered following SOLID principles, partitioned into two distinct operational modules to decouple heavy-duty data engineering from real-time inference logic.
## 1. Module: Filewalker (Ingestor)

* Role: A polymorphic ETL pipeline.
* Engineering Pattern: Utilizes a Factory Pattern to dynamically select document loaders (PDF, DOCX, TXT, MD, HTML).
* Process: Performs semantic chunking and high-dimensional vector embedding using nomic-embed-text.
* UI: Streamlit-based maintenance console with real-time telemetry (progress bars) for indexing large datasets.

## 2. Module: Query-App (Inference)

* Role: Context-aware reasoning engine.
* Engineering Pattern: Implements a Strategy Pattern for RAG chains, injecting retrieved document context into the LLM prompt window.
* Process: Performs a semantic similarity search against the local vector store and uses Llama 3.2 for deterministic, zero-temperature fact extraction.
* UI: Clean, conversation-oriented Web UI.

------------------------------
## 🚀 Technical Stack

* Orchestration: Docker Compose (Multi-container architecture)
* LLM Engine: Ollama (GPU-accelerated via NVIDIA Container Toolkit)
* Framework: LangChain (Orchestrator), Streamlit (Frontend)
* Vector Database: ChromaDB (Local persistent storage)
* Models: Llama 3.2 (Reasoning), Nomic-Embed-Text (Vectorization)

------------------------------
## 🛠️ Project Structure

.
├── docker-compose.yml     # Service orchestration (GPU reservations & Healthchecks)
├── Dockerfile             # Multi-stage-ready environment with system dependencies
├── src/
│   ├── shared/            # SSOT (Single Source of Truth) for system-wide Config
│   ├── filewalker/        # Module 1: The "Writer" (Vector Indexer)
│   └── app/               # Module 2: The "Reader" (LLM Query Interface)
└── data/                  # Source documents (Git-ignored)

------------------------------
## 🔧 Installation & Deployment## Prerequisites

* NVIDIA Container Toolkit installed on the host OS.
* Docker & Docker Compose (v3.8+).

## Quick Start

   1. Clone & Prepare:
   
   git clone <repository-url>
   mkdir data chroma_db  # Create local volumes
   
   2. Environment Sync:
   Place your private documents in the ./data folder.
   3. Launch Ecosystem:
   
   docker-compose up --build
   
   This will automatically pull the required models and initialize the GPU-accelerated backend.

## Operational Access

* Query Interface: http://localhost:8501
* Index Maintenance: http://localhost:8502

------------------------------
## 🛡️ Privacy & Security

* 100% Local: No data ever leaves the containerized network.
* Deterministic Output: Temperature is set to 0 to prevent hallucinations in sensitive data retrieval (e.g., National ID numbers).
* Encapsulated Config: System constants are managed via src/shared/config.py, allowing for environment-based overrides without touching business logic.

------------------------------
Author: Farlon Souto
Role: Senior Software Engineer
Focus: GPU accelerated, local AI | Clean Architecture

