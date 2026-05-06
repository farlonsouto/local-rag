FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
# - build-essential: for compiling C-based python extensions
# - curl: for healthchecks
# - libmagic-dev: required by 'unstructured' for file type identification
# - poppler-utils/tesseract-ocr: for advanced PDF/Image parsing if needed later
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    sudo \
    libmagic-dev \
    poppler-utils \
    tesseract-ocr \
    && rm -rf /var/lib/apt/lists/*

# Install Python requirements
# We include 'unstructured', 'docx2txt', and 'markdown' for Module 1
RUN pip install --no-cache-dir \
    langchain \
    langchain-ollama \
    langchain-community \
    chromadb \
    pypdf \
    streamlit \
    docx2txt \
    unstructured[md,html] \
    python-magic

# Copy the entire src tree into /app
COPY ./src /app

# Add /app to PYTHONPATH so modules can find each other
ENV PYTHONPATH=/app

EXPOSE 8501
EXPOSE 8502

# Default to the query app
CMD ["streamlit", "run", "app/main.py", "--server.port=8501", "--server.address=0.0.0.0"]
