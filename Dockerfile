# Dockerfile – Endee Infinite Context RAG Demo (Python app)
FROM python:3.11-slim

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python deps first (layer-caching optimisation)
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Pre-download embedding and reranker models so they're baked into the image
RUN python -c "\
    from sentence_transformers import SentenceTransformer, CrossEncoder; \
    SentenceTransformer('all-MiniLM-L6-v2'); \
    CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')"

# Copy application code
COPY . .

# Expose Gradio port
EXPOSE 7860

# Default environment variables (override via docker-compose or -e flags)
ENV ENDEE_HOST=http://endee:8080 \
    INDEX_NAME=sourya_rag \
    LLM_PROVIDER=openai \
    EMBED_MODEL=all-MiniLM-L6-v2 \
    EMBED_DIM=384 \
    CHUNK_SIZE=200 \
    CHUNK_OVERLAP=40 \
    TOP_K=5 \
    RERANK_TOP_N=3 \
    APP_HOST=0.0.0.0 \
    APP_PORT=7860

HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD curl -f http://localhost:7860/ || exit 1

CMD ["python", "app.py"]
