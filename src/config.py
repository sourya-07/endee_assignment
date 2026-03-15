"""
config.py – Centralised configuration for Multi-Subject Notes App
All values can be overridden with environment variables.
"""

import os
from dotenv import load_dotenv

load_dotenv()  # Load .env into os.environ before reading any values

# Endee Vector Database
ENDEE_HOST: str = os.getenv("ENDEE_HOST", "http://localhost:8080")
ENDEE_AUTH_TOKEN: str = os.getenv("ENDEE_AUTH_TOKEN", "")  # empty = no auth

# Index Settings
EMBED_MODEL: str = os.getenv("EMBED_MODEL", "all-MiniLM-L6-v2")
EMBED_DIM: int = int(os.getenv("EMBED_DIM", "384"))

# Chunking Parameters
# Word-based chunk size (~200-512 tokens; 1 token ≈ 0.75 words → 200 tokens ≈ 150 words)
CHUNK_SIZE: int = int(os.getenv("CHUNK_SIZE", "200"))   # words per chunk
CHUNK_OVERLAP: int = int(os.getenv("CHUNK_OVERLAP", "40"))  # overlapping words

# Retrieval Parameters
TOP_K: int = int(os.getenv("TOP_K", "5"))               # candidates to retrieve
RERANK_TOP_N: int = int(os.getenv("RERANK_TOP_N", "3")) # after reranking

# LLM Settings
LLM_PROVIDER: str = os.getenv("LLM_PROVIDER", "openai")   # "openai" | "google" | "ollama"
OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL: str = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
GOOGLE_API_KEY: str = os.getenv("GOOGLE_API_KEY", "")
GOOGLE_MODEL: str = os.getenv("GOOGLE_MODEL", "gemini-3-flash-preview")
OLLAMA_BASE_URL: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL: str = os.getenv("OLLAMA_MODEL", "llama3")
LLM_MAX_TOKENS: int = int(os.getenv("LLM_MAX_TOKENS", "4096"))
LLM_TEMPERATURE: float = float(os.getenv("LLM_TEMPERATURE", "0.2"))

# Database File
SUBJECTS_DB: str = "subjects.json"

#  App Settings
APP_TITLE: str = "Endee Infinite Context RAG Demo"
APP_PORT: int = int(os.getenv("APP_PORT", "7860"))
APP_HOST: str = os.getenv("APP_HOST", "0.0.0.0")
DEBUG: bool = os.getenv("DEBUG", "false").lower() == "true"
