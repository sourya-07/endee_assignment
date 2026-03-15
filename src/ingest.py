"""
ingest.py – Multi-Subject Notes App
Data ingestion pipeline:
  1. Fetch / read source documents
  2. Chunk into 200-512 token windows with overlap
  3. Embed with sentence-transformers (all-MiniLM-L6-v2, dim=384)
  4. Upsert into Endee vector index with cosine similarity

Usage:
  python ingest.py [--reset]
"""

import argparse
import hashlib
import logging
import time
import uuid
from typing import List, Dict, Any

import requests
from bs4 import BeautifulSoup
from sentence_transformers import SentenceTransformer
from endee import Endee, Precision

from .config import (
    ENDEE_HOST,
    ENDEE_AUTH_TOKEN,
    EMBED_MODEL,
    EMBED_DIM,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Old HARDCODED_DOCS removed for multi-subject.


def fetch_url_content(url: str, timeout: int = 10) -> str:
    """Fetch and clean text content from a URL."""
    try:
        headers = {"User-Agent": "Mozilla/5.0 (compatible; MyRAGBot/1.0)"}
        resp = requests.get(url, headers=headers, timeout=timeout)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")
        # Remove nav, footer, script, style
        for tag in soup(["nav", "footer", "script", "style", "header", "aside"]):
            tag.decompose()
        return soup.get_text(separator="\n", strip=True)
    except Exception as e:
        logger.warning(f"Could not fetch {url}: {e}")
        return ""


def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> List[str]:
    """
    Split text into overlapping chunks of approximately chunk_size words.
    We use word-based chunking for simplicity; sentence-boundary is preserved
    by splitting on newlines first then words.
    """
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        end = min(start + chunk_size, len(words))
        chunk = " ".join(words[start:end])
        if chunk.strip():
            chunks.append(chunk.strip())
        if end >= len(words):
            break
        start += chunk_size - overlap
    return chunks


def make_doc_id(text: str, index: int) -> str:
    """Generate a stable, unique ID for a chunk."""
    h = hashlib.sha256((text + str(index)).encode()).hexdigest()[:16]
    return f"chunk-{h}"


def build_endee_client() -> Endee:
    """Connect to Endee server."""
    client = Endee()
    # Override base URL from config
    base_url = f"{ENDEE_HOST}/api/v1"
    client.set_base_url(base_url)
    # Set auth token if provided
    if ENDEE_AUTH_TOKEN:
        client.set_auth_token(ENDEE_AUTH_TOKEN)
    return client


def ensure_index(client: Endee, index_name: str, reset: bool = False) -> Any:
    """Create (or reset) the Endee index dynamically."""
    if reset:
        try:
            logger.info(f"Deleting existing index '{index_name}'...")
            client.delete_index(index_name)
            time.sleep(1)
        except Exception:
            pass

    try:
        logger.info(f"Creating index '{index_name}' (dim={EMBED_DIM}, cosine, INT8)...")
        client.create_index(
            name=index_name,
            dimension=EMBED_DIM,
            space_type="cosine",
            precision=Precision.INT8,
        )
        logger.info("Index created.")
    except Exception as e:
        if "already exists" in str(e).lower() or "exist" in str(e).lower():
            logger.info(f"Index '{index_name}' already exists, reusing it.")
        else:
            raise
    return client.get_index(name=index_name)


def ingest_documents(index_name: str, docs: List[Dict[str, str]], links: List[str], reset: bool = False) -> int:
    """
    Main ingestion pipeline for multiple subjects.
    docs: list of dicts with {"title": str, "content": str}
    links: list of URLs to scrape
    Returns number of vectors upserted.
    """
    # 1. Load embedding model
    logger.info(f"Loading embedding model: {EMBED_MODEL}")
    embedder = SentenceTransformer(EMBED_MODEL)

    # 2. Connect to Endee
    client = build_endee_client()
    index = ensure_index(client, index_name, reset=reset)

    # 3. Process documents
    all_items = []
    
    # Standardize input docs
    processed_docs = []
    
    for d in docs:
        if d.get("content"):
            processed_docs.append({
                "title": d.get("title", "Uploaded Document"),
                "url": d.get("title", "Uploaded Document"),
                "category": "uploaded",
                "content": d["content"].strip()
            })
            
    # Process links
    for url in links:
        logger.info(f"Fetching live content from: {url}")
        content = fetch_url_content(url)
        if content and len(content) > 200:
            processed_docs.append({
                "title": f"Live: {url}",
                "url": url,
                "category": "web",
                "content": content[:8000],  # cap to avoid very long pages
            })

    # 4. Chunk and embed
    for doc in processed_docs:
        title = doc.get("title", "Untitled")
        url = doc.get("url", "")
        category = doc.get("category", "general")
        content = doc.get("content", "").strip()

        if not content:
            continue

        chunks = chunk_text(content)
        logger.info(f"  '{title}' → {len(chunks)} chunks")

        if not chunks:
            continue

        # Batch embed
        embeddings = embedder.encode(chunks, show_progress_bar=False, normalize_embeddings=True)

        for i, (chunk, emb) in enumerate(zip(chunks, embeddings)):
            doc_id = make_doc_id(chunk, i)
            all_items.append({
                "id": doc_id,
                "vector": emb.tolist(),
                "meta": {
                    "title": title,
                    "url": url,
                    "category": category,
                    "chunk_index": i,
                    "chunk_text": chunk[:500],  # store partial text in meta for reuse
                },
            })

    # 5. Batch upsert into Endee
    if not all_items:
        logger.info("No vectors to insert.")
        return 0

    logger.info(f"Upserting {len(all_items)} vectors into Endee index '{index_name}'...")
    BATCH_SIZE = 100
    for start in range(0, len(all_items), BATCH_SIZE):
        batch = all_items[start : start + BATCH_SIZE]
        try:
            index.upsert(batch)
            logger.info(f"  Upserted batch {start // BATCH_SIZE + 1} ({len(batch)} vectors)")
        except Exception as e:
            logger.error(f"  Upsert error on batch {start // BATCH_SIZE + 1}: {e}")
            raise

    logger.info(f"✅ Ingestion complete! {len(all_items)} vectors indexed in '{index_name}'.")
    return len(all_items)

# Script mode disabled to favor app.py triggers.
