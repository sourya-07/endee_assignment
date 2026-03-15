"""
retriever.py – Multi-Subject Notes App
Retrieval layer: queries Endee, optionally reranks results.
"""

import logging
from typing import List, Dict, Any, Optional

from sentence_transformers import SentenceTransformer, CrossEncoder
from endee import Endee

from .config import (
    ENDEE_HOST,
    ENDEE_AUTH_TOKEN,
    EMBED_MODEL,
    TOP_K,
    RERANK_TOP_N,
)

logger = logging.getLogger(__name__)


class EndeeRetriever:
    """
    Retrieves relevant document chunks from Endee using dense vector search.
    Optionally reranks results with a cross-encoder for higher precision.
    """

    def __init__(self, use_reranker: bool = True):
        logger.info(f"Loading embedding model: {EMBED_MODEL}")
        self.embedder = SentenceTransformer(EMBED_MODEL)

        # Cross-encoder reranker (small, fast, free)
        self.use_reranker = use_reranker
        if use_reranker:
            logger.info("Loading cross-encoder reranker: cross-encoder/ms-marco-MiniLM-L-6-v2")
            try:
                self.reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
            except Exception as e:
                logger.warning(f"Could not load reranker: {e}. Disabling reranking.")
                self.use_reranker = False
                self.reranker = None
        else:
            self.reranker = None

        # Connect to Endee
        client = Endee()
        base_url = f"{ENDEE_HOST}/api/v1"
        client.set_base_url(base_url)
        if ENDEE_AUTH_TOKEN:
            client.set_auth_token(ENDEE_AUTH_TOKEN)

        self.client = client

    def retrieve(
        self,
        index_name: str,
        query: str,
        top_k: int = TOP_K,
        category_filter: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Retrieve top-k semantically similar chunks for `query` from the specified index.
        Optionally filter by category (concept, architecture, practical, tooling, product, technical).
        Returns a list of dicts with keys: id, score, title, url, category, text.
        """
        try:
            index = self.client.get_index(name=index_name)
        except Exception:
            logger.error(f"Endee index '{index_name}' not available.")
            return []

        # Embed query
        query_vec = self.embedder.encode(
            query, normalize_embeddings=True, show_progress_bar=False
        ).tolist()

        # Build metadata filter
        meta_filter = None
        if category_filter:
            meta_filter = {"category": {"$eq": category_filter}}

        # Query Endee
        try:
            results = index.query(
                vector=query_vec,
                top_k=top_k,
                filter=meta_filter,
            )
        except TypeError:
            # Older SDK version may not support `filter` keyword
            results = index.query(vector=query_vec, top_k=top_k)

        if not results:
            return []

        # Normalise result format across SDK versions
        candidates = []
        for r in results:
            # SDK may return objects or dicts
            if hasattr(r, "id"):
                item_id = r.id
                score = getattr(r, "similarity", getattr(r, "score", 0.0))
                meta = getattr(r, "meta", {}) or {}
            else:
                item_id = r.get("id", "")
                score = r.get("similarity", r.get("score", 0.0))
                meta = r.get("meta", {}) or {}

            candidates.append(
                {
                    "id": item_id,
                    "score": float(score),
                    "title": meta.get("title", "Unknown"),
                    "url": meta.get("url", ""),
                    "category": meta.get("category", "general"),
                    "chunk_index": meta.get("chunk_index", 0),
                    "text": meta.get("chunk_text", ""),
                }
            )

        # Optional reranking
        if self.use_reranker and self.reranker and len(candidates) > 1:
            pairs = [(query, c["text"]) for c in candidates]
            rerank_scores = self.reranker.predict(pairs)
            for c, s in zip(candidates, rerank_scores):
                c["rerank_score"] = float(s)
            candidates.sort(key=lambda x: x["rerank_score"], reverse=True)

        return candidates[:RERANK_TOP_N]

    def retrieve_with_sources(
        self,
        index_name: str,
        query: str,
        top_k: int = TOP_K,
        category_filter: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Retrieve and format context + sources for use in the RAG prompt.
        Returns: {"context": str, "sources": list[dict]}
        """
        chunks = self.retrieve(index_name=index_name, query=query, top_k=top_k, category_filter=category_filter)

        context_parts = []
        sources = []
        seen_urls = set()

        for i, chunk in enumerate(chunks, 1):
            context_parts.append(
                f"[Source {i}: {chunk['title']}] (URL: {chunk['url']})\n{chunk['text']}\n"
            )
            url = chunk["url"]
            if url not in seen_urls:
                sources.append(
                    {
                        "title": chunk["title"],
                        "url": url,
                        "category": chunk["category"],
                        "score": chunk.get("rerank_score", chunk["score"]),
                    }
                )
                seen_urls.add(url)

        return {
            "context": "\n---\n".join(context_parts),
            "sources": sources,
            "chunks": chunks,
        }
