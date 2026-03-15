"""
rag_chain.py – Antigravity Infinite Context RAG Demo
RAG chain: retrieves context from Endee, prompts the LLM, returns answer + sources.
Supports OpenAI-compatible APIs, Google Gemini, and Ollama (local).

Evaluation metrics (faithfulness + answer relevancy) computed without external libraries.
"""

import json
import logging
import time
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

from .retriever import EndeeRetriever
from .config import (
    EMBED_MODEL,
    LLM_PROVIDER,
    OPENAI_API_KEY,
    OPENAI_MODEL,
    GOOGLE_API_KEY,
    GOOGLE_MODEL,
    OLLAMA_BASE_URL,
    OLLAMA_MODEL,
    LLM_MAX_TOKENS,
    LLM_TEMPERATURE,
    TOP_K,
)

logger = logging.getLogger(__name__)

RAG_SYSTEM_PROMPT = r"""You are an expert knowledge assistant.
Your knowledge comes strictly from the provided context.

INSTRUCTIONS:
- Answer ONLY using the provided context below.
- Always cite your sources using inline Markdown links to the provided URL, e.g. [[Source N]](URL).
- If the context does not contain sufficient information to answer, say so explicitly — do not hallucinate.
- Be concise but comprehensive. Use bullet points where helpful.
- For technical topics, include code examples if they appear in the context.
- Do NOT output the formula `${\displaystyle {\textbf {F}}={\frac {d\mathbf {p} }{dt}}}$` or any complex LaTeX equations in your answer.

CONTEXT:
{context}
"""

RAG_USER_TEMPLATE = """Question: {question}

Please provide a clear, well-structured answer with citations."""


# ─────────────────────────────────────────────────────────────────────────────
# LLM Backends
# ─────────────────────────────────────────────────────────────────────────────

def _call_openai(system_prompt: str, user_message: str) -> str:
    try:
        import openai
        client = openai.OpenAI(api_key=OPENAI_API_KEY)
        response = client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message},
            ],
            max_tokens=LLM_MAX_TOKENS,
            temperature=LLM_TEMPERATURE,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        logger.error(f"OpenAI API error: {e}")
        return f"[LLM Error] {e}"


def _call_google(system_prompt: str, user_message: str) -> str:
    try:
        import google.generativeai as genai
        genai.configure(api_key=GOOGLE_API_KEY)
        model = genai.GenerativeModel(
            model_name=GOOGLE_MODEL,
            system_instruction=system_prompt,
        )
        response = model.generate_content(
            user_message,
            generation_config=genai.types.GenerationConfig(
                max_output_tokens=LLM_MAX_TOKENS,
                temperature=LLM_TEMPERATURE,
            ),
        )
        return response.text.strip()
    except Exception as e:
        logger.error(f"Google Gemini API error: {e}")
        return f"[LLM Error] {e}"


def _call_ollama(system_prompt: str, user_message: str) -> str:
    try:
        import requests
        payload = {
            "model": OLLAMA_MODEL,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message},
            ],
            "stream": False,
            "options": {
                "num_predict": LLM_MAX_TOKENS,
                "temperature": LLM_TEMPERATURE,
            },
        }
        resp = requests.post(
            f"{OLLAMA_BASE_URL}/api/chat", json=payload, timeout=120
        )
        resp.raise_for_status()
        data = resp.json()
        return data["message"]["content"].strip()
    except Exception as e:
        logger.error(f"Ollama API error: {e}")
        return f"[LLM Error] {e}"


def _call_llm(system_prompt: str, user_message: str) -> str:
    """Dispatch to the configured LLM provider."""
    provider = LLM_PROVIDER.lower()
    if provider == "openai":
        return _call_openai(system_prompt, user_message)
    elif provider == "google":
        return _call_google(system_prompt, user_message)
    elif provider == "ollama":
        return _call_ollama(system_prompt, user_message)
    else:
        return f"[Config Error] Unknown LLM_PROVIDER='{LLM_PROVIDER}'. Set to 'openai', 'google', or 'ollama'."


# ─────────────────────────────────────────────────────────────────────────────
# Evaluation Metrics (lightweight, no external eval library required)
# ─────────────────────────────────────────────────────────────────────────────

class _MetricComputer:
    """Lazy-loaded singleton for metric computation."""
    _instance: Optional["_MetricComputer"] = None
    _embedder: Optional[SentenceTransformer] = None

    @classmethod
    def get(cls) -> "SentenceTransformer":
        if cls._embedder is None:
            cls._embedder = SentenceTransformer(EMBED_MODEL)
        return cls._embedder


def compute_faithfulness(answer: str, context: str) -> float:
    """
    Approximate faithfulness: cosine similarity between answer embedding
    and context embedding. Range [0, 1]. Higher = more grounded.
    """
    if not answer or not context:
        return 0.0
    embedder = _MetricComputer.get()
    embs = embedder.encode([answer, context], normalize_embeddings=True)
    score = float(np.dot(embs[0], embs[1]))
    return max(0.0, min(1.0, score))


def compute_answer_relevancy(question: str, answer: str) -> float:
    """
    Approximate answer relevancy: cosine similarity between question embedding
    and answer embedding. Range [0, 1]. Higher = more relevant.
    """
    if not question or not answer:
        return 0.0
    embedder = _MetricComputer.get()
    embs = embedder.encode([question, answer], normalize_embeddings=True)
    score = float(np.dot(embs[0], embs[1]))
    return max(0.0, min(1.0, score))


# ─────────────────────────────────────────────────────────────────────────────
# Main RAG Chain
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class RAGResponse:
    question: str
    answer: str
    sources: List[Dict[str, Any]]
    context: str
    faithfulness: float
    answer_relevancy: float
    latency_seconds: float
    chunks: List[Dict[str, Any]] = field(default_factory=list)


class RAGChain:
    """
    Full RAG pipeline:
      query → Endee retrieval → (optional rerank) → LLM prompt → grounded answer
    """

    def __init__(self, use_reranker: bool = True):
        self.retriever = EndeeRetriever(use_reranker=use_reranker)

    def run(
        self,
        index_name: str,
        question: str,
        top_k: int = TOP_K,
        category_filter: Optional[str] = None,
        compute_metrics: bool = True,
    ) -> RAGResponse:
        """
        Execute the full RAG chain for a user question against a specific index.
        Returns a RAGResponse dataclass.
        """
        t0 = time.time()

        # 1. Retrieve context from Endee
        retrieval = self.retriever.retrieve_with_sources(
            index_name=index_name, query=question, top_k=top_k, category_filter=category_filter
        )
        context = retrieval["context"]
        sources = retrieval["sources"]
        chunks = retrieval["chunks"]

        if not context.strip():
            return RAGResponse(
                question=question,
                answer=(
                    "I could not find relevant information in the knowledge base. "
                    "Please make sure documents have been ingested by running: python ingest.py"
                ),
                sources=[],
                context="",
                faithfulness=0.0,
                answer_relevancy=0.0,
                latency_seconds=time.time() - t0,
                chunks=[],
            )

        # 2. Build prompts
        system_prompt = RAG_SYSTEM_PROMPT.format(context=context)
        user_message = RAG_USER_TEMPLATE.format(question=question)

        # 3. Call LLM
        answer = _call_llm(system_prompt, user_message)

        latency = time.time() - t0

        # 4. Compute evaluation metrics
        faithfulness = 0.0
        answer_relevancy = 0.0
        if compute_metrics and answer and not answer.startswith("[LLM Error]"):
            try:
                faithfulness = compute_faithfulness(answer, context)
                answer_relevancy = compute_answer_relevancy(question, answer)
            except Exception as e:
                logger.warning(f"Metric computation failed: {e}")

        return RAGResponse(
            question=question,
            answer=answer,
            sources=sources,
            context=context,
            faithfulness=faithfulness,
            answer_relevancy=answer_relevancy,
            latency_seconds=latency,
            chunks=chunks,
        )

    def run_batch(self, index_name: str, questions: List[str]) -> List[RAGResponse]:
        """Run multiple questions and return all responses."""
        return [self.run(index_name=index_name, question=q) for q in questions]
