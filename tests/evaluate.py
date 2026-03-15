"""
evaluate.py – Multi-Subject Notes App
Offline evaluation: runs a benchmark question set and reports metrics.

Usage:
  python -m tests.evaluate --index <index_name> [--questions questions.json]
"""

import argparse
import json
import logging
import statistics
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.rag_chain import RAGChain
from src.subjects_db import load_subjects

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Default benchmark questions
DEFAULT_QUESTIONS = [
    "What is Endee's infinite context window?",
    "How does Endee support hybrid search?",
    "What are the key failure modes in RAG systems?",
    "Explain the chunking strategy for RAG pipelines.",
    "How does Endee use artifacts for durable context?",
    "What embedding model is recommended for Endee?",
    "What is the difference between model-native context and system-level memory?",
    "How does the cross-encoder reranker improve retrieval quality?",
    "What is the 'lost in the middle' problem in long-context LLMs?",
    "How does Endee's INT8 precision help with memory efficiency?",
]


def _resolve_index_name(index_arg: str | None) -> str:
    """Resolve an index name from CLI arg or fall back to the first subject."""
    if index_arg:
        return index_arg
    subjects = load_subjects()
    if subjects:
        first = next(iter(subjects.values()))
        name = first.get("index_name", "default")
        logger.info(f"No --index given, using first subject index: '{name}'")
        return name
    logger.error("No --index given and no subjects found in subjects.json")
    sys.exit(1)


def evaluate(index_name: str, questions: list[str]) -> dict:
    """Run evaluation and return aggregate metrics."""
    logger.info(f"Evaluating {len(questions)} questions against index '{index_name}'...")
    chain = RAGChain(use_reranker=True)

    results = []
    for i, q in enumerate(questions, 1):
        logger.info(f"[{i}/{len(questions)}] {q[:60]}...")
        resp = chain.run(index_name=index_name, question=q, compute_metrics=True)
        results.append({
            "question": q,
            "answer_preview": resp.answer[:200],
            "faithfulness": resp.faithfulness,
            "answer_relevancy": resp.answer_relevancy,
            "latency": resp.latency_seconds,
            "num_sources": len(resp.sources),
        })

    # Aggregate
    faithfulness_scores = [r["faithfulness"] for r in results]
    relevancy_scores = [r["answer_relevancy"] for r in results]
    latencies = [r["latency"] for r in results]

    summary = {
        "index_name": index_name,
        "total_questions": len(questions),
        "avg_faithfulness": statistics.mean(faithfulness_scores),
        "avg_answer_relevancy": statistics.mean(relevancy_scores),
        "avg_latency_seconds": statistics.mean(latencies),
        "min_faithfulness": min(faithfulness_scores),
        "max_faithfulness": max(faithfulness_scores),
        "results": results,
    }

    # Print report
    print("\n" + "=" * 60)
    print("📊 RAG Evaluation Report")
    print("=" * 60)
    print(f"Index:                {summary['index_name']}")
    print(f"Total Questions:      {summary['total_questions']}")
    print(f"Avg Faithfulness:     {summary['avg_faithfulness']:.3f}  (higher = more grounded)")
    print(f"Avg Answer Relevancy: {summary['avg_answer_relevancy']:.3f}  (higher = more on-topic)")
    print(f"Avg Latency:          {summary['avg_latency_seconds']:.2f}s")
    print("=" * 60)

    output_path = "eval_results.json"
    with open(output_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nFull results saved to: {output_path}")

    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate RAG pipeline")
    parser.add_argument(
        "--index",
        type=str,
        default=None,
        help="Endee index name to evaluate against (defaults to first subject)",
    )
    parser.add_argument(
        "--questions",
        type=str,
        default=None,
        help="Path to JSON file with list of question strings",
    )
    args = parser.parse_args()

    index_name = _resolve_index_name(args.index)

    if args.questions:
        with open(args.questions) as f:
            questions = json.load(f)
    else:
        questions = DEFAULT_QUESTIONS

    evaluate(index_name, questions)
