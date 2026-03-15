"""
evaluate.py – Multi-Subject Notes App
Offline evaluation: runs a benchmark question set and reports metrics.

Usage:
  python evaluate.py [--questions questions.json]
"""

import argparse
import json
import logging
import statistics
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.rag_chain import RAGChain

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Default benchmark questions
DEFAULT_QUESTIONS = [
    "What is Antigravity's infinite context window?",
    "How does Endee support hybrid search?",
    "What are the key failure modes in RAG systems?",
    "Explain the chunking strategy for RAG pipelines.",
    "How does Antigravity use artifacts for durable context?",
    "What embedding model is recommended for Endee?",
    "What is the difference between model-native context and system-level memory?",
    "How does the cross-encoder reranker improve retrieval quality?",
    "What is the 'lost in the middle' problem in long-context LLMs?",
    "How does Endee's INT8 precision help with memory efficiency?",
]


def evaluate(questions: list[str]) -> dict:
    """Run evaluation and return aggregate metrics."""
    logger.info(f"Evaluating {len(questions)} questions...")
    chain = RAGChain(use_reranker=True)

    results = []
    for i, q in enumerate(questions, 1):
        logger.info(f"[{i}/{len(questions)}] {q[:60]}...")
        resp = chain.run(q, compute_metrics=True)
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
        "--questions",
        type=str,
        default=None,
        help="Path to JSON file with list of question strings",
    )
    args = parser.parse_args()

    if args.questions:
        with open(args.questions) as f:
            questions = json.load(f)
    else:
        questions = DEFAULT_QUESTIONS

    evaluate(questions)
