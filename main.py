"""
main.py
====================================================================
Unified CLI entry point:
  - Ad-Hoc single query execution with live LangGraph pipeline
  - Offline ablation evaluation suite with snapshot artifacts
====================================================================
"""

import argparse
from retriever import MultimodalRetriever
from generator import MultimodalGenerator
from graph import build_rag_graph
from evaluation.eval_harness import run_evaluation
from config import MAX_K, GOLDEN_PATH, RESULTS_DIR


def main():
    parser = argparse.ArgumentParser(description="Multimodal Document Intelligence Pipeline")
    parser.add_argument("--eval", action="store_true", help="Execute offline ablation benchmark")
    parser.add_argument("--golden", type=str, default=GOLDEN_PATH, help="Path to golden_set.json")
    parser.add_argument("--out", type=str, default=RESULTS_DIR, help="Results output directory")
    parser.add_argument("--query", type=str, help="Question to evaluate through live graph")
    parser.add_argument("--mode", type=str, default="hybrid_routed",
                        choices=["text", "visual", "hybrid_routed", "fusion_unrouted"],
                        help="Retrieval arm for single-query execution")
    parser.add_argument("--k", type=int, default=MAX_K, help="Candidate count limit")
    args = parser.parse_args()

    # 1. Benchmark Execution Branch
    if args.eval:
        run_evaluation(golden_path=args.golden, output_dir=args.out)
        return

    # 2. Single-Query Execution Branch
    if not args.query:
        parser.error("--query is required when --eval is not passed.")

    # Initialize components
    retriever = MultimodalRetriever()
    generator = MultimodalGenerator(retriever=retriever)
    app = build_rag_graph(retriever, generator)

    print(f"\n=======================================================")
    print(f"Executing Query: '{args.query}' (Mode: {args.mode}, k={args.k})")
    print(f"=======================================================\n")

    result = app.invoke({
        "mode": args.mode,
        "query": args.query,
        "k": args.k
    })

    print(f"Retrieval Latency: {result['retrieval_latency'] * 1000:.2f} ms")
    print("\n--- Retrieved Page-Collapsed Candidates ---")
    for idx, hit in enumerate(result["retrieved_docs"], 1):
        score_val = hit.get("rrf_score", hit["score"])
        print(f" {idx}. [{hit['source']}] Doc: {hit['doc_id'][:32]} | Page {hit['page_num']} | Type: {hit['page_type']} | Score: {score_val:.4f}")

    print("\n--- Generation Output ---")
    print(result["answer"])


if __name__ == "__main__":
    main()