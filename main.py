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
from evaluation.retrieval_eval import run_evaluation
from config import RETRIEVAL_MODES
from config import MAX_K, GOLDEN_PATH, RESULTS_DIR
from context_builder import ContextBuilder

from evaluation.generation_eval import (
    run_generation_evaluation,
)

from config import (
    GENERATOR_CONTEXT_PAGES,
    GENERATION_RESULTS_DIR,
)

def main():
    parser = argparse.ArgumentParser(description="Multimodal Document Intelligence Pipeline")
    parser.add_argument("--eval", nargs="?", const="retrieval", choices=["retrieval", "generation"],
                    default=None, help="Run retrieval or generation evaluation (alone defaults to retrieval).")
    parser.add_argument("--golden", type=str, default=GOLDEN_PATH, help="Path to golden dataset JSON.")
    parser.add_argument("--out", type=str, default=None, help="Override evaluation output directory.")
    parser.add_argument("--query", type=str, help="Single query for end-to-end RAG.")
    parser.add_argument("--mode", type=str, choices=RETRIEVAL_MODES, default="hybrid_v2_d3", help="Retrieval mode for single-query execution.")
    parser.add_argument("--k", type=int, default=GENERATOR_CONTEXT_PAGES, help="Number of context pages (default: 3).")
    parser.add_argument("--experiment", choices=["retrieved", "oracle", "text_only"], default="retrieved", help="Generation evaluation variant.")
    parser.add_argument("--max-queries", type=int, default=None, help="Optional limit for a pilot evaluation.")

    args = parser.parse_args()

    # 1. Benchmark Execution Branch

    # ------------------------------------------------------
    # Independent evaluation entry points.
    # ------------------------------------------------------

    if args.eval == "retrieval":
        # Existing retrieval benchmark is unchanged.
        run_evaluation(
            golden_path=args.golden,
            output_dir=args.out or RESULTS_DIR,
        )
        return

    if args.eval == "generation":

        retriever = MultimodalRetriever()
        context_builder = ContextBuilder()
        generator = MultimodalGenerator(
                provider="gemini",
                model_name="gemini-3.8-flash",
                layout_text_enabled= False)

        run_generation_evaluation(
            golden_path=args.golden,
            retriever=retriever,
            context_builder=context_builder,
            generator=generator,
            experiment=args.experiment,
            output_dir=(
                args.out or GENERATION_RESULTS_DIR
            ),
            max_queries=args.max_queries,
        )
        return


    # 2. Single-Query Execution Branch
    if not args.query:
        parser.error("--query is required when --eval is not passed.")

    # Initialize components
    retriever = MultimodalRetriever()
    context_builder = ContextBuilder()
    generator = MultimodalGenerator()

    app = build_rag_graph(
        retriever=retriever,
        generator=generator,
        context_builder=context_builder,
    )

    print(f"\n=======================================================")
    print(f"Executing Query: '{args.query}' (Mode: {args.mode}, k={args.k})")
    print(f"=======================================================\n")

    result = app.invoke({
        "mode": args.mode,
        "query": args.query,
        "k": args.k
    })

    print("\n========== Generation Result ==========")

    print("\nAnswer:")
    print(result["answer"])

    print("\nAnswerable from supplied evidence:")
    print(result["answerable"])

    print("\nCitations:")
    for citation in result["citations"]:
        print(
            f"[{citation['citation_id']}] "
            f"{citation['doc_id']} "
            f"(PDF page {citation['page_num']})"
        )

    print("\nRetrieved Pages:")
    for rank, hit in enumerate(
        result["retrieved_docs"],
        start=1,
    ):
        score = hit.get(
            "rrf_score",
            hit.get("score", 0.0),
        )

        print(
            f"{rank}. {hit['doc_id']} "
            f"| Page {hit['page_num']} "
            f"| Type {hit['page_type']} "
            f"| Score {score:.4f}"
        )

    print("\nToken Usage:")
    print(result["usage"])

    print("\nLatency:")
    print(
        f"Retrieval: "
        f"{result['retrieval_latency']:.3f}s"
    )

    print(
        f"Context construction: "
        f"{result['context_latency']:.3f}s"
    )

    print(
        f"Generation: "
        f"{result['generation_latency']:.3f}s"
    )


if __name__ == "__main__":
    main()