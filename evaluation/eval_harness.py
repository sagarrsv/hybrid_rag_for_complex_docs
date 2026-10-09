"""
eval_harness.py
====================================================================
Offline Evaluation Harness:
  - Loads and validates evaluation/golden_set.json
  - Excludes suspect papers and filters unanswerable queries
  - Validates gold (doc_id, page_num) existence in Qdrant
  - Computes MRR@10 and Recall@k, NDCG@k across K_VALUES
  - Slices metrics by page_type, evidence_type, difficulty, reasoning_type,
    multi_page, and near-duplicate buckets
  - Records p50/p95 latency and saves query-level traces with snapshot()
====================================================================
"""

import os
import json
import math
import numpy as np
from typing import List, Dict, Any, Tuple
from collections import defaultdict

from config import (
    GOLDEN_PATH,
    RESULTS_DIR,
    EXCLUDE_DOCS,
    K_VALUES,
    MAX_K,
    RETRIEVAL_MODES,
    PT_TEXT,
    PT_VISUAL,
    snapshot,
)
from retriever import MultimodalRetriever


# ------------------------------------------------------------------
# Metric Utilities
# ------------------------------------------------------------------
def compute_metrics_at_k(
    hits: List[Dict[str, Any]],
    gold_pages: List[Tuple[str, int]],
    k_vals: List[int]
) -> Tuple[Dict[int, float], Dict[int, float], float]:
    """
    Computes Recall@k, NDCG@k for each k, and overall MRR (at MAX_K).
    gold_pages is a list of (doc_id, page_num) pairs.
    """
    retrieved_pages = [(h["doc_id"], h["page_num"]) for h in hits]
    gold_set = set(gold_pages)

    recalls = {}
    ndcgs = {}

    for k in k_vals:
        top_k = retrieved_pages[:k]
        hits_in_k = [1 if p in gold_set else 0 for p in top_k]
        num_hits = sum(hits_in_k)

        # Recall@k
        recalls[k] = num_hits / len(gold_set) if gold_set else 0.0

        # DCG@k
        dcg = sum((val / math.log2(idx + 2)) for idx, val in enumerate(hits_in_k))
        # IDCG@k
        ideal_hits = [1] * min(len(gold_set), k)
        idcg = sum((val / math.log2(idx + 2)) for idx, val in enumerate(ideal_hits))
        ndcgs[k] = (dcg / idcg) if idcg > 0.0 else 0.0

    # MRR calculation (first relevant page match)
    mrr = 0.0
    for idx, page in enumerate(retrieved_pages[:MAX_K]):
        if page in gold_set:
            mrr = 1.0 / (idx + 1)
            break

    return recalls, ndcgs, mrr


# ------------------------------------------------------------------
# Dataset Loading & Validation
# ------------------------------------------------------------------
def load_and_validate_dataset(
    golden_path: str,
    retriever: MultimodalRetriever
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    if not os.path.exists(golden_path):
        raise FileNotFoundError(f"Golden dataset not found at '{golden_path}'")

    with open(golden_path, "r", encoding="utf-8") as f:
        records = json.load(f)

    seen_qids = set()
    eval_records = []
    unanswerable_records = []

    print("[Harness] Validating golden dataset and vector store registrations...")
    for rec in records:
        for req_field in ["qid", "question", "gold_pages", "answerable"]:
            if req_field not in rec:
                raise ValueError(f"Record missing required field '{req_field}': {rec}")

        qid = rec["qid"]
        if qid in seen_qids:
            raise ValueError(f"Duplicate qid detected: {qid}")
        seen_qids.add(qid)

        # Extract doc IDs and filter suspect papers
        doc_ids = {page[0] for page in rec["gold_pages"]}
        if any(d in EXCLUDE_DOCS for d in doc_ids):
            continue

        # Separate unanswerable records for CRAG evaluation
        if not rec["answerable"]:
            unanswerable_records.append(rec)
            continue

        # Fail loudly if any gold page ID does not exist in Qdrant
        for doc_id, page_num in rec["gold_pages"]:
            if not retriever.id_exists(doc_id, page_num):
                raise ValueError(
                    f"Validation failed: Gold target ({doc_id}, page {page_num}) "
                    f"for qid '{qid}' does not exist in Qdrant index."
                )

        eval_records.append(rec)

    print(f"[Harness] Validated {len(eval_records)} answerable queries ({len(unanswerable_records)} unanswerable held out).")
    return eval_records, unanswerable_records


# ------------------------------------------------------------------
# Ablation Benchmark Execution
# ------------------------------------------------------------------
def run_evaluation(golden_path: str = GOLDEN_PATH, output_dir: str = RESULTS_DIR):
    retriever = MultimodalRetriever()
    retriever.warmup()

    eval_set, unanswerable_set = load_and_validate_dataset(golden_path, retriever)
    os.makedirs(output_dir, exist_ok=True)

    # Annotate gold pages with verified page_type from Qdrant
    for rec in eval_set:
        rec["gold_page_types"] = [
            retriever.get_page_type(doc_id, page_num)
            for doc_id, page_num in rec["gold_pages"]
        ]

    detailed_results = {mode: [] for mode in RETRIEVAL_MODES}
    latencies = {mode: [] for mode in RETRIEVAL_MODES}

    print("\n" + "=" * 70)
    print(" Running Offline Retrieval Benchmark across 4 Study Arms")
    print("=" * 70)

    for idx, rec in enumerate(eval_set, 1):
        q = rec["question"]
        gold_pages = [tuple(p) for p in rec["gold_pages"]]

        for mode in RETRIEVAL_MODES:
            hits, lat = retriever.search(mode, q, k=MAX_K)
            latencies[mode].append(lat)

            recalls, ndcgs, mrr = compute_metrics_at_k(hits, gold_pages, K_VALUES)
            detailed_results[mode].append({
                "qid": rec["qid"],
                "question": q,
                "gold_pages": gold_pages,
                "gold_page_types": rec["gold_page_types"],
                "evidence_type": rec.get("evidence_type", "unspecified"),
                "difficulty": rec.get("difficulty", "unspecified"),
                "reasoning_type": rec.get("reasoning_type", "unspecified"),
                "multi_page": rec.get("multi_page", len(gold_pages) > 1),
                "bucket": rec.get("bucket", "standard"),
                "latency_sec": lat,
                "recalls": recalls,
                "ndcgs": ndcgs,
                "mrr": mrr,
                "hits": [(h["doc_id"], h["page_num"], h["score"]) for h in hits]
            })

        if idx % 10 == 0 or idx == len(eval_set):
            print(f"Processed {idx}/{len(eval_set)} queries...")

    # ------------------------------------------------------------------
    # Results Aggregation and Breakdown Slicing
    # ------------------------------------------------------------------
    summary_table = {}

    for mode in RETRIEVAL_MODES:
        mode_data = detailed_results[mode]
        total_q = len(mode_data)

        # Overall Metrics
        mean_r3 = np.mean([item["recalls"][3] for item in mode_data]) * 100
        mean_ndcg3 = np.mean([item["ndcgs"][3] for item in mode_data]) * 100
        mean_mrr = np.mean([item["mrr"] for item in mode_data]) * 100

        # Page Type Breakdown (Recall@3)
        # Sliced based on whether the primary gold page is text-dense or layout-heavy
        text_dense_q = [
            item for item in mode_data
            if item["gold_page_types"] and item["gold_page_types"][0] == PT_TEXT
        ]
        layout_heavy_q = [
            item for item in mode_data
            if item["gold_page_types"] and item["gold_page_types"][0] == PT_VISUAL
        ]

        r3_text = (np.mean([item["recalls"][3] for item in text_dense_q]) * 100) if text_dense_q else 0.0
        r3_layout = (np.mean([item["recalls"][3] for item in layout_heavy_q]) * 100) if layout_heavy_q else 0.0

        # Latencies
        lats = latencies[mode]
        p50 = np.percentile(lats, 50) * 1000
        p95 = np.percentile(lats, 95) * 1000

        summary_table[mode] = {
            "Overall_Recall@3": mean_r3,
            "TextDense_Recall@3": r3_text,
            "LayoutHeavy_Recall@3": r3_layout,
            "Overall_NDCG@3": mean_ndcg3,
            "MRR": mean_mrr,
            "Latency_p50_ms": p50,
            "Latency_p95_ms": p95,
        }

    # ------------------------------------------------------------------
    # Print Main Ablation Table
    # ------------------------------------------------------------------
    print("\n" + "=" * 90)
    print("                 MAIN MULTIMODAL RETRIEVAL ABLATION TABLE")
    print("=" * 90)
    print(f"{'Study Mode':<18} | {'Overall R@3':<12} | {'Text-Dense':<12} | {'Layout-Heavy':<12} | {'MRR':<8} | {'p50 (ms)':<10} | {'p95 (ms)':<10}")
    print("-" * 90)
    for mode in RETRIEVAL_MODES:
        row = summary_table[mode]
        print(
            f"{mode:<18} | "
            f"{row['Overall_Recall@3']:>10.1f}% | "
            f"{row['TextDense_Recall@3']:>10.1f}% | "
            f"{row['LayoutHeavy_Recall@3']:>10.1f}% | "
            f"{row['MRR']:>6.1f}% | "
            f"{row['Latency_p50_ms']:>8.1f}ms | "
            f"{row['Latency_p95_ms']:>8.1f}ms"
        )
    print("=" * 90)

    # ------------------------------------------------------------------
    # Print Secondary Slices (Hybrid-Routed vs Fusion-Unrouted)
    # ------------------------------------------------------------------
    print("\n" + "=" * 90)
    print("     SECONDARY SLICES BREAKDOWN (Recall@3: Hybrid Routed vs Fusion Unrouted)")
    print("=" * 90)

    for slice_key in ["evidence_type", "difficulty", "reasoning_type", "multi_page", "bucket"]:
        print(f"\n--- Breakdown by: {slice_key} ---")
        categories = set(item[slice_key] for item in detailed_results["hybrid_routed"])
        for cat in sorted(categories, key=str):
            sub_hr = [it for it in detailed_results["hybrid_routed"] if it[slice_key] == cat]
            sub_fu = [it for it in detailed_results["fusion_unrouted"] if it[slice_key] == cat]
            hr_r3 = np.mean([it["recalls"][3] for it in sub_hr]) * 100 if sub_hr else 0.0
            fu_r3 = np.mean([it["recalls"][3] for it in sub_fu]) * 100 if sub_fu else 0.0
            print(f"  {str(cat):<25} (N={len(sub_hr):<3}) | Hybrid Routed: {hr_r3:>6.1f}% | Fusion Unrouted: {fu_r3:>6.1f}%")

    # ------------------------------------------------------------------
    # Save Artifacts
    # ------------------------------------------------------------------
    results_path = os.path.join(output_dir, "benchmark_results.json")
    snapshot_path = os.path.join(output_dir, "config_snapshot.json")

    with open(results_path, "w", encoding="utf-8") as f:
        json.dump({
            "summary": summary_table,
            "detailed": detailed_results,
            "unanswerable_held_out": unanswerable_set
        }, f, indent=2)

    with open(snapshot_path, "w", encoding="utf-8") as f:
        json.dump(snapshot(), f, indent=2)

    print(f"\n[Harness] Benchmark complete. Artifacts saved to '{output_dir}'.")