"""
eval_harness.py
====================================================================
Offline Evaluation Harness:
  - Loads and validates evaluation/golden_dataset.json
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
    SAVE_K,
    snapshot,
)
from retriever import MultimodalRetriever

# ------------------------------------------------------------------
# Metric Utilities
# ------------------------------------------------------------------
def compute_metrics_at_k(
    hits: List[Dict[str, Any]],
    gold_pages: List[Tuple[str, int]],
    k_vals: List[int] = K_VALUES
) -> Tuple[Dict[int, float], Dict[int, float], float]:
    """
    Computes Recall@k, NDCG@k for each k, and overall MRR (at MAX_K).
    Accepts hits as either a list of dicts with 'doc_id'/'page_num' OR a list of (doc_id, page_num) tuples.
    """
    retrieved_pages = []
    for h in hits:
        if isinstance(h, dict):
            retrieved_pages.append((str(h["doc_id"]), int(h["page_num"])))
        elif isinstance(h, (list, tuple)):
            retrieved_pages.append((str(h[0]), int(h[1])))

    gold_set = set((str(doc_id), int(page_num)) for doc_id, page_num in gold_pages)
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
# Dataset Normalization, Loading & Validation
# ------------------------------------------------------------------
def normalize_gold_page(p: Any) -> Tuple[str, int]:
    """Coerces both dict format {'doc_id': ..., 'page_num': ...} and tuple/list format to (doc_id, page_num)."""
    if isinstance(p, dict):
        if "doc_id" not in p or "page_num" not in p:
            raise ValueError(f"Malformed gold_page dict missing 'doc_id' or 'page_num': {p}")
        return str(p["doc_id"]), int(p["page_num"])
    elif isinstance(p, (list, tuple)) and len(p) >= 2:
        return str(p[0]), int(p[1])
    else:
        raise ValueError(f"Unrecognized gold_page item structure: {p}")

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

        # NORMALIZATION STEP: Convert to List[Tuple[str, int]]
        rec["gold_pages"] = [normalize_gold_page(p) for p in rec["gold_pages"]]

        # Check exclusion targets
        doc_ids = {doc_id for doc_id, _ in rec["gold_pages"]}
        if any(d in EXCLUDE_DOCS for d in doc_ids):
            continue

        # Separate unanswerable records
        if not rec["answerable"]:
            unanswerable_records.append(rec)
            continue

        # Confirm target exists in Qdrant
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
    print(f" Running Offline Retrieval Benchmark across {len(RETRIEVAL_MODES)} Study Arms")
    print("=" * 70)

    for idx, rec in enumerate(eval_set, 1):
        q = rec["question"]
        gold_pages = [tuple(p) for p in rec["gold_pages"]]

        for mode in RETRIEVAL_MODES:
            # 1. Retrieve at SAVE_K with pools
            hits, lat, pools = retriever.search(mode, q, k=SAVE_K, return_pools=True)
            latencies[mode].append(lat)

            # 2. Compute metrics (passing the actual hits list directly)
            recalls, ndcgs, mrr = compute_metrics_at_k(hits, gold_pages)

            # 3. Rich hits structure
            rich_hits = [
                (
                    h.get("doc_id"),
                    int(h.get("page_num")),
                    float(h.get("score", 0.0)),
                    h.get("page_type"),
                    h.get("source"),
                )
                for h in hits
            ]

            # 4. Rich pools structure
            rich_pools = {
                name: [
                    (
                        p.get("doc_id"),
                        int(p.get("page_num")),
                        float(p.get("score", 0.0)),
                        p.get("page_type"),
                    )
                    for p in pool
                ]
                for name, pool in (pools or {}).items()
            }

            entry = {
                "qid": rec["qid"],
                "question": q,
                "gold_pages": gold_pages,
                "gold_page_types": rec.get("gold_page_types", []),
                "evidence_type": rec.get("evidence_type", "unknown"),
                "difficulty": rec.get("difficulty", "unknown"),
                "reasoning_type": rec.get("reasoning_type", "unknown"),
                "multi_page": rec.get("multi_page", False),
                "bucket": rec.get("bucket", "standard"),
                "latency_sec": float(lat),
                "recalls": recalls,
                "ndcgs": ndcgs,
                "mrr": float(mrr),
                "hits": rich_hits,
                "pools": rich_pools,
            }
            detailed_results[mode].append(entry)

        if idx % 5 == 0 or idx == len(eval_set):
            print(f"Processed {idx}/{len(eval_set)} queries...")

    # ------------------------------------------------------------------
    # Results Aggregation and Breakdown Slicing
    # ------------------------------------------------------------------
    summary_table = {}
    for mode in RETRIEVAL_MODES:
        mode_data = detailed_results[mode]

        # Overall Metrics
        mean_r3 = np.mean([item["recalls"][3] for item in mode_data]) * 100
        mean_ndcg3 = np.mean([item["ndcgs"][3] for item in mode_data]) * 100
        mean_mrr = np.mean([item["mrr"] for item in mode_data]) * 100

        # Page Type Breakdown (Recall@3)
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
    # Print Secondary Slices Breakdown
    # ------------------------------------------------------------------
    print("\n" + "=" * 98)
    print("     SECONDARY SLICES BREAKDOWN (Recall@3: Routed vs Hybrid v2 vs Fusion Unrouted)")
    print("=" * 98)
    for slice_field in ["evidence_type", "difficulty", "reasoning_type", "multi_page", "bucket"]:
        print(f"\n--- Breakdown by: {slice_field} ---")
        base_recs = detailed_results["text"]
        slice_keys = sorted(list({str(r.get(slice_field)) for r in base_recs}))
        for k_val in slice_keys:
            q_count = sum(1 for r in base_recs if str(r.get(slice_field)) == k_val)
            sub_routed = [r for r in detailed_results["hybrid_routed"] if str(r.get(slice_field)) == k_val]
            sub_v2 = [r for r in detailed_results.get("hybrid_v2", []) if str(r.get(slice_field)) == k_val]
            sub_unrouted = [r for r in detailed_results["fusion_unrouted"] if str(r.get(slice_field)) == k_val]

            r3_routed = np.mean([r["recalls"][3] for r in sub_routed]) * 100 if sub_routed else 0.0
            r3_v2 = np.mean([r["recalls"][3] for r in sub_v2]) * 100 if sub_v2 else 0.0
            r3_unrouted = np.mean([r["recalls"][3] for r in sub_unrouted]) * 100 if sub_unrouted else 0.0

            print(
                f"  {k_val:<32} (N={q_count:2d} ) | Routed: {r3_routed:5.1f}% | "
                f"Hybrid v2: {r3_v2:5.1f}% | Unrouted: {r3_unrouted:5.1f}%"
            )

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