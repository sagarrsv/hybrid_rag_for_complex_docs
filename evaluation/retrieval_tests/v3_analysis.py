#!/usr/bin/env python3
"""
v3_analysis.py
========================================================================================
Comprehensive Offline Retrieval & Split-Conditioned Ablation Analysis (V3):
  - Evaluates all live modes (including hybrid_v2 and hybrid_v2_2x)
  - Evaluates offline hybrid_v2 fusion variants at depth=20, rrf_k=60:
      * Plain RRF (1.0x)
      * Flat 2.0x Text-Dense Boost
      * Decayed Second-Vote Boost (decay = 2, 3, 4)
      * Protect Retriever Top-1 (RRF#1 + Text#1 + Visual#1)
  - Side-by-side Split Comparison (dev vs. test vs. all)
  - Bootstrapped 95% Confidence Intervals across live + offline shortlisted modes
========================================================================================
"""

import argparse
import csv
from collections import Counter, defaultdict
from io import StringIO
import json
import math
import os
from typing import Any, Dict, List, Tuple
import numpy as np


# ---------------------------------------------------------------------------
# Metric & RRF Fusion Utilities
# ---------------------------------------------------------------------------
def is_text_dense(ptype: str) -> bool:
    if not ptype:
        return False
    return str(ptype).lower().strip() in ("text-dense", "text", "text_dense", "txt")


def get_metric(d: Dict[Any, Any], key: Any, default: float = 0.0) -> float:
    """Safely retrieves metric by integer or string key."""
    if key in d:
        return float(d[key])
    str_k = str(key)
    if str_k in d:
        return float(d[str_k])
    try:
        int_k = int(key)
        if int_k in d:
            return float(d[int_k])
    except (ValueError, TypeError):
        pass
    return default


def compute_recalls_ndcgs_mrr(
    retrieved_pages: List[Tuple[str, int]],
    gold_set: set,
    k_vals: Tuple[int, ...] = (1, 3, 5, 10),
    max_k: int = 10,
) -> Tuple[Dict[int, float], Dict[int, float], float]:
    """Computes Recall@k, NDCG@k, and MRR@10 matching eval_harness.py."""
    recalls = {}
    ndcgs = {}
    for k in k_vals:
        top_k = retrieved_pages[:k]
        hits_in_k = [1 if p in gold_set else 0 for p in top_k]
        recalls[k] = sum(hits_in_k) / len(gold_set) if gold_set else 0.0

        dcg = sum((val / math.log2(idx + 2)) for idx, val in enumerate(hits_in_k))
        ideal_hits = [1] * min(len(gold_set), k)
        idcg = sum((val / math.log2(idx + 2)) for idx, val in enumerate(ideal_hits))
        ndcgs[k] = (dcg / idcg) if idcg > 0.0 else 0.0

    mrr = 0.0
    for rank_idx, p in enumerate(retrieved_pages[:max_k], start=1):
        if p in gold_set:
            mrr = 1.0 / rank_idx
            break

    return recalls, ndcgs, mrr


def rrf_merge_v2(
    t_pool: List[Any],
    v_pool: List[Any],
    rrf_k: int = 60,
    depth: int = 20,
    boost_mode: str = "none",  # "none", "flat_2x", "decayed"
    decay_factor: float = 2.0,
    protect_top1: bool = False,
) -> List[Tuple[str, int]]:
    """
    Offline RRF merge on hybrid_v2 pools that preserves exact insertion-order
    tie-breaking of retriever._fuse_rrf.
    """
    scores: Dict[Tuple[str, int], float] = {}
    t_sliced = t_pool[:depth]
    v_sliced = v_pool[:depth]

    for r_idx, h in enumerate(t_sliced, start=1):
        doc_id, page_num = str(h[0]), int(h[1])
        ptype = h[3] if len(h) > 3 else "layout-heavy"
        cand = (doc_id, page_num)
        base_vote = 1.0 / (rrf_k + r_idx)

        if is_text_dense(ptype):
            if boost_mode == "flat_2x":
                vote = 2.0 * base_vote
            elif boost_mode == "decayed":
                second_vote = 1.0 / (rrf_k + (decay_factor * r_idx))
                vote = base_vote + second_vote
            else:
                vote = base_vote
        else:
            vote = base_vote

        scores[cand] = scores.get(cand, 0.0) + vote

    for r_idx, h in enumerate(v_sliced, start=1):
        doc_id, page_num = str(h[0]), int(h[1])
        cand = (doc_id, page_num)
        scores[cand] = scores.get(cand, 0.0) + (1.0 / (rrf_k + r_idx))

    ranked_rrf = sorted(scores.keys(), key=lambda p: scores[p], reverse=True)

    if not protect_top1:
        return ranked_rrf

    final_top = []
    if ranked_rrf:
        final_top.append(ranked_rrf[0])
    if t_sliced:
        cand_t1 = (str(t_sliced[0][0]), int(t_sliced[0][1]))
        if cand_t1 not in final_top:
            final_top.append(cand_t1)
    if v_sliced:
        cand_v1 = (str(v_sliced[0][0]), int(v_sliced[0][1]))
        if cand_v1 not in final_top:
            final_top.append(cand_v1)

    for cand in ranked_rrf:
        if cand not in final_top:
            final_top.append(cand)

    return final_top


def synthesize_offline_mode_entries(
    v2_items: List[Dict[str, Any]],
    boost_mode: str = "none",
    decay_factor: float = 2.0,
    protect_top1: bool = False,
    rrf_k: int = 60,
    depth: int = 20,
) -> List[Dict[str, Any]]:
    """Builds synthetic detailed mode entries for offline hybrid_v2 variants."""
    synth_entries = []
    for item in v2_items:
        gold_set = {(str(p[0]), int(p[1])) for p in item["gold_pages"]}
        t_pool = item.get("pools", {}).get("text", [])
        v_pool = item.get("pools", {}).get("visual", [])

        ranked = rrf_merge_v2(
            t_pool=t_pool,
            v_pool=v_pool,
            rrf_k=rrf_k,
            depth=depth,
            boost_mode=boost_mode,
            decay_factor=decay_factor,
            protect_top1=protect_top1,
        )
        recalls, ndcgs, mrr = compute_recalls_ndcgs_mrr(ranked, gold_set)

        new_item = dict(item)
        new_item["recalls"] = recalls
        new_item["ndcgs"] = ndcgs
        new_item["mrr"] = mrr
        new_item["hits"] = [(doc_id, page_num, 0.0) for doc_id, page_num in ranked[:20]]
        synth_entries.append(new_item)
    return synth_entries


def summarize_mode_slice(items: List[Dict[str, Any]]) -> Dict[str, float]:
    """Computes Overall R@3, Text-Dense R@3, Layout-Heavy R@3, and MRR for a list of query items."""
    if not items:
        return {"n": 0, "n_txt": 0, "n_lay": 0, "r3_all": 0.0, "r3_txt": 0.0, "r3_lay": 0.0, "mrr": 0.0}

    txt_items = [i for i in items if i.get("gold_page_types") and is_text_dense(i["gold_page_types"][0])]
    lay_items = [i for i in items if not (i.get("gold_page_types") and is_text_dense(i["gold_page_types"][0]))]

    r3_all = float(np.mean([get_metric(i["recalls"], 3) for i in items])) * 100.0
    r3_txt = float(np.mean([get_metric(i["recalls"], 3) for i in txt_items])) * 100.0 if txt_items else 0.0
    r3_lay = float(np.mean([get_metric(i["recalls"], 3) for i in lay_items])) * 100.0 if lay_items else 0.0
    mrr = float(np.mean([i.get("mrr", 0.0) for i in items])) * 100.0

    return {
        "n": len(items),
        "n_txt": len(txt_items),
        "n_lay": len(lay_items),
        "r3_all": r3_all,
        "r3_txt": r3_txt,
        "r3_lay": r3_lay,
        "mrr": mrr,
    }


# ---------------------------------------------------------------------------
# Main Analysis Suite
# ---------------------------------------------------------------------------
def run_deep_analysis(json_path: str, out_dir: str = None, split_filter: str = "all"):
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"Results JSON not found at: {json_path}")

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    raw_detailed: Dict[str, List[Dict[str, Any]]] = dict(data["detailed"])
    live_modes = list(raw_detailed.keys())

    # Synthesize the offline Hybrid v2 shortlisted variants so they appear in all tables
    if "hybrid_v2" in raw_detailed:
        base_v2 = raw_detailed["hybrid_v2"]
        if "hybrid_v2_2x" not in raw_detailed:
            raw_detailed["v2_flat_2x (off)"] = synthesize_offline_mode_entries(
                base_v2, boost_mode="flat_2x"
            )
        raw_detailed["v2_decay_2 (off)"] = synthesize_offline_mode_entries(
            base_v2, boost_mode="decayed", decay_factor=2.0
        )
        raw_detailed["v2_decay_3 (off)"] = synthesize_offline_mode_entries(
            base_v2, boost_mode="decayed", decay_factor=3.0
        )
        raw_detailed["v2_decay_4 (off)"] = synthesize_offline_mode_entries(
            base_v2, boost_mode="decayed", decay_factor=4.0
        )
        raw_detailed["v2_protect_top1"] = synthesize_offline_mode_entries(
            base_v2, boost_mode="none", protect_top1=True
        )

    all_modes = list(raw_detailed.keys())

    # Apply split filter for main report sections (1..12)
    detailed: Dict[str, List[Dict[str, Any]]] = {}
    for m in all_modes:
        if split_filter == "all":
            detailed[m] = raw_detailed[m]
        else:
            detailed[m] = [q for q in raw_detailed[m] if q.get("split", "dev") == split_filter]

    buf = StringIO()

    def log(msg: str = ""):
        print(msg)
        buf.write(msg + "\n")

    queries_by_qid = defaultdict(dict)
    for mode in all_modes:
        for item in detailed[mode]:
            queries_by_qid[item["qid"]][mode] = item
    qids = list(queries_by_qid.keys())

    # =========================================================================
    # 0. SPLIT-CONDITIONED COMPARISON (DEV vs. TEST vs. ALL)
    # =========================================================================
    log("=" * 118)
    log(" 0. SPLIT-CONDITIONED COMPARISON (DEV vs. TEST vs. COMBINED ALL) AT DEPTH=20, RRF_K=60")
    log("=" * 118)

    for sp_label in ["dev", "test", "all"]:
        sp_items_first = (
            raw_detailed[all_modes[0]]
            if sp_label == "all"
            else [q for q in raw_detailed[all_modes[0]] if q.get("split", "dev") == sp_label]
        )
        if not sp_items_first:
            continue
        s_meta = summarize_mode_slice(sp_items_first)
        log(
            f"\n>>> SPLIT: {sp_label.upper()} "
            f"(Total N={s_meta['n']} | Text-Dense Gold={s_meta['n_txt']} | Layout-Heavy Gold={s_meta['n_lay']})"
        )
        log(f"{'Mode / Configuration':<24} | {'Overall R@3':<12} | {'Text-Dense R@3':<15} | {'Layout-Heavy R@3':<17} | {'MRR':<8}")
        log("-" * 88)
        for m in all_modes:
            m_items = (
                raw_detailed[m]
                if sp_label == "all"
                else [q for q in raw_detailed[m] if q.get("split", "dev") == sp_label]
            )
            stats = summarize_mode_slice(m_items)
            log(
                f"{m:<24} | {stats['r3_all']:>10.1f}% | "
                f"{stats['r3_txt']:>13.1f}% | "
                f"{stats['r3_lay']:>15.1f}% | "
                f"{stats['mrr']:>6.1f}%"
            )

    # =========================================================================
    # 1. OVERALL METRIC EXPANSION ACROSS ALL MODES (Filtered by --split)
    # =========================================================================
    log("\n" + "=" * 118)
    log(f" 1. OVERALL METRIC EXPANSION [SPLIT={split_filter.upper()}, N={len(qids)}] (R@1, R@3, R@5, R@10, NDCG, MRR)")
    log("=" * 118)

    k_keys = [1, 3, 5, 10]
    header = (
        f"{'Mode':<22} | "
        + " | ".join([f"R@{k:<2}" for k in k_keys])
        + " | "
        + " | ".join([f"NDCG@{k:<2}" for k in k_keys])
        + " | MRR"
    )
    log(header)
    log("-" * len(header))

    summary_csv_rows = []
    for mode in all_modes:
        items = detailed[mode]
        if not items:
            continue
        r_avgs = {k: np.mean([get_metric(item["recalls"], k) for item in items]) * 100 for k in k_keys}
        n_avgs = {k: np.mean([get_metric(item["ndcgs"], k) for item in items]) * 100 for k in k_keys}
        mrr_avg = np.mean([item.get("mrr", 0.0) for item in items]) * 100
        s_slice = summarize_mode_slice(items)

        summary_csv_rows.append({
            "split": split_filter,
            "mode": mode,
            "TextDense_Recall@3": round(s_slice["r3_txt"], 2),
            "LayoutHeavy_Recall@3": round(s_slice["r3_lay"], 2),
            **{f"Recall@{k}": round(r_avgs[k], 2) for k in k_keys},
            **{f"NDCG@{k}": round(n_avgs[k], 2) for k in k_keys},
            "MRR": round(mrr_avg, 2),
        })

        r_str = " | ".join([f"{r_avgs[k]:5.1f}%" for k in k_keys])
        n_str = " | ".join([f"{n_avgs[k]:6.1f}%" for k in k_keys])
        log(f"{mode:<22} | {r_str} | {n_str} | {mrr_avg:5.1f}%")

    # =========================================================================
    # 2. GOLD TARGET RANK HISTOGRAM
    # =========================================================================
    log("\n" + "=" * 118)
    log(f" 2. GOLD TARGET RANK HISTOGRAM (1-10 vs Not Found) [SPLIT={split_filter.upper()}]")
    log("=" * 118)

    for mode in all_modes:
        hist = Counter()
        for item in detailed[mode]:
            gold_set = {(str(p[0]), int(p[1])) for p in item["gold_pages"]}
            hit_rank = ">10"
            for rank_idx, h in enumerate(item["hits"][:10], start=1):
                if (str(h[0]), int(h[1])) in gold_set:
                    hit_rank = str(rank_idx)
                    break
            hist[hit_rank] += 1

        row_str = " ".join([f"R{r}:{hist[str(r)]:2d}" for r in range(1, 11)])
        log(f"{mode:<22} | {row_str} | Not Found: {hist['>10']:2d}")

    # =========================================================================
    # 3. DETAILED METRIC SLICES
    # =========================================================================
    log("\n" + "=" * 118)
    log(f" 3. DETAILED METRIC SLICES (R@3 & MRR) [SPLIT={split_filter.upper()}]")
    log("=" * 118)

    slice_fields = ["split", "evidence_type", "difficulty", "reasoning_type", "multi_page", "bucket"]
    display_modes = [m for m in ["text", "visual", "fusion_unrouted", "hybrid_v2", "hybrid_v2_2x", "v2_decay_3 (off)", "v2_decay_4 (off)", "v2_protect_top1"] if m in all_modes]

    for field in slice_fields:
        log(f"\n--- Slicing by: {field} ---")
        field_vals = sorted(list({str(item.get(field, "dev" if field == "split" else "unknown")) for item in detailed[all_modes[0]]}))
        col_headers = " | ".join([f"{m[:12]:<12}" for m in display_modes])
        log(f"{'Slice Value':<28} | N  | {col_headers}")
        log("-" * (36 + len(col_headers)))

        for val in field_vals:
            n_slice = sum(1 for item in detailed[all_modes[0]] if str(item.get(field, "dev" if field == "split" else "unknown")) == val)
            res_str = []
            for mode in display_modes:
                sub = [item for item in detailed[mode] if str(item.get(field, "dev" if field == "split" else "unknown")) == val]
                r3 = np.mean([get_metric(item["recalls"], 3) for item in sub]) * 100 if sub else 0.0
                mrr = np.mean([item.get("mrr", 0.0) for item in sub]) * 100 if sub else 0.0
                res_str.append(f"{r3:4.0f}%/{mrr:4.0f}% ")
            log(f"{val:<28} | {n_slice:2d} | " + " | ".join(res_str))

    # =========================================================================
    # 4. CROSS-TABULATION: GOLD PAGE TYPE vs EVIDENCE TYPE
    # =========================================================================
    log("\n" + "=" * 118)
    log(f" 4. CROSS-TABULATION: GOLD PAGE TYPE vs EVIDENCE TYPE [SPLIT={split_filter.upper()}]")
    log("=" * 118)

    crosstab = defaultdict(Counter)
    evidence_types, page_types = set(), set()

    for item in detailed[all_modes[0]]:
        ev = item.get("evidence_type", "unknown")
        pt = item["gold_page_types"][0] if item.get("gold_page_types") else "layout-heavy"
        crosstab[pt][ev] += 1
        evidence_types.add(ev)
        page_types.add(pt)

    ev_list = sorted(list(evidence_types))
    log(f"{'Page Type / Evidence':<25} | " + " | ".join([f"{e:<12}" for e in ev_list]) + " | Total")
    log("-" * 75)
    for pt in sorted(list(page_types)):
        counts = [crosstab[pt][e] for e in ev_list]
        log(f"{pt:<25} | " + " | ".join([f"{c:<12d}" for c in counts]) + f" | {sum(counts):<5d}")

    # =========================================================================
    # 5. OFFLINE FUSION & SHORTLISTED HYBRID_V2 ABLATION SUITE
    # =========================================================================
    if "hybrid_v2" in detailed and detailed["hybrid_v2"]:
        log("\n" + "=" * 118)
        log(f" 5. OFFLINE HYBRID_V2 MERGE & ABLATION SUITE (depth=20, rrf_k=60) [SPLIT={split_filter.upper()}]")
        log("=" * 118)

        v2_queries = detailed["hybrid_v2"]
        offline_variants = [
            ("1. Plain Hybrid v2 (1.0x, no boost)", "none", 2.0, False),
            ("2. Hybrid v2 + Flat 2.0x Text-Dense Boost", "flat_2x", 2.0, False),
            ("3. Hybrid v2 + Decayed Boost (decay=2)", "decayed", 2.0, False),
            ("4. Hybrid v2 + Decayed Boost (decay=3)", "decayed", 3.0, False),
            ("5. Hybrid v2 + Decayed Boost (decay=4)", "decayed", 4.0, False),
            ("6. Hybrid v2 + Protect Top-1 (RRF#1 + T#1 + V#1)", "none", 2.0, True),
        ]

        log(f"{'Offline Merge Variant':<50} | {'Overall R@3':<11} | {'Text-Dense':<11} | {'Layout-Heavy':<12} | {'MRR':<8}")
        log("-" * 104)
        for label, b_mode, dec, prot in offline_variants:
            synth = synthesize_offline_mode_entries(
                v2_queries, boost_mode=b_mode, decay_factor=dec, protect_top1=prot, rrf_k=60, depth=20
            )
            st = summarize_mode_slice(synth)
            log(
                f"{label:<50} | {st['r3_all']:>9.1f}% | "
                f"{st['r3_txt']:>9.1f}% | "
                f"{st['r3_lay']:>10.1f}% | "
                f"{st['mrr']:>6.1f}%"
            )

    # =========================================================================
    # 6. ERROR TAXONOMY: WRONG PAPER VS WRONG PAGE (TOP-10 CANDIDATES)
    # =========================================================================
    log("\n" + "=" * 118)
    log(f" 6. ERROR TAXONOMY: WRONG PAPER VS WRONG PAGE (TOP-10 CANDIDATES) [SPLIT={split_filter.upper()}]")
    log("=" * 118)

    for mode in all_modes:
        wrong_paper, wrong_page_same_paper, total_misses = 0, 0, 0
        for item in detailed[mode]:
            gold_set = {(str(p[0]), int(p[1])) for p in item["gold_pages"]}
            gold_docs = {str(p[0]) for p in item["gold_pages"]}
            hit_pages = {(str(h[0]), int(h[1])) for h in item["hits"][:10]}
            hit_docs = {str(h[0]) for h in item["hits"][:10]}

            if not any(p in gold_set for p in hit_pages):
                total_misses += 1
                if any(d in gold_docs for d in hit_docs):
                    wrong_page_same_paper += 1
                else:
                    wrong_paper += 1

        log(
            f"{mode:<22} | Misses@10: {total_misses:2d} | "
            f"Wrong Page (Same Paper): {wrong_page_same_paper:2d} | "
            f"Wrong Paper Entirely: {wrong_paper:2d}"
        )

    # =========================================================================
    # 7. BOOTSTRAPPED 95% CONFIDENCE INTERVALS (Overall, Text-Dense, Layout)
    # =========================================================================
    log("\n" + "=" * 118)
    log(f" 7. BOOTSTRAPPED 95% CONFIDENCE INTERVALS (1000 Iterations) [SPLIT={split_filter.upper()}]")
    log("=" * 118)

    rng = np.random.default_rng(42)
    n_boot = 1000

    def boot_ci_str(vals: np.ndarray) -> str:
        if len(vals) == 0:
            return "  0.0% [  0.0%,   0.0%]"
        means = [np.mean(rng.choice(vals, size=len(vals), replace=True)) * 100 for _ in range(n_boot)]
        return f"{np.mean(vals)*100:5.1f}% [{np.percentile(means, 2.5):5.1f}%, {np.percentile(means, 97.5):5.1f}%]"

    for mode in all_modes:
        items = detailed[mode]
        if not items:
            continue
        r3_all = np.array([get_metric(i["recalls"], 3) for i in items])
        r3_txt = np.array([get_metric(i["recalls"], 3) for i in items if i.get("gold_page_types") and is_text_dense(i["gold_page_types"][0])])
        r3_lay = np.array([get_metric(i["recalls"], 3) for i in items if not (i.get("gold_page_types") and is_text_dense(i["gold_page_types"][0]))])
        mrr_vals = np.array([i.get("mrr", 0.0) for i in items])

        log(
            f"{mode:<22} | Overall R@3: {boot_ci_str(r3_all)} | "
            f"Txt R@3: {boot_ci_str(r3_txt)} | "
            f"Lay R@3: {boot_ci_str(r3_lay)} | "
            f"MRR: {boot_ci_str(mrr_vals)}"
        )

    # =========================================================================
    # 8. LATENCY SLICES (LIVE MODES ONLY)
    # =========================================================================
    log("\n" + "=" * 118)
    log(f" 8. LATENCY SLICES (p50 / p95 in ms by Evidence Type) [SPLIT={split_filter.upper()}]")
    log("=" * 118)

    for mode in live_modes:
        log(f"Mode: {mode}")
        by_ev = defaultdict(list)
        for item in detailed[mode]:
            by_ev[item.get("evidence_type", "unknown")].append(item["latency_sec"] * 1000)
        for ev, lats in sorted(by_ev.items()):
            log(f"    {ev:<10} (N={len(lats):2d}) | p50: {np.percentile(lats, 50):6.1f} ms | p95: {np.percentile(lats, 95):6.1f} ms")

    # =========================================================================
    # 9. PER-QUERY TEXT-DENSE DIAGNOSTIC TABLE (Comparing v2 Variants)
    # =========================================================================
    if "hybrid_v2" in detailed:
        log("\n" + "=" * 118)
        log(f" 9. TEXT-DENSE QUERIES RANK AUDIT ACROSS V2 VARIANTS [SPLIT={split_filter.upper()}]")
        log("=" * 118)
        log(f"{'QID':<7} | {'Split':<5} | {'Text#':<6} | {'Vis#':<6} | {'plain_v2':<9} | {'flat_2x':<8} | {'decay_3':<8} | {'decay_4':<8} | {'protect_1':<9}")
        log("-" * 92)

        for qid in qids:
            q_item = queries_by_qid[qid]["hybrid_v2"]
            if not (q_item.get("gold_page_types") and is_text_dense(q_item["gold_page_types"][0])):
                continue

            gold_set = {(str(p[0]), int(p[1])) for p in q_item["gold_pages"]}
            t_pool = q_item.get("pools", {}).get("text", [])
            v_pool = q_item.get("pools", {}).get("visual", [])

            def find_rank(pool_or_ranked) -> str:
                for idx, h in enumerate(pool_or_ranked, start=1):
                    cand = (str(h[0]), int(h[1]))
                    if cand in gold_set:
                        return f"#{idx}"
                return ">20"

            r_t = find_rank(t_pool)
            r_v = find_rank(v_pool)
            r_plain = find_rank(rrf_merge_v2(t_pool, v_pool, 60, 20, "none"))
            r_2x = find_rank(rrf_merge_v2(t_pool, v_pool, 60, 20, "flat_2x"))
            r_d3 = find_rank(rrf_merge_v2(t_pool, v_pool, 60, 20, "decayed", 3.0))
            r_d4 = find_rank(rrf_merge_v2(t_pool, v_pool, 60, 20, "decayed", 4.0))
            r_prot = find_rank(rrf_merge_v2(t_pool, v_pool, 60, 20, "none", protect_top1=True))

            log(
                f"{qid:<7} | {q_item.get('split', 'dev'):<5} | {r_t:<6} | {r_v:<6} | "
                f"{r_plain:<9} | {r_2x:<8} | {r_d3:<8} | {r_d4:<8} | {r_prot:<9}"
            )

    # =========================================================================
    # Artifact Exporter
    # =========================================================================
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
        suffix = f"_{split_filter}" if split_filter != "all" else ""
        report_file = os.path.join(out_dir, f"v3_deep_analysis_report{suffix}.md")
        with open(report_file, "w", encoding="utf-8") as rf:
            rf.write(f"# Offline Retrieval Ablation Deep Analysis V3 (Split: {split_filter})\n\n")
            rf.write("```text\n")
            rf.write(buf.getvalue())
            rf.write("```\n")
        print(f"\n[Saved] Analytical report: {report_file}")

        if summary_csv_rows:
            csv_file = os.path.join(out_dir, f"v3_ablation_metrics_summary{suffix}.csv")
            with open(csv_file, "w", newline="", encoding="utf-8") as cf:
                writer = csv.DictWriter(cf, fieldnames=summary_csv_rows[0].keys())
                writer.writeheader()
                writer.writerows(summary_csv_rows)
            print(f"[Saved] CSV Metrics Summary: {csv_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="V3 Offline Deep Retrieval & Split Analysis")
    parser.add_argument(
        "--file",
        type=str,
        default="evaluation/results_v3/benchmark_results.json",
        help="Path to benchmark_results.json",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default="evaluation/results_v3",
        help="Directory to save report and CSV artifacts",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="all",
        choices=["all", "dev", "test"],
        help="Filter detailed sections by split (Section 0 always compares dev, test, and all)",
    )
    args = parser.parse_args()
    run_deep_analysis(args.file, args.out_dir, args.split)