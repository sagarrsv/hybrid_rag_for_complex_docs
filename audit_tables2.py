#!/usr/bin/env python3
"""
audit_tables.py
========================================================================================
Audits and generates rich Markdown/HTML inspection tables directly from 
evaluation/results_v2/benchmark_results.json:
  1. Table 1: Complete audit of the 9 Text-Dense Gold Queries
  2. Table 2: Complete audit of Layout-Heavy queries degraded into False Positives by 2.0x
========================================================================================
"""
import os
import json
import argparse
from collections import defaultdict
from typing import Dict, List, Tuple, Any

try:
    from tabulate import tabulate
    HAS_TABULATE = True
except ImportError:
    HAS_TABULATE = False


def score_rrf(t_pool: List[Any], v_pool: List[Any], text_dense_mult: float = 1.0, rrf_k: int = 60) -> List[Tuple[str, int]]:
    """Calculates reciprocal rank fusion scores across all candidate pairs."""
    scores = defaultdict(float)
    
    for r_idx, h in enumerate(t_pool, start=1):
        doc_id, page_num = str(h[0]), int(h[1])
        ptype = h[3] if len(h) > 3 else "layout-heavy"
        weight = text_dense_mult if ptype == "text-dense" else 1.0
        scores[(doc_id, page_num)] += weight * (1.0 / (rrf_k + r_idx))
        
    for r_idx, h in enumerate(v_pool, start=1):
        doc_id, page_num = str(h[0]), int(h[1])
        scores[(doc_id, page_num)] += 1.0 * (1.0 / (rrf_k + r_idx))
        
    return sorted(scores.keys(), key=lambda p: scores[p], reverse=True)


def get_gold_rank(ranked_pages: List[Tuple[str, int]], gold_set: set) -> str:
    for idx, p in enumerate(ranked_pages, start=1):
        if p in gold_set:
            return str(idx)
    return ">20"


def format_text_pool_top(pool: List[Any], gold_set: set, max_items: int = 5) -> str:
    """Formats top-k text pool with (layout)/(text) and (gold) tags."""
    items = []
    for idx, h in enumerate(pool[:max_items], start=1):
        doc_id, page_num = str(h[0]), int(h[1])
        ptype = h[3] if len(h) > 3 else "layout-heavy"
        tag = "(text)" if ptype == "text-dense" else "(layout)"
        gold_tag = " [GOLD]" if (doc_id, page_num) in gold_set else ""
        doc_abbrev = doc_id.split("_")[0] if "_" in doc_id else doc_id[:12]
        items.append(f"{idx}. {doc_abbrev}:p{page_num} {tag}{gold_tag}")
    return "\n".join(items)


def format_visual_pool_top(pool: List[Any], gold_set: set, max_items: int = 5) -> str:
    """Formats top-k visual pool with (gold) tags."""
    items = []
    for idx, h in enumerate(pool[:max_items], start=1):
        doc_id, page_num = str(h[0]), int(h[1])
        gold_tag = " [GOLD]" if (doc_id, page_num) in gold_set else ""
        doc_abbrev = doc_id.split("_")[0] if "_" in doc_id else doc_id[:12]
        items.append(f"{idx}. {doc_abbrev}:p{page_num}{gold_tag}")
    return "\n".join(items)


def check_consensus(t_pool: List[Any], v_pool: List[Any], top_n: int = 20) -> Tuple[bool, List[str]]:
    """Checks for page overlaps between top candidates across both retrievers."""
    t_set = set((str(h[0]), int(h[1])) for h in t_pool[:top_n])
    v_set = set((str(h[0]), int(h[1])) for h in v_pool[:top_n])
    common = list(t_set.intersection(v_set))
    common_labels = [f"{c[0].split('_')[0]}:p{c[1]}" for c in common]
    return len(common) > 0, common_labels


def run_audit(results_path: str, out_dir: str = "evaluation/results_v2"):
    if not os.path.exists(results_path):
        raise FileNotFoundError(f"Cannot find results file at: {results_path}")
        
    with open(results_path, "r", encoding="utf-8") as f:
        data = json.load(f)
        
    queries = data["detailed"]["hybrid_v2"]
    os.makedirs(out_dir, exist_ok=True)

    # =========================================================================
    # 1. TABLE 1: 9 TEXT-DENSE GOLD QUERIES AUDIT
    # =========================================================================
    text_dense_q = [
        q for q in queries 
        if q.get("gold_page_types") and q["gold_page_types"][0] == "text-dense"
    ]
    
    t1_headers = [
        "QID", "Gold Target", "Base Rank", "2.0x Rank", 
        "Dual Consensus(top-20)?", "Top-5 Text Pool (with ptype & gold)", "Top-5 Visual Pool"
    ]
    t1_rows = []

    for q in text_dense_q:
        qid = q["qid"]
        gold_set = {(str(g[0]), int(g[1])) for g in q["gold_pages"]}
        gold_str = ", ".join([f"{str(g[0]).split('_')[0]}:p{g[1]}" for g in q["gold_pages"]])
        
        t_pool = q.get("pools", {}).get("text", [])
        v_pool = q.get("pools", {}).get("visual", [])
        
        base_ranked = score_rrf(t_pool, v_pool, text_dense_mult=1.0)
        boosted_ranked = score_rrf(t_pool, v_pool, text_dense_mult=2.0)
        
        r_base = get_gold_rank(base_ranked, gold_set)
        r_boost = get_gold_rank(boosted_ranked, gold_set)
        
        has_overlap, common = check_consensus(t_pool, v_pool, top_n=20)
        has_overlap, common = check_consensus(t_pool, v_pool, top_n=20)
        if has_overlap:
            top_samples = common[:2]  # Show at most 2 matches
            extra = len(common) - len(top_samples)
            consensus_str = f"YES ({', '.join(top_samples)} +{extra})" if extra > 0 else f"YES ({', '.join(top_samples)})"
        else:
            consensus_str = "NO"
        
        t_str = format_text_pool_top(t_pool, gold_set, max_items=5)
        v_str = format_visual_pool_top(v_pool, gold_set, max_items=5)
        
        t1_rows.append([
            qid, gold_str, f"Rank {r_base}", f"Rank {r_boost}",
            consensus_str, t_str, v_str
        ])

    # =========================================================================
    # 2. TABLE 2: LAYOUT DEGRADATION AUDIT (FALSE POSITIVES FROM 2.0x)
    # =========================================================================
    layout_q = [
        q for q in queries 
        if q.get("gold_page_types") and q["gold_page_types"][0] == "layout-heavy"
    ]
    
    t2_headers = [
        "QID", "Gold Layout Target", "Base Rank", "2.0x Rank", 
        "Displacing Text-Dense Chunks", "Top-5 Text Pool (with ptype & gold)", "Top-5 Visual Pool"
    ]
    t2_rows = []

    for q in layout_q:
        gold_set = {(str(g[0]), int(g[1])) for g in q["gold_pages"]}
        t_pool = q.get("pools", {}).get("text", [])
        v_pool = q.get("pools", {}).get("visual", [])
        
        base_ranked = score_rrf(t_pool, v_pool, text_dense_mult=1.0)
        boosted_ranked = score_rrf(t_pool, v_pool, text_dense_mult=2.0)
        
        hit_base = any(p in gold_set for p in base_ranked[:3])
        hit_boost = any(p in gold_set for p in boosted_ranked[:3])
        
        # Survived in Top-3 under baseline, but dropped in 2.0x
        if hit_base and not hit_boost:
            r_base = get_gold_rank(base_ranked, gold_set)
            r_boost = get_gold_rank(boosted_ranked, gold_set)
            gold_str = ", ".join([f"{str(g[0]).split('_')[0]}:p{g[1]}" for g in q["gold_pages"]])
            
            top3_boosted = boosted_ranked[:3]
            intruding = []
            for cand in top3_boosted:
                for r_in_t, h in enumerate(t_pool, start=1):
                    if (str(h[0]), int(h[1])) == cand and h[3] == "text-dense":
                        doc_abbrev = str(h[0]).split('_')[0]
                        intruding.append(f"{doc_abbrev}:p{h[1]} (TextRank #{r_in_t})")
            
            displacers_str = "\n".join(intruding) if intruding else "Rank alternation"
            t_str = format_text_pool_top(t_pool, gold_set, max_items=5)
            v_str = format_visual_pool_top(v_pool, gold_set, max_items=5)
            
            t2_rows.append([
                q["qid"], gold_str, f"Rank {r_base}", f"Rank {r_boost}",
                displacers_str, t_str, v_str
            ])

    # =========================================================================
    # 3. PRINT TERMINAL & EXPORT ARTIFACTS
    # =========================================================================
    if HAS_TABULATE:
        print("\n" + "=" * 140)
        print(" TABLE 1: 9 TEXT-DENSE GOLD QUERIES (HYBRID V2 BASELINE vs. 2.0x MERGE)")
        print("=" * 140)
        print(tabulate(t1_rows, headers=t1_headers, tablefmt="grid"))
        
        print("\n" + "=" * 140)
        print(f" TABLE 2: LAYOUT-HEAVY DEGRADATION AUDIT (DROPPED FROM TOP-3 UNDER 2.0x) [N={len(t2_rows)}]")
        print("=" * 140)
        print(tabulate(t2_rows, headers=t2_headers, tablefmt="grid"))
    else:
        # Fallback simple print if tabulate is not installed
        for title, headers, rows in [("TABLE 1", t1_headers, t1_rows), ("TABLE 2", t2_headers, t2_rows)]:
            print(f"\n--- {title} ---")
            for r in rows:
                print(f"[{r[0]}] Target: {r[1]} | Base: {r[2]} -> 2x: {r[3]}")
                print(f"  Top Text:\n    " + r[5].replace('\n', '\n    '))
                print(f"  Top Visual:\n    " + r[6].replace('\n', '\n    '))

    # Save to Markdown file
    md_path = os.path.join(out_dir, "audit_pool_comparison.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("# Multimodal RAG Hybrid V2 Pool Audit\n\n")
        f.write("## Table 1: 9 Text-Dense Gold Queries Audit\n\n")
        if HAS_TABULATE:
            f.write(tabulate(t1_rows, headers=t1_headers, tablefmt="pipe"))
        f.write(f"\n\n## Table 2: Layout Degradation Audit (False Positives from 2.0x) [N={len(t2_rows)}]\n\n")
        if HAS_TABULATE:
            f.write(tabulate(t2_rows, headers=t2_headers, tablefmt="pipe"))
    print(f"\n[Artifact Saved] Markdown table saved to: {md_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Audit Hybrid v2 Merge Tables")
    parser.add_argument(
        "--file", 
        type=str, 
        default="evaluation/results_v2/benchmark_results.json",
        help="Path to evaluation benchmark_results.json"
    )
    parser.add_argument(
        "--out_dir", 
        type=str, 
        default="evaluation/results_v2",
        help="Directory to save generated markdown audit tables"
    )
    args = parser.parse_args()
    run_audit(args.file, args.out_dir)