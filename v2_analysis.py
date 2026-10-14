#!/usr/bin/env python3
import argparse
import csv
from collections import Counter, defaultdict
from io import StringIO
import json
import os
from typing import Any, Dict, List, Tuple
import numpy as np


# ---------------------------------------------------------------------------
# Metric Calculation Utilities
# ---------------------------------------------------------------------------
def compute_recalls_and_mrr(
    retrieved_pages: List[Tuple[str, int]],
    gold_set: set,
    k_vals: Tuple[int, ...] = (1, 3, 5, 10),
) -> Tuple[Dict[int, float], float]:
  """Computes Recall@k and MRR over a ranked list of (doc_id, page_num) pairs."""
  recalls = {}
  for k in k_vals:
    top_k = retrieved_pages[:k]
    hits = sum(1 for p in top_k if p in gold_set)
    recalls[k] = hits / len(gold_set) if gold_set else 0.0

  mrr = 0.0
  for rank_idx, p in enumerate(retrieved_pages, start=1):
    if p in gold_set:
      mrr = 1.0 / rank_idx
      break
  return recalls, mrr


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


# ---------------------------------------------------------------------------
# Main Analysis Suite
# ---------------------------------------------------------------------------
def run_deep_analysis(json_path: str, out_dir: str = None):
  if not os.path.exists(json_path):
    raise FileNotFoundError(f"Results JSON not found at: {json_path}")

  with open(json_path, "r", encoding="utf-8") as f:
    data = json.load(f)

  detailed: Dict[str, List[Dict[str, Any]]] = data["detailed"]
  modes = list(detailed.keys())

  buf = StringIO()

  def log(msg: str = ""):
    print(msg)
    buf.write(msg + "\n")

  queries_by_qid = defaultdict(dict)
  for mode in modes:
    for item in detailed[mode]:
      queries_by_qid[item["qid"]][mode] = item

  qids = list(queries_by_qid.keys())

  # =========================================================================
  # 1. OVERALL METRIC EXPANSION ACROSS ALL ACTIVE MODES
  # =========================================================================
  log("=" * 102)
  log(
      " 1. OVERALL METRIC EXPANSION ACROSS ALL MODES (R@1, R@3, R@5, R@10, NDCG,"
      " MRR)"
  )
  log("=" * 102)
  k_keys = [1, 3, 5, 10]
  header = (
      f"{'Mode':<18} | "
      + " | ".join([f"R@{k}" for k in k_keys])
      + " | "
      + " | ".join([f"NDCG@{k}" for k in k_keys])
      + " | MRR"
  )
  log(header)
  log("-" * len(header))

  summary_csv_rows = []
  for mode in modes:
    items = detailed[mode]
    r_avgs = {
        k: np.mean([get_metric(item["recalls"], k) for item in items]) * 100
        for k in k_keys
    }
    n_avgs = {
        k: np.mean([get_metric(item["ndcgs"], k) for item in items]) * 100
        for k in k_keys
    }
    mrr_avg = np.mean([item.get("mrr", 0.0) for item in items]) * 100

    summary_csv_rows.append({
        "mode": mode,
        **{f"Recall@{k}": round(r_avgs[k], 2) for k in k_keys},
        **{f"NDCG@{k}": round(n_avgs[k], 2) for k in k_keys},
        "MRR": round(mrr_avg, 2),
    })

    r_str = " | ".join([f"{r_avgs[k]:5.1f}%" for k in k_keys])
    n_str = " | ".join([f"{n_avgs[k]:6.1f}%" for k in k_keys])
    log(f"{mode:<18} | {r_str} | {n_str} | {mrr_avg:5.1f}%")

  # =========================================================================
  # 2. GOLD TARGET RANK HISTOGRAM
  # =========================================================================
  log("\n" + "=" * 102)
  log(" 2. GOLD TARGET RANK HISTOGRAM (1-10 vs Not Found)")
  log("=" * 102)
  for mode in modes:
    hist = Counter()
    for item in detailed[mode]:
      gold_set = {(str(p[0]), int(p[1])) for p in item["gold_pages"]}
      hit_rank = ">10"
      for rank_idx, h in enumerate(item["hits"], start=1):
        if (str(h[0]), int(h[1])) in gold_set:
          hit_rank = str(rank_idx)
          break
      hist[hit_rank] += 1

    row_str = " ".join([f"R{r}:{hist[str(r)]:2d}" for r in range(1, 11)])
    log(f"{mode:<18} | {row_str} | Not Found: {hist['>10']:2d}")

  # =========================================================================
  # 3. DETAILED METRIC SLICES
  # =========================================================================
  log("\n" + "=" * 102)
  log(" 3. DETAILED METRIC SLICES (R@3 & MRR ACROSS ALL ACTIVE MODES)")
  log("=" * 102)
  slice_fields = [
      "evidence_type",
      "difficulty",
      "reasoning_type",
      "multi_page",
      "bucket",
  ]

  for field in slice_fields:
    log(f"\n--- Slicing by: {field} ---")
    field_vals = sorted(
        list({str(item.get(field)) for item in detailed[modes[0]]})
    )
    col_headers = " | ".join([f"{m[:7]} R@3 / MRR" for m in modes])
    log(f"{'Slice Value':<35} | N  | {col_headers}")
    log("-" * (42 + len(col_headers)))
    for val in field_vals:
      n_slice = sum(
          1 for item in detailed[modes[0]] if str(item.get(field)) == val
      )
      res_str = []
      for mode in modes:
        sub = [item for item in detailed[mode] if str(item.get(field)) == val]
        r3 = (
            np.mean([get_metric(item["recalls"], 3) for item in sub]) * 100
            if sub
            else 0.0
        )
        mrr = np.mean([item.get("mrr", 0.0) for item in sub]) * 100 if sub else 0.0
        res_str.append(f"{r3:4.0f}%/{mrr:4.0f}%")
      log(f"{val:<35} | {n_slice:2d} | " + " | ".join(res_str))

  # =========================================================================
  # 4. CROSS-TABULATION: GOLD PAGE TYPE vs EVIDENCE TYPE
  # =========================================================================
  log("\n" + "=" * 102)
  log(" 4. CROSS-TABULATION: GOLD PAGE TYPE vs EVIDENCE TYPE")
  log("=" * 102)
  crosstab = defaultdict(Counter)
  evidence_types = set()
  page_types = set()
  for item in detailed[modes[0]]:
    ev = item.get("evidence_type", "unknown")
    pt = (
        item["gold_page_types"][0]
        if item.get("gold_page_types")
        else "layout-heavy"
    )
    crosstab[pt][ev] += 1
    evidence_types.add(ev)
    page_types.add(pt)

  ev_list = sorted(list(evidence_types))
  header_title = "Page Type / Evidence"
  log(
      f"{header_title:<25} | "
      + " | ".join([f"{e:<12}" for e in ev_list])
      + " | Total"
  )
  log("-" * 75)
  for pt in sorted(list(page_types)):
    counts = [crosstab[pt][e] for e in ev_list]
    log(
        f"{pt:<25} | "
        + " | ".join([f"{c:<12d}" for c in counts])
        + f" | {sum(counts):<5d}"
    )

  # =========================================================================
  # 5. OFFLINE FUSION & RE-RANKING EXPERIMENTS
  # =========================================================================
  log("\n" + "=" * 102)
  log(" 5. OFFLINE FUSION EXPERIMENTS ON SAVED CANDIDATE POOLS")
  log("=" * 102)

  def get_arm_pools(qid: str, mode: str):
    """Extracts candidate pools saved in JSON, with fallback to hits."""
    arm = queries_by_qid[qid].get(mode, {})
    pools = arm.get("pools", {})
    t_pool = (
        pools.get("text", [])
        if pools
        else queries_by_qid[qid]["text"].get("hits", [])
    )
    v_pool = (
        pools.get("visual", [])
        if pools
        else queries_by_qid[qid]["visual"].get("hits", [])
    )
    return t_pool, v_pool

  def run_eval_sweep(
      name: str,
      fused_scores_dict: Dict[str, Dict[Tuple[str, int], float]],
      mode_key: str,
  ):
    r1, r3, r5, mrr = [], [], [], []
    txt_r3, lay_r3 = [], []

    for qid in qids:
      arm_data = queries_by_qid[qid][mode_key]
      gold_set = {(str(p[0]), int(p[1])) for p in arm_data["gold_pages"]}
      is_text = (
          arm_data["gold_page_types"]
          and arm_data["gold_page_types"][0] == "text-dense"
      )

      scores = fused_scores_dict[qid]
      ranked = sorted(scores.keys(), key=lambda p: scores[p], reverse=True)

      recs, m = compute_recalls_and_mrr(ranked, gold_set)
      r1.append(recs[1])
      r3.append(recs[3])
      r5.append(recs[5])
      mrr.append(m)

      if is_text:
        txt_r3.append(recs[3])
      else:
        lay_r3.append(recs[3])

    mean_r3 = np.mean(r3) * 100
    mean_mrr = np.mean(mrr) * 100
    mean_txt = np.mean(txt_r3) * 100 if txt_r3 else 0.0
    mean_lay = np.mean(lay_r3) * 100 if lay_r3 else 0.0

    log(
        f"  • {name:<52} | R@3: {mean_r3:5.1f}% (Txt: {mean_txt:4.1f}% | Lay:"
        f" {mean_lay:4.1f}%) | MRR: {mean_mrr:5.1f}%"
    )

  # -------------------------------------------------------------------------
  # PART 1: The 6 Original Merges on HYBRID_ROUTED (Strictly Disjoint Pools)
  # -------------------------------------------------------------------------
  log(
      "\n>>> PART 1: 6 Original Merges on 'hybrid_routed' (Text-Dense only +"
      " Layout-Heavy only):"
  )

  h_routed_r3 = (
      np.mean([
          get_metric(queries_by_qid[q]["hybrid_routed"]["recalls"], 3)
          for q in qids
      ])
      * 100
  )
  h_routed_mrr = (
      np.mean(
          [queries_by_qid[q]["hybrid_routed"].get("mrr", 0.0) for q in qids]
      )
      * 100
  )
  log(
      f"  [Harness In-Run Baseline: hybrid_routed]          | R@3:"
      f" {h_routed_r3:5.1f}% | MRR: {h_routed_mrr:5.1f}%\n"
  )

  original_configs = [
      ("Equal RRF (k=60)", 1.0, 1.0, 60),
      ("Equal RRF (k=20)", 1.0, 1.0, 20),
      ("Visual-Weighted RRF (Vis=1.5, Txt=1.0, k=60)", 1.0, 1.5, 60),
      ("Visual-Weighted RRF (Vis=2.0, Txt=1.0, k=60)", 1.0, 2.0, 60),
      ("Text-Weighted RRF (Vis=1.0, Txt=1.5, k=60)", 1.5, 1.0, 60),
  ]

  for label, wt, wv, rrf_k in original_configs:
    fused_routed = {}
    for qid in qids:
      t_pool, v_pool = get_arm_pools(qid, "hybrid_routed")
      scores = defaultdict(float)
      for r, h in enumerate(t_pool, 1):
        scores[(str(h[0]), int(h[1]))] += wt * (1.0 / (rrf_k + r))
      for r, h in enumerate(v_pool, 1):
        scores[(str(h[0]), int(h[1]))] += wv * (1.0 / (rrf_k + r))
      fused_routed[qid] = scores
    run_eval_sweep(label, fused_routed, "hybrid_routed")

  # Min-Max Normalization Merge on hybrid_routed
  min_max_routed = {}
  for qid in qids:
    t_pool, v_pool = get_arm_pools(qid, "hybrid_routed")
    scores = defaultdict(float)
    t_vals = [float(h[2]) for h in t_pool] if t_pool else [0.0]
    v_vals = [float(h[2]) for h in v_pool] if v_pool else [0.0]

    def scale_scores(vals):
      mn, mx = min(vals), max(vals)
      return [(v - mn) / (mx - mn + 1e-6) for v in vals]

    for h, s in zip(t_pool, scale_scores(t_vals)):
      scores[(str(h[0]), int(h[1]))] = max(scores[(str(h[0]), int(h[1]))], s)
    for h, s in zip(v_pool, scale_scores(v_vals)):
      scores[(str(h[0]), int(h[1]))] = max(scores[(str(h[0]), int(h[1]))], s)
    min_max_routed[qid] = scores
  run_eval_sweep(
      "Min-Max Score Normalization Merge", min_max_routed, "hybrid_routed"
  )

  # -------------------------------------------------------------------------
  # PART 2: Partition-Aware & Gated Merges on HYBRID_V2
  # -------------------------------------------------------------------------
  if "hybrid_v2" in modes:
    log(
        "\n>>> PART 2: Partition-Aware & Gated Merges on 'hybrid_v2' (All text"
        " + Layout-only visual):"
    )

    h_v2_r3 = (
        np.mean([
            get_metric(queries_by_qid[q]["hybrid_v2"]["recalls"], 3)
            for q in qids
        ])
        * 100
    )
    h_v2_mrr = (
        np.mean([queries_by_qid[q]["hybrid_v2"].get("mrr", 0.0) for q in qids])
        * 100
    )
    log(
        f"  [Harness In-Run Baseline: hybrid_v2]              | R@3:"
        f" {h_v2_r3:5.1f}% | MRR: {h_v2_mrr:5.1f}%\n"
    )

    # 1. Partition-aware merge: Compensate single-arm text-dense pages
    for boost in [1.5, 2.0, 2.2, 2.5]:
      fused_v2 = {}
      for qid in qids:
        t_pool, v_pool = get_arm_pools(qid, "hybrid_v2")
        scores = defaultdict(float)
        for r, h in enumerate(t_pool, 1):
          ptype = h[3] if len(h) > 3 else "layout-heavy"
          w = boost if ptype == "text-dense" else 1.0
          scores[(str(h[0]), int(h[1]))] += w * (1.0 / (60 + r))
        for r, h in enumerate(v_pool, 1):
          scores[(str(h[0]), int(h[1]))] += 1.0 * (1.0 / (60 + r))
        fused_v2[qid] = scores
      run_eval_sweep(
          f"Partition-Aware Hybrid-v2 (Text-Dense Boost={boost}x)",
          fused_v2,
          "hybrid_v2",
      )

    # 2. Global text weight sweeps on hybrid_v2
    for w_t in [1.2, 1.5, 1.8, 2.0]:
      fused_wt = {}
      for qid in qids:
        t_pool, v_pool = get_arm_pools(qid, "hybrid_v2")
        scores = defaultdict(float)
        for r, h in enumerate(t_pool, 1):
          scores[(str(h[0]), int(h[1]))] += w_t * (1.0 / (60 + r))
        for r, h in enumerate(v_pool, 1):
          scores[(str(h[0]), int(h[1]))] += 1.0 * (1.0 / (60 + r))
        fused_wt[qid] = scores
      run_eval_sweep(
          f"Text-Weighted RRF on v2 (w_text={w_t}, w_vis=1.0)",
          fused_wt,
          "hybrid_v2",
      )

    # 3. Confidence score-gap gating on hybrid_v2
    for gap in [0.025, 0.035, 0.045]:
      fused_gate = {}
      for qid in qids:
        t_pool, v_pool = get_arm_pools(qid, "hybrid_v2")
        scores = defaultdict(float)
        has_gap = len(t_pool) >= 2 and (
            float(t_pool[0][2]) - float(t_pool[1][2])
        ) >= gap
        for r, h in enumerate(t_pool, 1):
          bonus = 0.04 if (has_gap and r == 1) else 0.0
          scores[(str(h[0]), int(h[1]))] += (1.0 / (60 + r)) + bonus
        for r, h in enumerate(v_pool, 1):
          scores[(str(h[0]), int(h[1]))] += 1.0 / (60 + r)
        fused_gate[qid] = scores
      run_eval_sweep(
          f"Confidence Gated on v2 (Text Gap >= {gap:.3f})",
          fused_gate,
          "hybrid_v2",
      )

    # 4. Optimal compound fusion on hybrid_v2
    compound_all = {}
    for qid in qids:
      t_pool, v_pool = get_arm_pools(qid, "hybrid_v2")
      scores = defaultdict(float)
      has_gap = len(t_pool) >= 2 and (
          float(t_pool[0][2]) - float(t_pool[1][2])
      ) >= 0.035
      for r, h in enumerate(t_pool, 1):
        ptype = h[3] if len(h) > 3 else "layout-heavy"
        w = 2.0 if ptype == "text-dense" else 1.0
        bonus = 0.035 if (has_gap and r == 1) else 0.0
        scores[(str(h[0]), int(h[1]))] += (w * (1.0 / (60 + r))) + bonus
      for r, h in enumerate(v_pool, 1):
        scores[(str(h[0]), int(h[1]))] += 1.0 / (60 + r)
      compound_all[qid] = scores
    run_eval_sweep(
        "Compound Fusion on v2 (2.0x Boost + Gap Gating)",
        compound_all,
        "hybrid_v2",
    )

  # =========================================================================
  # 6. ERROR TAXONOMY: WRONG PAPER VS WRONG PAGE
  # =========================================================================
  log("\n" + "=" * 102)
  log(" 6. ERROR TAXONOMY: WRONG PAPER VS WRONG PAGE (TOP-10 CANDIDATES)")
  log("=" * 102)
  for mode in modes:
    wrong_paper = 0
    wrong_page_same_paper = 0
    total_misses = 0

    for item in detailed[mode]:
      gold_set = {(str(p[0]), int(p[1])) for p in item["gold_pages"]}
      gold_docs = {str(p[0]) for p in item["gold_pages"]}
      hit_pages = {(str(h[0]), int(h[1])) for h in item["hits"]}
      hit_docs = {str(h[0]) for h in item["hits"]}

      if not any(p in gold_set for p in hit_pages):
        total_misses += 1
        if any(d in gold_docs for d in hit_docs):
          wrong_page_same_paper += 1
        else:
          wrong_paper += 1

    log(
        f"{mode:<18} | Misses@10: {total_misses:2d} | Wrong Page (Same Paper):"
        f" {wrong_page_same_paper:2d} | Wrong Paper Entirely: {wrong_paper:2d}"
    )

  # =========================================================================
  # 7. DOCUMENT CONCENTRATION
  # =========================================================================
  log("\n" + "=" * 102)
  log(" 7. DOCUMENT CONCENTRATION (TOP-3 RETRIEVED PAPERS PER MODE)")
  log("=" * 102)
  for mode in modes:
    doc_counter = Counter()
    for item in detailed[mode]:
      for h in item["hits"]:
        doc_counter[str(h[0])] += 1
    top_docs = doc_counter.most_common(3)
    top_str = ", ".join([f"{d[:25]}... ({c})" for d, c in top_docs])
    log(f"{mode:<18} | Top hits: {top_str}")

  # =========================================================================
  # 8. BOOTSTRAPPED 95% CONFIDENCE INTERVALS
  # =========================================================================
  log("\n" + "=" * 102)
  log(" 8. BOOTSTRAPPED 95% CONFIDENCE INTERVALS (1000 Iterations)")
  log("=" * 102)
  np.random.seed(42)
  n_boot = 1000
  for mode in modes:
    items = detailed[mode]
    n_queries = len(items)
    r3_vals = np.array([get_metric(item["recalls"], 3) for item in items])
    mrr_vals = np.array([item.get("mrr", 0.0) for item in items])

    boot_r3, boot_mrr = [], []
    for _ in range(n_boot):
      sample_idx = np.random.choice(n_queries, size=n_queries, replace=True)
      boot_r3.append(np.mean(r3_vals[sample_idx]) * 100)
      boot_mrr.append(np.mean(mrr_vals[sample_idx]) * 100)

    r3_ci = (np.percentile(boot_r3, 2.5), np.percentile(boot_r3, 97.5))
    mrr_ci = (np.percentile(boot_mrr, 2.5), np.percentile(boot_mrr, 97.5))
    log(
        f"{mode:<18} | R@3: {np.mean(r3_vals)*100:5.1f}% [{r3_ci[0]:5.1f}%,"
        f" {r3_ci[1]:5.1f}%] | MRR: {np.mean(mrr_vals)*100:5.1f}%"
        f" [{mrr_ci[0]:5.1f}%, {mrr_ci[1]:5.1f}%]"
    )

  # =========================================================================
  # 9. LATENCY SLICES
  # =========================================================================
  log("\n" + "=" * 102)
  log(" 9. LATENCY SLICES (p50 / p95 in ms by Evidence Type)")
  log("=" * 102)
  for mode in modes:
    log(f"Mode: {mode}")
    by_ev = defaultdict(list)
    for item in detailed[mode]:
      by_ev[item.get("evidence_type", "unknown")].append(
          item["latency_sec"] * 1000
      )
    for ev, lats in sorted(by_ev.items()):
      p50 = np.percentile(lats, 50)
      p95 = np.percentile(lats, 95)
      log(
          f"  • {ev:<10} (N={len(lats):2d}) | p50: {p50:6.1f} ms | p95:"
          f" {p95:6.1f} ms"
      )

  # =========================================================================
  # 10. SCORE CONFIDENCE SIGNAL
  # =========================================================================
  log("\n" + "=" * 102)
  log(" 10. SCORE CONFIDENCE SIGNAL (Rank 1 vs Rank 2 Gap)")
  log("=" * 102)
  for mode in ["text", "visual"]:
    correct_gaps, incorrect_gaps = [], []
    for item in detailed[mode]:
      hits = item["hits"]
      if len(hits) < 2:
        continue
      gold_set = {(str(p[0]), int(p[1])) for p in item["gold_pages"]}
      gap = abs(float(hits[0][2]) - float(hits[1][2]))
      is_rank1_correct = (str(hits[0][0]), int(hits[0][1])) in gold_set
      if is_rank1_correct:
        correct_gaps.append(gap)
      else:
        incorrect_gaps.append(gap)

    log(
        f"{mode:<18} | Rank-1 Correct Gap: {np.mean(correct_gaps):.4f}"
        f" (std={np.std(correct_gaps):.4f}) | Rank-1 Incorrect Gap:"
        f" {np.mean(incorrect_gaps):.4f} (std={np.std(incorrect_gaps):.4f})"
    )

  # =========================================================================
  # 11. PER-QUERY WINNERS & LOSERS
  # =========================================================================
  log("\n" + "=" * 102)
  log(" 11. PER-QUERY WINNERS & LOSERS (DISCREPANCY ANALYSIS)")
  log("=" * 102)
  hybrid_failures_others_won = []
  visual_won_text_failed = []
  text_won_visual_failed = []

  for qid in qids:
    arm_dict = queries_by_qid[qid]
    q_text = arm_dict["text"]["question"][:55] + "..."
    t_r3 = get_metric(arm_dict["text"]["recalls"], 3)
    v_r3 = get_metric(arm_dict["visual"]["recalls"], 3)
    h_r3 = get_metric(arm_dict["hybrid_routed"]["recalls"], 3)
    u_r3 = get_metric(arm_dict["fusion_unrouted"]["recalls"], 3)

    if h_r3 == 0.0 and (t_r3 > 0.0 or v_r3 > 0.0 or u_r3 > 0.0):
      hybrid_failures_others_won.append((qid, q_text, t_r3, v_r3, u_r3))

    if v_r3 > 0.0 and t_r3 == 0.0:
      visual_won_text_failed.append(
          (qid, q_text, arm_dict["text"]["evidence_type"])
      )

    if t_r3 > 0.0 and v_r3 == 0.0:
      text_won_visual_failed.append(
          (qid, q_text, arm_dict["text"]["evidence_type"])
      )

  log(
      f"\n[A] Queries where Hybrid Routed FAILS (R@3=0) but other arms succeed"
      f" ({len(hybrid_failures_others_won)} queries):"
  )
  for qid, q_text, tr, vr, ur in hybrid_failures_others_won:
    log(
        f"  • {qid}: Text:{tr > 0} | Vis:{vr > 0} | Unrouted:{ur > 0} ->"
        f" {q_text}"
    )

  log(
      f"\n[B] Queries where Visual WINS (R@3>0) and Text FAILS"
      f" ({len(visual_won_text_failed)} queries):"
  )
  for qid, q_text, ev in visual_won_text_failed:
    log(f"  • {qid} [{ev}]: {q_text}")

  log(
      f"\n[C] Queries where Text WINS (R@3>0) and Visual FAILS"
      f" ({len(text_won_visual_failed)} queries):"
  )
  for qid, q_text, ev in text_won_visual_failed:
    log(f"  • {qid} [{ev}]: {q_text}")

  # =========================================================================
  # 12. HELD-OUT UNANSWERABLE QUERIES FOR CRAG
  # =========================================================================
  unanswerable = data.get("unanswerable_held_out", [])
  log("\n" + "=" * 102)
  log(
      f" 12. HELD-OUT UNANSWERABLE QUERIES FOR CRAG: {len(unanswerable)} records"
      " ready"
  )
  log("=" * 102)
  for item in unanswerable:
    log(f"  • {item['qid']}: {item['question']}")

  # =========================================================================
  # Artifact Exporter
  # =========================================================================
  if out_dir:
    os.makedirs(out_dir, exist_ok=True)
    report_file = os.path.join(out_dir, "deep_analysis_report.md")
    with open(report_file, "w", encoding="utf-8") as rf:
      rf.write("# Offline Retrieval Ablation Deep Analysis\n\n")
      rf.write("```text\n")
      rf.write(buf.getvalue())
      rf.write("```\n")
    print(f"\n[Saved] Analytical report: {report_file}")

    csv_file = os.path.join(out_dir, "ablation_metrics_summary.csv")
    with open(csv_file, "w", newline="", encoding="utf-8") as cf:
      writer = csv.DictWriter(cf, fieldnames=summary_csv_rows[0].keys())
      writer.writeheader()
      writer.writerows(summary_csv_rows)
    print(f"[Saved] CSV Metrics Summary: {csv_file}")


if __name__ == "__main__":
  parser = argparse.ArgumentParser(description="Offline Deep Retrieval Analysis")
  parser.add_argument(
      "--file",
      type=str,
      default="evaluation/results_v2/benchmark_results.json",
      help="Path to benchmark_results.json",
  )
  parser.add_argument(
      "--out_dir",
      type=str,
      default="evaluation/results_v2",
      help="Directory to save report and CSV artifacts",
  )
  args = parser.parse_args()
  run_deep_analysis(args.file, args.out_dir)