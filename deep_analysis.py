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

  # In-memory text buffer to mirror terminal prints to markdown report
  buf = StringIO()

  def log(msg: str = ""):
    print(msg)
    buf.write(msg + "\n")

  # Group all arms by QID for side-by-side comparative analysis
  queries_by_qid = defaultdict(dict)
  for mode in modes:
    for item in detailed[mode]:
      queries_by_qid[item["qid"]][mode] = item

  # =========================================================================
  # 1. OVERALL METRIC EXPANSION ACROSS ALL MODES
  # =========================================================================
  log("=" * 95)
  log(
      " 1. OVERALL METRIC EXPANSION ACROSS ALL MODES (R@1, R@3, R@5, R@10, NDCG,"
      " MRR)"
  )
  log("=" * 95)
  k_keys = ["1", "3", "5", "10"]
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
        k: np.mean([item["recalls"].get(k, 0.0) for item in items]) * 100
        for k in k_keys
    }
    n_avgs = {
        k: np.mean([item["ndcgs"].get(k, 0.0) for item in items]) * 100
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
  log("\n" + "=" * 95)
  log(" 2. GOLD TARGET RANK HISTOGRAM (1-10 vs Not Found)")
  log("=" * 95)
  for mode in modes:
    hist = Counter()
    for item in detailed[mode]:
      gold_set = {(p[0], p[1]) for p in item["gold_pages"]}
      hit_rank = ">10"
      for rank_idx, h in enumerate(item["hits"], start=1):
        if (h[0], h[1]) in gold_set:
          hit_rank = str(rank_idx)
          break
      hist[hit_rank] += 1

    row_str = " ".join([f"R{r}:{hist[str(r)]:2d}" for r in range(1, 11)])
    log(f"{mode:<18} | {row_str} | Not Found: {hist['>10']:2d}")

  # =========================================================================
  # 3. DETAILED METRIC SLICES
  # =========================================================================
  log("\n" + "=" * 95)
  log(" 3. DETAILED METRIC SLICES (R@3 & MRR ACROSS ALL 4 MODES)")
  log("=" * 95)
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
    log("-" * 95)
    for val in field_vals:
      n_slice = sum(
          1 for item in detailed[modes[0]] if str(item.get(field)) == val
      )
      res_str = []
      for mode in modes:
        sub = [item for item in detailed[mode] if str(item.get(field)) == val]
        r3 = (
            np.mean([item["recalls"].get("3", 0.0) for item in sub]) * 100
            if sub
            else 0.0
        )
        mrr = np.mean([item.get("mrr", 0.0) for item in sub]) * 100 if sub else 0.0
        res_str.append(f"{r3:4.0f}%/{mrr:4.0f}%")
      log(f"{val:<35} | {n_slice:2d} | " + " | ".join(res_str))

  # =========================================================================
  # 4. CROSS-TABULATION: GOLD PAGE TYPE vs EVIDENCE TYPE
  # =========================================================================
  log("\n" + "=" * 95)
  log(" 4. CROSS-TABULATION: GOLD PAGE TYPE vs EVIDENCE TYPE")
  log("=" * 95)
  crosstab = defaultdict(Counter)
  evidence_types = set()
  page_types = set()
  for item in detailed[modes[0]]:
    ev = item.get("evidence_type", "unknown")
    pt = item["gold_page_types"][0] if item.get("gold_page_types") else "unknown"
    crosstab[pt][ev] += 1
    evidence_types.add(ev)
    page_types.add(pt)

    ev_list = sorted(list(evidence_types))
    header_title = "Page Type \ Evidence"
    log(f"{header_title:<25} | " + " | ".join([f"{e:<12}" for e in ev_list]) + " | Total")
    log("-" * 75)
  log("-" * 75)
  for pt in sorted(list(page_types)):
    counts = [crosstab[pt][e] for e in ev_list]
    log(
        f"{pt:<25} | "
        + " | ".join([f"{c:<12d}" for c in counts])
        + f" | {sum(counts):<5d}"
    )

  # =========================================================================
  # 5. PER-QUERY WINNERS & LOSERS
  # =========================================================================
  log("\n" + "=" * 95)
  log(" 5. PER-QUERY WINNERS & LOSERS (DISCREPANCY ANALYSIS)")
  log("=" * 95)
  hybrid_failures_others_won = []
  visual_won_text_failed = []
  text_won_visual_failed = []

  for qid, arm_dict in queries_by_qid.items():
    q_text = arm_dict["text"]["question"][:55] + "..."
    t_r3 = arm_dict["text"]["recalls"].get("3", 0.0)
    v_r3 = arm_dict["visual"]["recalls"].get("3", 0.0)
    h_r3 = arm_dict["hybrid_routed"]["recalls"].get("3", 0.0)
    u_r3 = arm_dict["fusion_unrouted"]["recalls"].get("3", 0.0)

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
  # 6. ERROR TAXONOMY: WRONG PAPER VS WRONG PAGE
  # =========================================================================
  log("\n" + "=" * 95)
  log(" 6. ERROR TAXONOMY: WRONG PAPER VS WRONG PAGE (TOP-10 CANDIDATES)")
  log("=" * 95)
  for mode in modes:
    wrong_paper = 0
    wrong_page_same_paper = 0
    total_misses = 0

    for item in detailed[mode]:
      gold_set = {(p[0], p[1]) for p in item["gold_pages"]}
      gold_docs = {p[0] for p in item["gold_pages"]}
      hit_pages = {(h[0], h[1]) for h in item["hits"]}
      hit_docs = {h[0] for h in item["hits"]}

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
  log("\n" + "=" * 95)
  log(" 7. DOCUMENT CONCENTRATION (TOP-3 RETRIEVED PAPERS PER MODE)")
  log("=" * 95)
  for mode in modes:
    doc_counter = Counter()
    for item in detailed[mode]:
      for h in item["hits"]:
        doc_counter[h[0]] += 1
    top_docs = doc_counter.most_common(3)
    top_str = ", ".join([f"{d[:25]}... ({c})" for d, c in top_docs])
    log(f"{mode:<18} | Top hits: {top_str}")

  # =========================================================================
  # 8. BOOTSTRAPPED 95% CONFIDENCE INTERVALS
  # =========================================================================
  log("\n" + "=" * 95)
  log(" 8. BOOTSTRAPPED 95% CONFIDENCE INTERVALS (1000 Iterations)")
  log("=" * 95)
  np.random.seed(42)
  n_boot = 1000
  for mode in modes:
    items = detailed[mode]
    n_queries = len(items)
    r3_vals = np.array([item["recalls"].get("3", 0.0) for item in items])
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
  log("\n" + "=" * 95)
  log(" 9. LATENCY SLICES (p50 / p95 in ms by Evidence Type)")
  log("=" * 95)
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
  log("\n" + "=" * 95)
  log(" 10. SCORE CONFIDENCE SIGNAL (Rank 1 vs Rank 2 Gap)")
  log("=" * 95)
  for mode in ["text", "visual"]:
    correct_gaps, incorrect_gaps = [], []
    for item in detailed[mode]:
      hits = item["hits"]
      if len(hits) < 2:
        continue
      gold_set = {(p[0], p[1]) for p in item["gold_pages"]}
      gap = abs(hits[0][2] - hits[1][2])
      is_rank1_correct = (hits[0][0], hits[0][1]) in gold_set
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
  # 11. OFFLINE FUSION EXPERIMENTS ON TOP-10 SAVED CANDIDATES
  # =========================================================================
  log("\n" + "=" * 95)
  log(" 11. OFFLINE FUSION EXPERIMENTS ON TOP-10 SAVED CANDIDATES")
  log("=" * 95)
  qids = list(queries_by_qid.keys())
  base_r3 = (
      np.mean([
          queries_by_qid[q]["fusion_unrouted"]["recalls"].get("3", 0.0)
          for q in qids
      ])
      * 100
  )
  base_mrr = (
      np.mean([
          queries_by_qid[q]["fusion_unrouted"].get("mrr", 0.0) for q in qids
      ])
      * 100
  )
  log(
      f"Harness Ground Truth Unrouted Fusion (k=20 pool) | R@3: {base_r3:5.1f}%"
      f" | MRR: {base_mrr:5.1f}%\n"
  )

  experiments = [
      ("Equal RRF (k=60)", 1.0, 1.0, 60),
      ("Equal RRF (k=20)", 1.0, 1.0, 20),
      ("Visual-Weighted RRF (Vis=1.5, Txt=1.0, k=60)", 1.0, 1.5, 60),
      ("Visual-Weighted RRF (Vis=2.0, Txt=1.0, k=60)", 1.0, 2.0, 60),
      ("Text-Weighted RRF (Vis=1.0, Txt=1.5, k=60)", 1.5, 1.0, 60),
  ]

  for name, w_text, w_vis, rrf_k in experiments:
    exp_r3, exp_mrr = [], []
    for qid in qids:
      t_hits = queries_by_qid[qid]["text"]["hits"]
      v_hits = queries_by_qid[qid]["visual"]["hits"]
      gold_set = {
          (p[0], p[1]) for p in queries_by_qid[qid]["text"]["gold_pages"]
      }

      fused_scores = defaultdict(float)
      for r_idx, h in enumerate(t_hits, start=1):
        fused_scores[(h[0], h[1])] += w_text * (1.0 / (rrf_k + r_idx))
      for r_idx, h in enumerate(v_hits, start=1):
        fused_scores[(h[0], h[1])] += w_vis * (1.0 / (rrf_k + r_idx))

      ranked = sorted(
          fused_scores.keys(), key=lambda p: fused_scores[p], reverse=True
      )
      recs, mrr = compute_recalls_and_mrr(ranked, gold_set)
      exp_r3.append(recs[3])
      exp_mrr.append(mrr)

    log(
        f"  • {name:<46} | R@3: {np.mean(exp_r3)*100:5.1f}% | MRR:"
        f" {np.mean(exp_mrr)*100:5.1f}%"
    )

  # Min-Max Normalization Simulation
  norm_r3, norm_mrr = [], []
  for qid in qids:
    t_hits = queries_by_qid[qid]["text"]["hits"]
    v_hits = queries_by_qid[qid]["visual"]["hits"]
    gold_set = {(p[0], p[1]) for p in queries_by_qid[qid]["text"]["gold_pages"]}

    t_scores = [h[2] for h in t_hits]
    v_scores = [h[2] for h in v_hits]

    def min_max(s_list: List[float]) -> List[float]:
      mn, mx = min(s_list), max(s_list)
      return [(s - mn) / (mx - mn + 1e-6) for s in s_list]

    t_norm = min_max(t_scores)
    v_norm = min_max(v_scores)

    fused_scores = defaultdict(float)
    for h, score in zip(t_hits, t_norm):
      fused_scores[(h[0], h[1])] = max(fused_scores[(h[0], h[1])], score)
    for h, score in zip(v_hits, v_norm):
      fused_scores[(h[0], h[1])] = max(fused_scores[(h[0], h[1])], score)

    ranked = sorted(
        fused_scores.keys(), key=lambda p: fused_scores[p], reverse=True
    )
    recs, mrr = compute_recalls_and_mrr(ranked, gold_set)
    norm_r3.append(recs[3])
    norm_mrr.append(mrr)

  log(
      f"  • {'Min-Max Score Normalization Merge':<46} | R@3:"
      f" {np.mean(norm_r3)*100:5.1f}% | MRR: {np.mean(norm_mrr)*100:5.1f}%"
  )

  # =========================================================================
  # 12. HELD-OUT UNANSWERABLE QUERIES FOR CRAG
  # =========================================================================
  unanswerable = data.get("unanswerable_held_out", [])
  log("\n" + "=" * 95)
  log(
      " 12. HELD-OUT UNANSWERABLE QUERIES FOR CRAG:"
      f" {len(unanswerable)} records ready"
  )
  log("=" * 95)
  for item in unanswerable:
    log(f"  • {item['qid']}: {item['question']}")

  # =========================================================================
  # Artifact Exporter
  # =========================================================================
  if out_dir:
    os.makedirs(out_dir, exist_ok=True)

    # 1. Full text / markdown log
    report_file = os.path.join(out_dir, "deep_analysis_report.md")
    with open(report_file, "w", encoding="utf-8") as rf:
      rf.write("# Offline Retrieval Ablation Deep Analysis\n\n")
      rf.write("```text\n")
      rf.write(buf.getvalue())
      rf.write("```\n")
    print(f"\n[Saved] Analytical report: {report_file}")

    # 2. Key summary comparison matrix as CSV
    csv_file = os.path.join(out_dir, "ablation_metrics_summary.csv")
    with open(csv_file, "w", newline="", encoding="utf-8") as cf:
      writer = csv.DictWriter(cf, fieldnames=summary_csv_rows[0].keys())
      writer.writeheader()
      writer.writerows(summary_csv_rows)
    print(f"[Saved] CSV Metrics Summary: {csv_file}")


if __name__ == "__main__":
  parser = argparse.ArgumentParser(
      description="Deep Evaluation Analysis on benchmark_results.json"
  )
  parser.add_argument(
      "--file",
      type=str,
      default="evaluation/results/benchmark_results.json",
      help="Path to benchmark_results.json",
  )
  parser.add_argument(
      "--out_dir",
      type=str,
      default="evaluation/results",
      help="Directory to save report and CSV artifacts",
  )
  args = parser.parse_args()
  run_deep_analysis(args.file, args.out_dir)