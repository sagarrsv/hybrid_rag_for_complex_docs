#!/usr/bin/env python3
#!/usr/bin/env python3
"""
hybrid_v2_tests.py
========================================================================================
Offline Parameter Sweep, Split Check & Plot Generator for Hybrid v2 Retrieval:
  - Sweeps: rrf_k in {5, 10, 20, 60} x depth in {5, 10, 15, 20}
  - Configurations:
      1. Plain RRF (No boost)
      2. Flat 2.0x Boost (text-dense)
      3. Decayed Second Vote (decay = 2, 3, 4)
      4. Protect Retriever Top-1 (RRF winner + Text#1 + Visual#1)
  - Plots:
      1. seesaw_scatter_<split>.png (Text-Dense R@3 vs Layout-Heavy R@3 at d=20, k=60)
      2. depth_curves_<split>.png (Overall, Text-Dense, Layout-Heavy R@3 vs Depth at k=60
         with 95% bootstrap error bars)
========================================================================================
"""

import os
import json
import argparse
from typing import List, Tuple, Dict, Any
import numpy as np
from tabulate import tabulate

try:
    import matplotlib.pyplot as plt
    HAS_MPL = True
except ImportError:
    HAS_MPL = False

RRF_K_LIST = [5, 10, 20, 60]
DEPTH_LIST = [5, 10, 15, 20]


def is_text_dense(ptype: str) -> bool:
    if not ptype:
        return False
    return str(ptype).lower().strip() in ("text-dense", "text", "text_dense", "txt")


def rrf_merge(
    t_pool: List[Any],
    v_pool: List[Any],
    rrf_k: int,
    depth: int,
    boost_mode: str = "none",  # "none", "flat_2x", "decayed"
    decay_factor: float = 2.0,
    protect_top1: bool = False
) -> List[Tuple[str, int]]:
    """Preserves insertion-order tie-breaking to match live retriever._fuse_rrf."""
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


def compute_query_vectors(
    queries: List[Dict[str, Any]],
    rrf_k: int,
    depth: int,
    boost_mode: str = "none",
    decay_factor: float = 2.0,
    protect_top1: bool = False
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    hits_all, hits_txt, hits_lay, mrrs = [], [], [], []

    for q in queries:
        gold_set = {(str(g[0]), int(g[1])) for g in q["gold_pages"]}
        q_is_text = bool(q.get("gold_page_types") and is_text_dense(q["gold_page_types"][0]))

        t_pool = q.get("pools", {}).get("text", [])
        v_pool = q.get("pools", {}).get("visual", [])

        ranked = rrf_merge(
            t_pool=t_pool, v_pool=v_pool, rrf_k=rrf_k, depth=depth,
            boost_mode=boost_mode, decay_factor=decay_factor, protect_top1=protect_top1
        )

        top3 = ranked[:3]
        rec3 = sum(1 for p in top3 if p in gold_set) / len(gold_set) if gold_set else 0.0

        hits_all.append(rec3 * 100.0)
        if q_is_text:
            hits_txt.append(rec3 * 100.0)
        else:
            hits_lay.append(rec3 * 100.0)

        q_rr = 0.0
        for rank_idx, cand in enumerate(ranked[:10], start=1):
            if cand in gold_set:
                q_rr = 1.0 / rank_idx
                break
        mrrs.append(q_rr)

    return (
        np.array(hits_all, dtype=float),
        np.array(hits_txt, dtype=float),
        np.array(hits_lay, dtype=float),
        np.array(mrrs, dtype=float),
    )


def evaluate_queries(
    queries: List[Dict[str, Any]],
    rrf_k: int,
    depth: int,
    boost_mode: str = "none",
    decay_factor: float = 2.0,
    protect_top1: bool = False
) -> Tuple[float, float, float, float]:
    h_all, h_txt, h_lay, mrrs = compute_query_vectors(
        queries, rrf_k, depth, boost_mode, decay_factor, protect_top1
    )
    return (
        float(np.mean(h_all)) if len(h_all) else 0.0,
        float(np.mean(h_txt)) if len(h_txt) else 0.0,
        float(np.mean(h_lay)) if len(h_lay) else 0.0,
        float(np.mean(mrrs)) if len(mrrs) else 0.0,
    )


def bootstrap_ci(arr: np.ndarray, n_boot: int = 1000, seed: int = 42) -> Tuple[float, float, float]:
    if len(arr) == 0:
        return 0.0, 0.0, 0.0
    rng = np.random.default_rng(seed)
    means = [np.mean(rng.choice(arr, size=len(arr), replace=True)) for _ in range(n_boot)]
    return float(np.mean(arr)), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def extract_reference_mode_metrics(mode_items: List[Dict[str, Any]]) -> Tuple[float, float, float]:
    all_r3, txt_r3, lay_r3 = [], [], []
    for item in mode_items:
        r3 = float(item["recalls"].get("3", item["recalls"].get(3, 0.0))) * 100.0
        all_r3.append(r3)
        if item.get("gold_page_types") and is_text_dense(item["gold_page_types"][0]):
            txt_r3.append(r3)
        else:
            lay_r3.append(r3)
    return (
        float(np.mean(all_r3)) if all_r3 else 0.0,
        float(np.mean(txt_r3)) if txt_r3 else 0.0,
        float(np.mean(lay_r3)) if lay_r3 else 0.0,
    )


def generate_plots(
    data: Dict[str, Any],
    queries_v2: List[Dict[str, Any]],
    configurations: List[Dict[str, Any]],
    out_dir: str,
    split_label: str = "all"
):
    if not HAS_MPL:
        print("[Warning] matplotlib is not installed. Skipping plot generation.")
        return

    suffix = f"_{split_label}" if split_label != "all" else ""

    # 1. Plot 1: Seesaw Scatter at depth=20, rrf_k=60
    fig, ax = plt.subplots(figsize=(8.5, 6.5), dpi=180)
    ref_specs = [
        ("text", "Text-Only", "#1f77b4", "s"),
        ("visual", "Visual-Only", "#ff7f0e", "^"),
        ("fusion_unrouted", "Unrouted Fusion", "#2ca02c", "D"),
    ]
    for mode_key, label, color, marker in ref_specs:
        if mode_key in data.get("detailed", {}):
            items = [
                q for q in data["detailed"][mode_key]
                if split_label == "all" or q.get("split", "dev") == split_label
            ]
            _, r3_txt, r3_lay = extract_reference_mode_metrics(items)
            ax.scatter(
                r3_txt, r3_lay, color=color, marker=marker, s=130,
                edgecolor="black", linewidth=1.0, zorder=5, label=f"{label} (Ref)"
            )
            ax.annotate(
                f"{label}\n({r3_txt:.1f}%, {r3_lay:.1f}%)",
                (r3_txt, r3_lay), textcoords="offset points", xytext=(8, -6), fontsize=8.5
            )

    v2_colors = ["#d62728", "#9467bd", "#8c564b", "#e377c2", "#17becf", "#bcbd22"]
    for idx, cfg in enumerate(configurations):
        _, r3_txt, r3_lay, _ = evaluate_queries(
            queries=queries_v2, rrf_k=60, depth=20,
            boost_mode=cfg["mode"], decay_factor=cfg["decay"], protect_top1=cfg["protect"]
        )
        short_label = cfg["short_name"]
        color = v2_colors[idx % len(v2_colors)]
        ax.scatter(
            r3_txt, r3_lay, color=color, marker="o", s=120,
            edgecolor="black", linewidth=0.8, zorder=6, label=short_label
        )
        y_offset = 7 if idx % 2 == 0 else -14
        ax.annotate(
            f"{short_label} ({r3_txt:.1f}%, {r3_lay:.1f}%)",
            (r3_txt, r3_lay), textcoords="offset points", xytext=(7, y_offset), fontsize=8.5
        )

    ax.set_title(f"Seesaw Tradeoff: Text-Dense vs. Layout-Heavy R@3 (d=20, rrf_k=60 | split={split_label})", fontsize=11, fontweight="bold")
    ax.set_xlabel("Text-Dense Recall@3 (%)", fontsize=10)
    ax.set_ylabel("Layout-Heavy Recall@3 (%)", fontsize=10)
    ax.grid(True, linestyle="--", alpha=0.45)
    ax.legend(loc="lower left", fontsize=8.5, framealpha=0.9)
    fig.tight_layout()

    scatter_path = os.path.join(out_dir, f"seesaw_scatter{suffix}.png")
    fig.savefig(scatter_path)
    plt.close(fig)
    print(f"[Saved Plot 1] Seesaw scatter saved to: {scatter_path}")

    # 2. Plot 2: Depth Curves at rrf_k=60 with 95% Bootstrap Error Bars
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.2), dpi=180, sharex=True)
    panel_titles = ["Overall Recall@3", "Text-Dense Recall@3", "Layout-Heavy Recall@3"]
    jitters = np.linspace(-0.35, 0.35, len(configurations))

    for idx, cfg in enumerate(configurations):
        color = v2_colors[idx % len(v2_colors)]
        short_label = cfg["short_name"]
        x_vals = [d + jitters[idx] for d in DEPTH_LIST]

        means_all, err_low_all, err_high_all = [], [], []
        means_txt, err_low_txt, err_high_txt = [], [], []
        means_lay, err_low_lay, err_high_lay = [], [], []

        for d in DEPTH_LIST:
            h_all, h_txt, h_lay, _ = compute_query_vectors(
                queries=queries_v2, rrf_k=60, depth=d,
                boost_mode=cfg["mode"], decay_factor=cfg["decay"], protect_top1=cfg["protect"]
            )
            for arr, m_list, l_list, u_list in [
                (h_all, means_all, err_low_all, err_high_all),
                (h_txt, means_txt, err_low_txt, err_high_txt),
                (h_lay, means_lay, err_low_lay, err_high_lay),
            ]:
                mean_v, ci_l, ci_u = bootstrap_ci(arr, n_boot=1000, seed=42)
                m_list.append(mean_v)
                l_list.append(max(0.0, mean_v - ci_l))
                u_list.append(max(0.0, ci_u - mean_v))

        for ax_idx, (m_list, l_list, u_list) in enumerate([
            (means_all, err_low_all, err_high_all),
            (means_txt, err_low_txt, err_high_txt),
            (means_lay, err_low_lay, err_high_lay),
        ]):
            axes[ax_idx].errorbar(
                x_vals, m_list, yerr=[l_list, u_list],
                label=short_label, color=color, marker="o", markersize=5,
                linewidth=1.8, capsize=3, alpha=0.88
            )

    for ax_idx, ax in enumerate(axes):
        ax.set_title(panel_titles[ax_idx], fontsize=11, fontweight="bold")
        ax.set_xlabel("Candidate Pool Depth", fontsize=10)
        if ax_idx == 0:
            ax.set_ylabel("Recall@3 (%)", fontsize=10)
        ax.set_xticks(DEPTH_LIST)
        ax.grid(True, linestyle="--", alpha=0.45)

    axes[0].legend(loc="best", fontsize=8.5, framealpha=0.9)
    fig.suptitle(f"Recall@3 vs. Candidate Pool Depth at rrf_k=60 (95% Bootstrap CIs | split={split_label})", fontsize=12, fontweight="bold")
    fig.tight_layout()

    curves_path = os.path.join(out_dir, f"depth_curves{suffix}.png")
    fig.savefig(curves_path)
    plt.close(fig)
    print(f"[Saved Plot 2] Depth curves saved to: {curves_path}")


def format_table(
    title: str,
    metric_name: str,
    grid_data: Dict[Tuple[int, int], float],
    is_percentage: bool = True,
    tablefmt: str = "grid"
) -> str:
    headers = ["rrf_k \\ depth"] + [str(d) for d in DEPTH_LIST]
    table_rows = []
    for k in RRF_K_LIST:
        row = [str(k)]
        for d in DEPTH_LIST:
            val = grid_data[(k, d)]
            row.append(f"{val:5.1f}%" if is_percentage else f"{val:.3f}")
        table_rows.append(row)
    return f"\n>> {title} | {metric_name}\n" + tabulate(table_rows, headers=headers, tablefmt=tablefmt)


def run_configuration_sweep(
    queries: List[Dict[str, Any]],
    config_name: str,
    boost_mode: str = "none",
    decay_factor: float = 2.0,
    protect_top1: bool = False
) -> Tuple[Dict[str, Any], str]:
    print("\n" + "#" * 90)
    print(f" EXPERIMENT CONFIGURATION: {config_name} (N={len(queries)})")
    print("#" * 90)

    grids = {"R@3 overall": {}, "R@3 text-dense": {}, "R@3 layout-heavy": {}, "MRR": {}}
    json_payload = {
        "config": config_name,
        "boost_mode": boost_mode,
        "decay_factor": decay_factor,
        "protect_top1": protect_top1,
        "results": {}
    }

    for k in RRF_K_LIST:
        for d in DEPTH_LIST:
            r3_all, r3_txt, r3_lay, mrr = evaluate_queries(
                queries=queries, rrf_k=k, depth=d,
                boost_mode=boost_mode, decay_factor=decay_factor, protect_top1=protect_top1
            )
            grids["R@3 overall"][(k, d)] = r3_all
            grids["R@3 text-dense"][(k, d)] = r3_txt
            grids["R@3 layout-heavy"][(k, d)] = r3_lay
            grids["MRR"][(k, d)] = mrr

            json_payload["results"][f"k{k}_d{d}"] = {
                "rrf_k": k, "depth": d,
                "r3_overall": round(r3_all, 2),
                "r3_text_dense": round(r3_txt, 2),
                "r3_layout_heavy": round(r3_lay, 2),
                "mrr": round(mrr, 4)
            }

    for m_name, is_pct in [("R@3 overall", True), ("R@3 text-dense", True), ("R@3 layout-heavy", True), ("MRR", False)]:
        print(format_table(config_name, m_name, grids[m_name], is_percentage=is_pct, tablefmt="grid"))

    md_content = f"\n### {config_name}\n"
    for m_name, is_pct in [("R@3 overall", True), ("R@3 text-dense", True), ("R@3 layout-heavy", True), ("MRR", False)]:
        md_content += format_table(config_name, m_name, grids[m_name], is_percentage=is_pct, tablefmt="pipe") + "\n"

    return json_payload, md_content


def main():
    parser = argparse.ArgumentParser(description="Sweep RRF parameters, generate tables and plots.")
    parser.add_argument("--file", type=str, default="evaluation/results_v3/benchmark_results.json")
    parser.add_argument("--out_dir", type=str, default="evaluation/results_v3")
    parser.add_argument("--split", type=str, default="all", choices=["all", "dev", "test"])
    args = parser.parse_args()

    if not os.path.exists(args.file):
        raise FileNotFoundError(f"Cannot find results file at: {args.file}")

    os.makedirs(args.out_dir, exist_ok=True)
    with open(args.file, "r", encoding="utf-8") as f:
        data = json.load(f)

    all_queries = data["detailed"]["hybrid_v2_d3"]
    queries = all_queries if args.split == "all" else [q for q in all_queries if q.get("split", "dev") == args.split]

    configurations = [
        {"name": "v2 (plain RRF, no boost)", "short_name": "plain v2", "mode": "none", "decay": 2.0, "protect": False},
        {"name": "v2 + plain 2x boost", "short_name": "plain 2x", "mode": "flat_2x", "decay": 2.0, "protect": False},
        {"name": "v2 + decayed boost (decay=2)", "short_name": "decay 2", "mode": "decayed", "decay": 2.0, "protect": False},
        {"name": "v2 + decayed boost (decay=3)", "short_name": "decay 3", "mode": "decayed", "decay": 3.0, "protect": False},
        {"name": "v2 + decayed boost (decay=4)", "short_name": "decay 4", "mode": "decayed", "decay": 4.0, "protect": False},
        {"name": "v2 + protect retriever top-1 (RRF#1 + T#1 + V#1)", "short_name": "protect top-1", "mode": "none", "decay": 2.0, "protect": True},
    ]

    all_json_results = []
    full_markdown_report = f"# Deep Analysis v2 Parameter Sweep & Ablation Results (Split: {args.split}, N={len(queries)})\n\n"

    for cfg in configurations:
        payload, md_block = run_configuration_sweep(
            queries=queries, config_name=cfg["name"],
            boost_mode=cfg["mode"], decay_factor=cfg["decay"], protect_top1=cfg["protect"]
        )
        all_json_results.append(payload)
        full_markdown_report += md_block

    suffix = f"_{args.split}" if args.split != "all" else ""
    json_path = os.path.join(args.out_dir, f"hybrid_v2_tests{suffix}.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(all_json_results, f, indent=2)
    print(f"\n[Saved JSON] Exported complete sweep data to: {json_path}")

    md_path = os.path.join(args.out_dir, f"hybrid_v2_tests{suffix}.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(full_markdown_report)
    print(f"[Saved Markdown] Exported formatted table report to: {md_path}")

    generate_plots(data, queries, configurations, args.out_dir, split_label=args.split)


if __name__ == "__main__":
    main()