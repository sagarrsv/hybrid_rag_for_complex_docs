import json
from collections import defaultdict

with open("evaluation/results_v2/benchmark_results.json", "r", encoding="utf-8") as f:
    data = json.load(f)

for q in data["detailed"]["hybrid_v2"]:
    if q["qid"] == "X-022":
        t_pool = q["pools"]["text"]
        v_pool = q["pools"]["visual"]
        gold_set = {(str(g[0]), int(g[1])) for g in q["gold_pages"]}
        
        print("=== TEXT POOL CANDIDATES (Top 5) ===")
        for idx, h in enumerate(t_pool[:5], 1):
            print(f"Rank {idx}: doc={h[0][:15]} p={h[1]} raw_ptype={repr(h[3]) if len(h)>3 else 'None'}")
            
        print("\n=== COMPUTING BASE RRF SCORES (mult=1.0) ===")
        scores = defaultdict(float)
        breakdown = defaultdict(dict)
        
        for r_idx, h in enumerate(t_pool, 1):
            key = (str(h[0]), int(h[1]))
            s = 1.0 / (60 + r_idx)
            scores[key] += s
            breakdown[key]["text_rank"] = r_idx
            breakdown[key]["text_score"] = s
            
        for r_idx, h in enumerate(v_pool, 1):
            key = (str(h[0]), int(h[1]))
            s = 1.0 / (60 + r_idx)
            scores[key] += s
            breakdown[key]["vis_rank"] = r_idx
            breakdown[key]["vis_score"] = s

        ranked = sorted(scores.keys(), key=lambda k: scores[k], reverse=True)
        
        print("\n=== TOP 6 FINAL RANKED CANDIDATES ===")
        for rank, cand in enumerate(ranked[:6], 1):
            is_gold = " [GOLD]" if cand in gold_set else ""
            b = breakdown[cand]
            t_info = f"Text #{b.get('text_rank', '-')} ({b.get('text_score', 0):.6f})"
            v_info = f"Vis #{b.get('vis_rank', '-')} ({b.get('vis_score', 0):.6f})"
            print(f"Rank {rank}: {cand[0][:15]}:p{cand[1]} | Total: {scores[cand]:.6f} | {t_info} | {v_info}{is_gold}")
        break