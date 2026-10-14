# Offline Retrieval Ablation Deep Analysis

```text
===============================================================================================
 1. OVERALL METRIC EXPANSION ACROSS ALL MODES (R@1, R@3, R@5, R@10, NDCG, MRR)
===============================================================================================
Mode               | R@1 | R@3 | R@5 | R@10 | NDCG@1 | NDCG@3 | NDCG@5 | NDCG@10 | MRR
--------------------------------------------------------------------------------------
text               |  57.3% |  75.8% |  82.3% |  91.1% |   58.1% |   68.5% |   71.2% |   74.1% |  69.2%
visual             |  56.5% |  76.6% |  83.1% |  91.1% |   58.1% |   68.8% |   71.6% |   74.3% |  69.4%
hybrid_routed      |   7.3% |  62.1% |  78.2% |  87.9% |    8.1% |   41.9% |   48.8% |   52.1% |  40.7%
fusion_unrouted    |  59.7% |  82.3% |  87.1% |  95.2% |   61.3% |   73.2% |   75.4% |   77.9% |  73.4%

===============================================================================================
 2. GOLD TARGET RANK HISTOGRAM (1-10 vs Not Found)
===============================================================================================
text               | R1:36 R2: 8 R3: 4 R4: 3 R5: 1 R6: 1 R7: 1 R8: 1 R9: 1 R10: 1 | Not Found:  5
visual             | R1:36 R2: 8 R3: 4 R4: 4 R5: 0 R6: 1 R7: 2 R8: 0 R9: 2 R10: 0 | Not Found:  5
hybrid_routed      | R1: 5 R2:33 R3: 1 R4: 9 R5: 1 R6: 4 R7: 1 R8: 1 R9: 0 R10: 0 | Not Found:  7
fusion_unrouted    | R1:38 R2: 9 R3: 5 R4: 2 R5: 1 R6: 1 R7: 0 R8: 2 R9: 1 R10: 1 | Not Found:  2

===============================================================================================
 3. DETAILED METRIC SLICES (R@3 & MRR ACROSS ALL 4 MODES)
===============================================================================================

--- Slicing by: evidence_type ---
Slice Value                         | N  | text R@3 / MRR | visual R@3 / MRR | hybrid_ R@3 / MRR | fusion_ R@3 / MRR
-----------------------------------------------------------------------------------------------
figure                              | 15 |   80%/  80% |   87%/  77% |   67%/  39% |   93%/  85%
table                               | 19 |   74%/  60% |   84%/  73% |   58%/  36% |   79%/  67%
text                                | 28 |   75%/  70% |   66%/  63% |   62%/  45% |   79%/  71%

--- Slicing by: difficulty ---
Slice Value                         | N  | text R@3 / MRR | visual R@3 / MRR | hybrid_ R@3 / MRR | fusion_ R@3 / MRR
-----------------------------------------------------------------------------------------------
easy                                | 28 |   82%/  74% |   79%/  70% |   75%/  47% |   82%/  72%
hard                                | 11 |   68%/  66% |   82%/  68% |   55%/  34% |   77%/  80%
hard_candidate                      |  1 |    0%/   0% |    0%/  11% |    0%/   0% |    0%/  25%
medium                              | 22 |   75%/  68% |   75%/  72% |   52%/  38% |   89%/  74%

--- Slicing by: reasoning_type ---
Slice Value                         | N  | text R@3 / MRR | visual R@3 / MRR | hybrid_ R@3 / MRR | fusion_ R@3 / MRR
-----------------------------------------------------------------------------------------------
blank_cell_marker_cross_subtable_hop |  1 |    0%/  14% |  100%/ 100% |  100%/  50% |  100%/  50%
callout_to_bar_mapping_then_count_row_lookup |  1 |    0%/  17% |  100%/  50% |    0%/  25% |  100%/ 100%
condition_to_row_cross_group_hop    |  1 |  100%/ 100% |  100%/  50% |    0%/  25% |  100%/ 100%
counterfactual_finding              |  1 |  100%/  50% |    0%/  25% |    0%/  17% |  100%/  50%
cross_element_reconciliation_figure_vs_table |  1 |  100%/ 100% |  100%/  33% |    0%/  17% |  100%/ 100%
cross_page_ablation_to_hyperparameter |  1 |   50%/ 100% |  100%/ 100% |  100%/  50% |   50%/ 100%
cross_panel_load_inversion          |  1 |  100%/ 100% |  100%/ 100% |  100%/  50% |  100%/ 100%
design_detail                       |  6 |   83%/  68% |   83%/  75% |  100%/  64% |  100%/  72%
equation based generation           |  1 |    0%/  10% |    0%/  14% |    0%/  12% |    0%/  20%
equation_interpretation             |  4 |   75%/  69% |   50%/  40% |   25%/  24% |   75%/  67%
equation_to_algorithm_role_synthesis |  1 |    0%/   0% |    0%/  11% |    0%/   0% |    0%/  25%
extremum_then_hop                   |  1 |  100%/  50% |  100%/ 100% |  100%/  50% |  100%/ 100%
figure_ablation_ordering            |  1 |  100%/ 100% |  100%/ 100% |  100%/  50% |  100%/ 100%
figure_bar_comparison               |  1 |  100%/ 100% |  100%/ 100% |  100%/  50% |  100%/ 100%
figure_bar_extremum                 |  1 |  100%/ 100% |  100%/ 100% |  100%/  50% |  100%/ 100%
figure_bar_trend_by_bin             |  1 |    0%/  20% |  100%/ 100% |  100%/  50% |  100%/  33%
figure_category_comparison          |  1 |  100%/ 100% |  100%/  50% |    0%/  25% |  100%/ 100%
figure_cross_curve_comparison       |  1 |  100%/ 100% |  100%/ 100% |  100%/  50% |  100%/ 100%
figure_curve_comparison             |  1 |  100%/ 100% |  100%/ 100% |  100%/  50% |  100%/ 100%
figure_curve_trend                  |  1 |  100%/ 100% |  100%/ 100% |  100%/  50% |  100%/ 100%
figure_diagram_ordering             |  1 |  100%/ 100% |  100%/ 100% |  100%/  50% |  100%/ 100%
figure_multi_curve_trend            |  1 |    0%/  25% |    0%/  25% |    0%/  17% |  100%/  33%
finding_lookup                      |  3 |   67%/  70% |   67%/  67% |   67%/  33% |   67%/  67%
multi_curve_elimination_against_reference_line |  1 |  100%/ 100% |  100%/ 100% |  100%/  50% |  100%/ 100%
multi_page_design_detail            |  1 |   50%/  33% |   50%/ 100% |   50%/ 100% |   50%/ 100%
multi_panel_non_monotonic_scan      |  1 |  100%/  33% |    0%/   0% |    0%/   0% |    0%/  10%
multi_step_mechanism                |  1 |  100%/ 100% |  100%/ 100% |  100%/  50% |  100%/ 100%
table row scanning with threshold filtering across non-adjacent columns |  1 |  100%/ 100% |  100%/ 100% |  100%/  50% |  100%/ 100%
table_ablation_delta                |  1 |  100%/  50% |  100%/  50% |    0%/  25% |  100%/  50%
table_column_max                    |  1 |    0%/   0% |    0%/  25% |    0%/  17% |    0%/  11%
table_column_trend                  |  1 |  100%/  33% |  100%/  50% |    0%/  25% |  100%/  33%
table_multi_row_max_with_second_column |  1 |  100%/ 100% |  100%/ 100% |  100%/  50% |  100%/ 100%
table_pair_comparison               | 10 |   70%/  60% |   80%/  71% |   60%/  35% |   70%/  64%
table_variant_disambiguation        |  1 |  100%/ 100% |  100%/ 100% |  100%/  50% |  100%/ 100%
two_part_design_detail              |  1 |    0%/  12% |  100%/ 100% |    0%/  14% |  100%/  50%
why_design_choice                   |  8 |  100%/  94% |   75%/  62% |   75%/  53% |   88%/  81%

--- Slicing by: multi_page ---
Slice Value                         | N  | text R@3 / MRR | visual R@3 / MRR | hybrid_ R@3 / MRR | fusion_ R@3 / MRR
-----------------------------------------------------------------------------------------------
False                               | 59 |   78%/  71% |   78%/  69% |   63%/  40% |   85%/  73%
True                                |  3 |   33%/  44% |   50%/  70% |   50%/  50% |   33%/  75%

--- Slicing by: bucket ---
Slice Value                         | N  | text R@3 / MRR | visual R@3 / MRR | hybrid_ R@3 / MRR | fusion_ R@3 / MRR
-----------------------------------------------------------------------------------------------
standard                            | 62 |   76%/  69% |   77%/  69% |   62%/  41% |   82%/  73%

===============================================================================================
 4. CROSS-TABULATION: GOLD PAGE TYPE vs EVIDENCE TYPE
===============================================================================================
Page Type \ Evidence      | table        | Total
---------------------------------------------------------------------------
Page Type \ Evidence      | table        | Total
---------------------------------------------------------------------------
Page Type \ Evidence      | table        | Total
---------------------------------------------------------------------------
Page Type \ Evidence      | table        | Total
---------------------------------------------------------------------------
Page Type \ Evidence      | table        | Total
---------------------------------------------------------------------------
Page Type \ Evidence      | table        | Total
---------------------------------------------------------------------------
Page Type \ Evidence      | table        | Total
---------------------------------------------------------------------------
Page Type \ Evidence      | table        | Total
---------------------------------------------------------------------------
Page Type \ Evidence      | table        | Total
---------------------------------------------------------------------------
Page Type \ Evidence      | table        | Total
---------------------------------------------------------------------------
Page Type \ Evidence      | table        | Total
---------------------------------------------------------------------------
Page Type \ Evidence      | table        | Total
---------------------------------------------------------------------------
Page Type \ Evidence      | table        | Total
---------------------------------------------------------------------------
Page Type \ Evidence      | table        | Total
---------------------------------------------------------------------------
Page Type \ Evidence      | table        | Total
---------------------------------------------------------------------------
Page Type \ Evidence      | table        | Total
---------------------------------------------------------------------------
Page Type \ Evidence      | table        | Total
---------------------------------------------------------------------------
Page Type \ Evidence      | table        | Total
---------------------------------------------------------------------------
Page Type \ Evidence      | table        | Total
---------------------------------------------------------------------------
Page Type \ Evidence      | figure       | table        | Total
---------------------------------------------------------------------------
Page Type \ Evidence      | figure       | table        | Total
---------------------------------------------------------------------------
Page Type \ Evidence      | figure       | table        | Total
---------------------------------------------------------------------------
Page Type \ Evidence      | figure       | table        | Total
---------------------------------------------------------------------------
Page Type \ Evidence      | figure       | table        | Total
---------------------------------------------------------------------------
Page Type \ Evidence      | figure       | table        | Total
---------------------------------------------------------------------------
Page Type \ Evidence      | figure       | table        | Total
---------------------------------------------------------------------------
Page Type \ Evidence      | figure       | table        | Total
---------------------------------------------------------------------------
Page Type \ Evidence      | figure       | table        | Total
---------------------------------------------------------------------------
Page Type \ Evidence      | figure       | table        | Total
---------------------------------------------------------------------------
Page Type \ Evidence      | figure       | table        | Total
---------------------------------------------------------------------------
Page Type \ Evidence      | figure       | table        | Total
---------------------------------------------------------------------------
Page Type \ Evidence      | figure       | table        | Total
---------------------------------------------------------------------------
Page Type \ Evidence      | figure       | table        | Total
---------------------------------------------------------------------------
Page Type \ Evidence      | figure       | table        | Total
---------------------------------------------------------------------------
Page Type \ Evidence      | figure       | table        | text         | Total
---------------------------------------------------------------------------
Page Type \ Evidence      | figure       | table        | text         | Total
---------------------------------------------------------------------------
Page Type \ Evidence      | figure       | table        | text         | Total
---------------------------------------------------------------------------
Page Type \ Evidence      | figure       | table        | text         | Total
---------------------------------------------------------------------------
Page Type \ Evidence      | figure       | table        | text         | Total
---------------------------------------------------------------------------
Page Type \ Evidence      | figure       | table        | text         | Total
---------------------------------------------------------------------------
Page Type \ Evidence      | figure       | table        | text         | Total
---------------------------------------------------------------------------
Page Type \ Evidence      | figure       | table        | text         | Total
---------------------------------------------------------------------------
Page Type \ Evidence      | figure       | table        | text         | Total
---------------------------------------------------------------------------
Page Type \ Evidence      | figure       | table        | text         | Total
---------------------------------------------------------------------------
Page Type \ Evidence      | figure       | table        | text         | Total
---------------------------------------------------------------------------
Page Type \ Evidence      | figure       | table        | text         | Total
---------------------------------------------------------------------------
Page Type \ Evidence      | figure       | table        | text         | Total
---------------------------------------------------------------------------
Page Type \ Evidence      | figure       | table        | text         | Total
---------------------------------------------------------------------------
Page Type \ Evidence      | figure       | table        | text         | Total
---------------------------------------------------------------------------
Page Type \ Evidence      | figure       | table        | text         | Total
---------------------------------------------------------------------------
Page Type \ Evidence      | figure       | table        | text         | Total
---------------------------------------------------------------------------
Page Type \ Evidence      | figure       | table        | text         | Total
---------------------------------------------------------------------------
Page Type \ Evidence      | figure       | table        | text         | Total
---------------------------------------------------------------------------
Page Type \ Evidence      | figure       | table        | text         | Total
---------------------------------------------------------------------------
Page Type \ Evidence      | figure       | table        | text         | Total
---------------------------------------------------------------------------
Page Type \ Evidence      | figure       | table        | text         | Total
---------------------------------------------------------------------------
Page Type \ Evidence      | figure       | table        | text         | Total
---------------------------------------------------------------------------
Page Type \ Evidence      | figure       | table        | text         | Total
---------------------------------------------------------------------------
Page Type \ Evidence      | figure       | table        | text         | Total
---------------------------------------------------------------------------
Page Type \ Evidence      | figure       | table        | text         | Total
---------------------------------------------------------------------------
Page Type \ Evidence      | figure       | table        | text         | Total
---------------------------------------------------------------------------
Page Type \ Evidence      | figure       | table        | text         | Total
---------------------------------------------------------------------------
---------------------------------------------------------------------------
layout-heavy              | 15           | 19           | 19           | 53   
text-dense                | 0            | 0            | 9            | 9    

===============================================================================================
 5. PER-QUERY WINNERS & LOSERS (DISCREPANCY ANALYSIS)
===============================================================================================

[A] Queries where Hybrid Routed FAILS (R@3=0) but other arms succeed (17 queries):
  • T-004: Text:True | Vis:True | Unrouted:True -> Does CIDEr keep improving as LoRA rank increases for GP...
  • T-010: Text:True | Vis:True | Unrouted:True -> Which 4-bit quantization method yields a lower WikiText...
  • T-011: Text:True | Vis:True | Unrouted:True -> How does model size affect Gemma 1.1 IT's score on the ...
  • T-014: Text:True | Vis:True | Unrouted:True -> In Corrective Retrieval Augmented Generation, how does ...
  • T-019: Text:True | Vis:True | Unrouted:True -> Among the baselines reported from prior work on the mul...
  • F-001: Text:True | Vis:True | Unrouted:True -> Which kind of memory waste takes up a larger share of t...
  • F-006: Text:False | Vis:False | Unrouted:True -> In speculative decoding, how does the optimal number of...
  • F-012: Text:True | Vis:True | Unrouted:True -> Why do the final validation errors in the ImageNet trai...
  • F-013: Text:False | Vis:True | Unrouted:True -> How many keyword-counting samples are solved correctly ...
  • F-015: Text:True | Vis:False | Unrouted:False -> On which benchmarks does the model pretrained on ImageN...
  • X-002: Text:True | Vis:False | Unrouted:False -> Why is the relative position bias in windowed self-atte...
  • X-004: Text:False | Vis:True | Unrouted:True -> In the critique-and-revision stage, how is the guiding ...
  • X-006: Text:False | Vis:False | Unrouted:True -> Under DPO's Bradley-Terry model, what does the preferen...
  • X-008: Text:True | Vis:False | Unrouted:False -> Why does Kahneman-Tversky Optimization typically requir...
  • X-009: Text:True | Vis:True | Unrouted:True -> In the KL estimate for the reference point, how are neg...
  • X-023: Text:True | Vis:True | Unrouted:True -> Why did the authors pick sinusoidal positional encoding...
  • X-025: Text:True | Vis:False | Unrouted:True -> What happens to ranking quality if an instruction-tuned...

[B] Queries where Visual WINS (R@3>0) and Text FAILS (6 queries):
  • T-012 [table]: Which contrastive vision-language model evaluated in Co...
  • T-018 [table]: How many layers does the only Switch variant without th...
  • F-008 [figure]: On easier MATH questions (difficulty level 2), does maj...
  • F-013 [figure]: How many keyword-counting samples are solved correctly ...
  • X-004 [text]: In the critique-and-revision stage, how is the guiding ...
  • X-017 [text]: How can state space models process variable length sequ...

[C] Queries where Text WINS (R@3>0) and Visual FAILS (6 queries):
  • F-015 [figure]: On which benchmarks does the model pretrained on ImageN...
  • X-002 [text]: Why is the relative position bias in windowed self-atte...
  • X-007 [text]: In DPO, how should the reference policy be initialized ...
  • X-008 [text]: Why does Kahneman-Tversky Optimization typically requir...
  • X-021 [text]: Why does Llama 3 block attention across distinct docume...
  • X-025 [text]: What happens to ranking quality if an instruction-tuned...

===============================================================================================
 6. ERROR TAXONOMY: WRONG PAPER VS WRONG PAGE (TOP-10 CANDIDATES)
===============================================================================================
text               | Misses@10:  5 | Wrong Page (Same Paper):  5 | Wrong Paper Entirely:  0
visual             | Misses@10:  5 | Wrong Page (Same Paper):  5 | Wrong Paper Entirely:  0
hybrid_routed      | Misses@10:  7 | Wrong Page (Same Paper):  7 | Wrong Paper Entirely:  0
fusion_unrouted    | Misses@10:  2 | Wrong Page (Same Paper):  2 | Wrong Paper Entirely:  0

===============================================================================================
 7. DOCUMENT CONCENTRATION (TOP-3 RETRIEVED PAPERS PER MODE)
===============================================================================================
text               | Top hits: 2103.14030_Swin_Transform... (22), 2408.03314_Scaling_LLM_Te... (22), 2405.21060_Transformers_a... (20)
visual             | Top hits: 2103.14030_Swin_Transform... (24), 2403.09629_QuietSTaR_Lang... (22), 2405.21060_Transformers_a... (20)
hybrid_routed      | Top hits: 2405.21060_Transformers_a... (21), 2401.15884_Corrective_Ret... (20), 2403.09629_QuietSTaR_Lang... (19)
fusion_unrouted    | Top hits: 2103.14030_Swin_Transform... (24), 2106.09685_LoRA_LowRank_A... (22), 2405.21060_Transformers_a... (21)

===============================================================================================
 8. BOOTSTRAPPED 95% CONFIDENCE INTERVALS (1000 Iterations)
===============================================================================================
text               | R@3:  75.8% [ 65.3%,  85.5%] | MRR:  69.2% [ 59.9%,  78.2%]
visual             | R@3:  76.6% [ 66.9%,  86.3%] | MRR:  69.4% [ 60.4%,  78.7%]
hybrid_routed      | R@3:  62.1% [ 50.8%,  74.2%] | MRR:  40.7% [ 34.7%,  46.7%]
fusion_unrouted    | R@3:  82.3% [ 72.6%,  90.3%] | MRR:  73.4% [ 64.2%,  81.5%]

===============================================================================================
 9. LATENCY SLICES (p50 / p95 in ms by Evidence Type)
===============================================================================================
Mode: text
  • figure     (N=15) | p50:   99.8 ms | p95:  149.5 ms
  • table      (N=19) | p50:  110.4 ms | p95:  219.4 ms
  • text       (N=28) | p50:   97.3 ms | p95:  143.4 ms
Mode: visual
  • figure     (N=15) | p50:  421.5 ms | p95:  476.0 ms
  • table      (N=19) | p50:  406.5 ms | p95:  679.2 ms
  • text       (N=28) | p50:  344.9 ms | p95:  457.4 ms
Mode: hybrid_routed
  • figure     (N=15) | p50:  322.7 ms | p95:  353.1 ms
  • table      (N=19) | p50:  351.2 ms | p95:  426.6 ms
  • text       (N=28) | p50:  300.1 ms | p95:  384.0 ms
Mode: fusion_unrouted
  • figure     (N=15) | p50:  474.5 ms | p95:  574.8 ms
  • table      (N=19) | p50:  494.1 ms | p95:  613.1 ms
  • text       (N=28) | p50:  434.1 ms | p95:  538.1 ms

===============================================================================================
 10. SCORE CONFIDENCE SIGNAL (Rank 1 vs Rank 2 Gap)
===============================================================================================
text               | Rank-1 Correct Gap: 0.0438 (std=0.0307) | Rank-1 Incorrect Gap: 0.0188 (std=0.0171)
visual             | Rank-1 Correct Gap: 1.2446 (std=1.0505) | Rank-1 Incorrect Gap: 0.5026 (std=0.3953)

===============================================================================================
 11. OFFLINE FUSION EXPERIMENTS ON TOP-10 SAVED CANDIDATES
===============================================================================================
Harness Ground Truth Unrouted Fusion (k=20 pool) | R@3:  82.3% | MRR:  73.4%

  • Equal RRF (k=60)                               | R@3:  82.3% | MRR:  73.4%
  • Equal RRF (k=20)                               | R@3:  82.3% | MRR:  73.4%
  • Visual-Weighted RRF (Vis=1.5, Txt=1.0, k=60)   | R@3:  81.5% | MRR:  75.8%
  • Visual-Weighted RRF (Vis=2.0, Txt=1.0, k=60)   | R@3:  79.8% | MRR:  72.7%
  • Text-Weighted RRF (Vis=1.0, Txt=1.5, k=60)     | R@3:  80.6% | MRR:  73.1%
  • Min-Max Score Normalization Merge              | R@3:  76.6% | MRR:  71.0%

===============================================================================================
 12. HELD-OUT UNANSWERABLE QUERIES FOR CRAG: 0 records ready
===============================================================================================
```
