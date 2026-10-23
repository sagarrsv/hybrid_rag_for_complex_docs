# Generation evaluation report

## Main evaluation results

| Metric | Result |
| --- | --- |
| Questions evaluated | 78 |
| Answer correctness | 80.8% (63/78) |
| Correctness given complete retrieval | 92.5% (62/67) |
| Complete gold-page retrieval | 85.9% (67/78) |
| Faithfulness (Claude, unadjusted) | 88.5% (69/78) |
| Faithfulness (manually adjudicated) | 97.4% (76/78) |
| Adjudicated faithfulness, answered only | 97.1% (67/69) |

## Performance by evidence type

| Group | N | Correctness | Given retrieval | Retrieval | Adjudicated faithfulness |
| --- | --- | --- | --- | --- | --- |
| Text | 43 | 76.7% (33/43) | 94.1% (32/34) | 79.1% (34/43) | 97.7% (42/43) |
| Table | 19 | 89.5% (17/19) | 100.0% (17/17) | 89.5% (17/19) | 100.0% (19/19) |
| Figure | 16 | 81.2% (13/16) | 81.2% (13/16) | 100.0% (16/16) | 93.8% (15/16) |

## Performance by difficulty

| Group | N | Correctness | Given retrieval | Retrieval | Adjudicated faithfulness |
| --- | --- | --- | --- | --- | --- |
| Easy | 36 | 80.6% (29/36) | 93.5% (29/31) | 86.1% (31/36) | 100.0% (36/36) |
| Medium | 28 | 85.7% (24/28) | 95.8% (23/24) | 85.7% (24/28) | 96.4% (27/28) |
| Hard | 14 | 71.4% (10/14) | 83.3% (10/12) | 85.7% (12/14) | 92.9% (13/14) |

## Performance by reasoning type

| Group | N | Correctness | Given retrieval | Retrieval | Adjudicated faithfulness |
| --- | --- | --- | --- | --- | --- |
| Comparison | 18 | 88.9% (16/18) | 100.0% (16/16) | 88.9% (16/18) | 100.0% (18/18) |
| Design Detail | 14 | 85.7% (12/14) | 100.0% (11/11) | 78.6% (11/14) | 100.0% (14/14) |
| Equation Interpretation | 6 | 83.3% (5/6) | 100.0% (5/5) | 83.3% (5/6) | 100.0% (6/6) |
| Lookup | 4 | 75.0% (3/4) | 100.0% (3/3) | 75.0% (3/4) | 100.0% (4/4) |
| Multi Hop Synthesis | 12 | 83.3% (10/12) | 100.0% (10/10) | 83.3% (10/12) | 100.0% (12/12) |
| Trend Or Extremum | 9 | 55.6% (5/9) | 62.5% (5/8) | 88.9% (8/9) | 88.9% (8/9) |
| Why Rationale | 15 | 80.0% (12/15) | 85.7% (12/14) | 93.3% (14/15) | 93.3% (14/15) |

## Latency and token usage

| Group | Gen (s) | Total (s) | Input tokens | Output tokens | Total tokens |
| --- | --- | --- | --- | --- | --- |
| Overall | 46.7 | 48.7 | 3,692 | 681 | 4,373 |
| Text | 25.3 | 27.4 | 3,716 | 541 | 4,257 |
| Table | 59.3 | 61.1 | 3,667 | 797 | 4,464 |
| Figure | 89.3 | 91.1 | 3,657 | 918 | 4,575 |

## Scoring notes

- Answer correctness: reference-based score of 2 (fully correct).
- Complete retrieval: every annotated gold page appears in retrieved pages.
- Given retrieval: correctness among questions with complete gold-page coverage.
- Faithfulness: support of stated claims by retrieved evidence, not answer completeness.
- Manual overrides are limited to the documented QIDs and retain Claude's raw verdict.
- Adjudicated faithfulness over all questions includes abstentions; the answered-only rate is separate.
- Percentages reflect available judgments, not necessarily all N when a judge result is missing.
