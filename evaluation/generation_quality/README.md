> **Architecture:** CLI entrypoint plus five focused modules. All six Python files must stay together.

> **Code style:** Short function docstrings; step-specific comments appear immediately before the relevant code. Judge prompts use numbered rules and vertically readable output schemas. No automatic judge retries.

# Hybrid RAG — Generation Evaluation v1.3 (modular)

The evaluator uses **saved predictions**, not new generation calls. It supports incremental runs as more predictions arrive. No estimated financial costs are calculated.

## Setup

From the RAG project's root:


The commands below assume `python evaluation/generation_quality/evaluate.py` and `PRED=/path/to/predictions.jsonl`. `PDF_DIR` must point to the **original local PDFs**; they were not included in the uploaded project ZIP. PDF names should match each `doc_id` exactly (e.g., `<doc_id>.pdf`). The page numbering defaults to 1-based PDF page numbers, matching the supplied reference examples **only if they use 1-based PDF indexing**; check against one known page before batching. If your ingestion's `page_num` is zero-based, use `--page-numbering zero`.

### 1. Offline metrics (free, run first)

```bash
python evaluation/generation_quality/evaluate.py offline \
  --predictions "evaluation/results_generation/v3_dataset/gemini_38_flash/gen_fail_test/gemini-3.8-flash/retrieved/predictions.jsonl" \
  --golden evaluation/golden_dataset_v3.json \
  --output evaluation/generation_quality/eval_results/offline.jsonl \
  --summary evaluation/generation_quality/eval_results/offline_summary.json
```

For models or experiments in **different files**, pass the files together after `--predictions`. Duplicates with identical (model, experiment, QID) keys are resolved by keeping the last prediction; keep experiments separate to avoid accidental overwrite.

Outputs per-query: retrieved gold-page coverage, citation whitelist and `P1/P2/...` mapping checks, abstention agreement, retrieval/generation/total latency, reported tokens, evidence type/difficulty/reasoning type. Aggregates include model and category breakdowns. **Citation validity does not imply claim support**. **Gold-page coverage is defined against annotated gold pages; other retrieved pages may also support an answer**. Abstention accuracy on answerable-only datasets is not informative alone; see data limitations below.

### 2. Correctness / completeness / factual-error judging (Groq, API calls)

```bash
export GROQ_API_KEY=...  # use .env or shell environment; never commit credentials
python evaluation/generation_quality/evaluate.py correctness \
  --predictions "evaluation/results_generation/v3_dataset/gemini_38_flash/gen_fail_test/gemini-3.8-flash/retrieved/predictions.jsonl" \
  --model openai/gpt-oss-120b \
  --max-requests 8 \
  --output evaluation/generation_quality/eval_results/correctness.jsonl
```

- This makes **up to 8 billable/quota-bearing API calls**, one per uncached prediction, with **no automatic retry**.
- Re-running with the same output file skips successful judgments. Error records can be retried only by running the command again.
- Scores: `0` wrong, `1` partial, `2` fully correct, `completeness` boolean, `factual_error` boolean, `missing_facts`, `wrong_facts`, rationale.
- Question-intent rubric: don't demand numbers that weren't requested; wrong numbers/ordering count as factual errors.
- `--max-requests` defaults to 10 as a safety guard; increase intentionally when calibration is verified.

### 3. Multimodal faithfulness (Anthropic Batch)

First **prepare** batch requests offline (does not call API):

```bash
python evaluation/generation_quality/evaluate.py prepare-batch \
  --predictions "evaluation/results_generation/v3_dataset/gemini_38_flash/gen_fail_test/gemini-3.8-flash/retrieved/predictions.jsonl" \
  --pdf-dir "$PDFDIR" \
  --dpi 150 \
  --output evaluation/generation_quality/eval_results/claude_requests.json \
  --manifest evaluation/generation_quality/eval_results/claude_manifest.jsonl
```

Review `claude_manifest.jsonl` and validate image/PDF page mapping before submitting. Layout-heavy pages are supplied as rendered JPEG images; text-dense pages are supplied as extracted text. This follows the `page_type` in the saved retrieved-pages metadata, which is a proxy for actual generator context. If that metadata or the original generator's context rules differ, adjust reconstruction before interpreting faithfulness results.

Submit **only after explicitly approving the spend**:

```bash
export ANTHROPIC_API_KEY=... # via secure environment
python evaluation/generation_quality/evaluate.py submit-batch \
  --requests evaluation/generation_quality/eval_results/claude_requests.json \
  --max-requests 1 --confirm \
  --batch-id-file evaluation/generation_quality/eval_results/calibration_batch_id.txt
```

`--confirm` is required. `--max-requests 8` prepares an 8-question calibration. For a full batch, set `--max-requests 0`. Never submit a request set containing already judged questions without filtering them first. Successful Anthropic batch submissions can incur charges; check the provider's actual pricing and batch status.

After Anthropic reports the batch ended:

```bash
python evaluation/generation_quality/evaluate.py collect-batch \
  --batch-id-file evaluation/generation_quality/eval_results/calibration_batch_id.txt \
  --manifest evaluation/generation_quality/eval_results/claude_manifest.jsonl \
  --output evaluation/generation_quality/eval_results/faithfulness.jsonl
```

Batch results are matched by `custom_id`, not order. The judge sees retrieved pages, question, answer and citations, but **no golden reference**. Claim-level statuses: `supported`, `unsupported`, `unverifiable`. The response additionally contains overall faithfulness and citation support. Results not correctly parsed retain an explicit error record rather than silently passing.

python evaluation/generation_quality/test_faithfulness.py \
  --requests evaluation/generation_quality/eval_results/claude_requests.json \
  --manifest evaluation/generation_quality/eval_results/claude_manifest.jsonl \
  --output evaluation/generation_quality/eval_results/faithfulness_test.jsonl \
  --max-requests 3 \
  --confirm

### 4. Merge and reporting

```bash
python evaluation/generation_quality/evaluate.py merge \
  --offline evaluation/generation_quality/eval_results/offline.jsonl \
  --correctness evaluation/generation_quality/eval_results/correctness.jsonl \
  --faithfulness evaluation/generation_quality/eval_results/faithfulness.jsonl \
  --output evaluation/generation_quality/eval_results/final.jsonl \
  --summary evaluation/generation_quality/eval_results/final_summary.json
```

When the remaining generation predictions arrive, rerun `offline` and rerun judge steps with the same files; correctness caches successful QIDs. **The Claude request builder does not automatically skip completed judgments**, so create a new filtered request set for newly completed QIDs (or add a skip-manifest feature) to avoid paying twice.

### Important evaluation caveats

- **No API has been called during file creation or local tests.** There is no hidden evaluation cost or retry mechanism.
- Human-review a stratified calibration of 10–15 text/table/figure examples; independent multimodal judges can hallucinate evidence or misread figures too.
- Reference answer is authoritative only for *correctness*, not for *faithfulness*.
- `claim_support_ratio` = supported / (supported + unsupported). Report `unverifiable` claims separately so this ratio isn't interpreted as support for unseen claims.
- Judge output is a *model assessment*, not verified ground truth. Human-audit disputed numeric/ranking/trend judgments.
- The `total_latency` is the sum of logged retrieval + context + generation durations, not independently measured end-to-end wall time (e.g. it excludes any external queueing or rate-limit sleep outside recorded spans).
- Summaries on partial runs show only evaluated predictions, not 78-question benchmark performance. For cross-model comparisons, evaluate common QIDs or carefully report different denominators.
- Absence of a citation produces `all_citations_valid=null`, **not** a perfect citation-validity score.
- Reported tokens reflect the original generator's provider metadata; no dollar cost estimates are calculated.
- Don't commit `claude_requests.json`: it embeds base64 document page images and can be very large. Add `eval_results/` to `.gitignore` where appropriate.


## Understanding `evaluate_answer_correctness`

This provider-independent name describes **what is evaluated**, not where the API is hosted. The current judge transport is Groq, configured with `--model openai/gpt-oss-120b`. To change the provider later, the evaluation objective and output schema should remain unchanged.

The function follows these steps:

1. Load existing successful judgments from `correctness.jsonl` as a cache keyed by generator model, experiment and question ID.
2. Iterate over completed predictions and skip previously evaluated questions.
3. Construct the judge's input from **question + golden reference answer + generated answer + answerability flags**. No retrieved PDF pages are given to this judge because faithfulness is scored separately.
4. Make **one** API call using a strict scoring rubric, temperature zero and JSON mode. `--max-requests` caps the number of requests, including failed ones.
5. Parse the JSON and validate correctness (0, 1 or 2), completeness (boolean), factual error (boolean), the missing/wrong fact lists, and the rationale.
6. Append the judgment, generator identity, judge model and judge token counts immediately to the result file; on failure, append an error instead of fabricating a score.

**What the metrics mean:** `correctness=2` requires all essential requested facts without material errors; `1` indicates partially correct; `0` is wrong. `completeness` checks whether all essential requested facts are provided. `factual_error` distinguishes wrong facts from omissions. The judge is fallible; manually audit difficult table and figure comparisons.

**Safety:** There are no automatic retries. The function never runs as part of the offline or faithfulness batch preparation stages.


## Module layout

```text
evaluation/generation_quality/
├── evaluate.py                 # CLI: dispatch only
├── data_loader.py              # JSONL I/O, deduplication, page IDs
├── deterministic_metrics.py    # Citation, coverage, abstention, usage
├── correctness_evaluator.py   # Reference-based correctness judge
├── faithfulness_evaluator.py  # PDF evidence and Claude batch judge
└── results_aggregator.py      # Combine judgments and report metrics
```

The commands, output JSON shapes, judge prompts and API safeguards are unchanged
from v1.2. Keep all Python files in the same directory so their imports resolve.


python evaluation/generation_quality/evaluate.py \
  prepare-batch \
  --predictions evaluation/results_generation/v3_dataset/gemini_flash_lite/image_only_test/gemini-3.5-flash-lite/retrieved/predictions.jsonl\
  --pdf-dir "/Users/sagar_rsv/Desktop/multimodal_doc_crag/modern_ai_corpus_raw_pdfs" \
  --qids T-016 F-016 \
  --output evaluation/generation_quality/eval_results/negative_requests.json \
  --manifest evaluation/generation_quality/eval_results/negative_manifest.jsonl
