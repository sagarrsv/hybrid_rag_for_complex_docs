# Hybrid V2 — Ingestion Efficiency

## Corpus and indexing strategy

The experiment covers **95 PDFs, 2,670 pages, and 11,444 text chunks**. Page classification identified **1,129 text-dense pages (42.3%)** and **1,541 layout-heavy pages (57.7%)**.

- **Full Visual (baseline):** ColModernVBert visual embeddings for all 2,670 pages.
- **Hybrid V2:** BGE-M3 text embeddings for **all 11,444 chunks**, plus ColModernVBert visual embeddings for **only the 1,541 layout-heavy pages**. Classification is performed for all pages.

The original experimental ingestion generated both embedding types for every page to support retrieval ablations. Hybrid V2-only ingestion costs below are **estimates based on those measurements**, not a separately timed deployment.

## Estimated raw vector storage

Assumptions: **BF16** visual multivectors (2 bytes/value), **Float32** BGE-M3 text vectors (1,024 dimensions, 4 bytes/value), and the logged visual matrices of **885 × 128** or **1,149 × 128** per page.

|   Stored vectors  | Full Visual | Hybrid V2  |
|-------------------|------------:|-----------:|
| Visual embeddings | 707.22 MiB  | 407.01 MiB |
| Text embeddings   | —           | 44.70 MiB  |
| **Total raw vectors** | **707.22 MiB** | **451.72 MiB** |

**Estimated storage saving:** **255.50 MiB (36.13%)** relative to Full Visual.

Calculation: `(707.22 − 451.72) / 707.22 × 100 ≈ 36.13%`.

## Estimated ingestion processing time

Calculated from the recorded per-page embedding and classification timings.

| Timed Component | Full Visual | Hybrid V2 |
|:----------------|------------:|----------:|
| Text Embedding (All Pages) | — | 11.37 min |
| Visual Embedding | 117.66 min | 67.96 min |
| Page Classification | — | 8.04 min |
| **Total Modeled Processing** | **117.66 min** | **87.36 min** |

**Estimated time saving:** **30.30 min (25.75%)** relative to Full Visual.

Calculation: `(117.66 − 87.36) / 117.66 × 100 ≈ 25.75%`.

## Interpretation and limitations

Hybrid V2 avoids visual embeddings for **42.3% of pages** while retaining text representations across the entire corpus. Its estimated benefits are **36.1% lower raw vector storage** and **25.8% less embedding-and-classification processing time** than Full Visual.

These are **modeled estimates**, not measured reductions in Qdrant's total disk usage, cloud charges, end-to-end ingestion wall time, or online query latency. Storage excludes payloads, indexes, and other Qdrant overhead. The ingestion measurements came from a full dual-modal run.

**Sources:** `corpus_export_report.json` (corpus statistics and per-document timings) and the original page-level ingestion log (visual matrix shapes and processing timings).
