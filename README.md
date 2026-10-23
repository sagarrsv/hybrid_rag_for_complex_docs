# Hybrid Multimodal RAG for Complex Document QA

*Balancing retrieval accuracy, vector storage efficiency, and ingestion cost for complex technical documents.*

## Overview

Traditional text-based RAG systems struggle to retrieve information from complex document layouts containing tables, figures, and equations. While visual retrieval can better preserve this information, indexing every page using multivector embeddings introduces substantial vector storage and ingestion overhead.

This project explores the **trade-off between retrieval effectiveness and indexing efficiency** through a multimodal RAG pipeline built with BGE-M3, ColModernVBERT, Qdrant, and LangGraph. It evaluates five retrieval strategies and develops **Hybrid V2**, which indexes text across all pages while selectively indexing layout-heavy pages visually, combining both retrieval channels using decayed Reciprocal Rank Fusion.

Evaluated on 78 curated questions across 95 research papers, Hybrid V2 achieved **87.8% Recall@3**, with estimated reductions of **36.1% in raw vector storage** and **25.8% in embedding/classification processing time** compared with full visual indexing. The end-to-end RAG pipeline using Gemini 3.8 Flash achieved **80.8% answer correctness**, with page-level citations and separate faithfulness evaluation.
