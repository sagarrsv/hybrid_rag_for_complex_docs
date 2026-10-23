# Hybrid Multimodal RAG for Complex Document QA

*Balancing retrieval accuracy, vector storage efficiency, and ingestion cost for complex technical documents.*

## Overview

Traditional text-based RAG systems struggle to retrieve information from complex document layouts containing tables, figures, and equations. While visual retrieval can better preserve this information, indexing every page using multivector embeddings introduces substantial vector storage and ingestion overhead.

This project explores the **trade-off between retrieval effectiveness and indexing efficiency** through a multimodal RAG pipeline built with BGE-M3, ColModernVBERT, Qdrant, and LangGraph. It evaluates five retrieval strategies and develops **Hybrid V2**, which indexes text across all pages while selectively indexing layout-heavy pages visually, combining both retrieval channels using decayed Reciprocal Rank Fusion.

Evaluated on 78 curated questions across 95 research papers, Hybrid V2 achieved **87.8% Recall@3**, with estimated reductions of **36.1% in raw vector storage** and **25.8% in embedding/classification processing time** compared with full visual indexing. The end-to-end RAG pipeline using Gemini 3.8 Flash achieved **80.8% answer correctness**, with page-level citations and separate faithfulness evaluation.

Highlights

Efficient Hybrid Ingestion: Selective visual indexing reduces vector storage and ingestion overhead.

Five Retrieval Strategies: Systematic ablations comparing text, visual, routed, and fused retrieval approaches.

Grounded Multimodal QA: LangGraph-based answer generation with page citations and evaluated answer quality.

System Architecture

<img width="968" height="1774" alt="mermaid-diagram (1)" src="https://github.com/user-attachments/assets/42e74641-71ce-4922-bd8e-9ae726062430" />


The system consists of three main stages:

Document Ingestion: PDFs are classified by page layout and processed into text and visual embeddings. Hybrid V2 retains text embeddings for all pages and requires visual embeddings only for layout-heavy pages.

Hybrid Retrieval: BGE-M3 retrieves relevant text while ColModernVBERT searches layout-heavy pages. Decayed Reciprocal Rank Fusion combines both rankings and selects the top relevant pages.

Answer Generation: A LangGraph pipeline assembles retrieved page text and images, passes them to a multimodal LLM, and returns an evidence-grounded answer with page-level citations.
