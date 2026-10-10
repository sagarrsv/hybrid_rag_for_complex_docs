"""
retriever.py
====================================================================
Unified retrieval dispatcher for all 4 study modes:
  1. text: Full Text Baseline (Overfetched & collapsed to pages)
  2. visual: Full Visual Baseline (ColModernVBert MaxSim, native page granularity)
  3. hybrid_routed: Text-dense text (collapsed) + Layout-heavy visual fused via RRF
  4. fusion_unrouted: Unfiltered text (collapsed) + Unfiltered visual fused via RRF
====================================================================
"""

import time
import numpy
from typing import List, Dict, Any, Optional, Tuple
import torch
from sentence_transformers import SentenceTransformer
from colpali_engine.models import ColModernVBert, ColModernVBertProcessor
from qdrant_client import QdrantClient, models

from config import (
    QDRANT_URL,
    QDRANT_TIMEOUT,
    TEXT_COLLECTION,
    VISUAL_COLLECTION,
    TEXT_MODEL_NAME,
    VIS_MODEL_NAME,
    DEVICE,
    VIS_QUERY_DTYPE,
    PT_TEXT,
    PT_VISUAL,
    MAX_K,
    CANDIDATE_POOL,
    TEXT_OVERFETCH,
    RRF_K,
)


class MultimodalRetriever:
    def __init__(self):
        print(f"[Retriever] Connecting to Qdrant at {QDRANT_URL}...")
        self.qdrant = QdrantClient(url=QDRANT_URL, timeout=QDRANT_TIMEOUT)

        print(f"[Retriever] Loading Text Embedder ({TEXT_MODEL_NAME}) on {DEVICE}...")
        self.text_model = SentenceTransformer(TEXT_MODEL_NAME, device=str(DEVICE))

        print(f"[Retriever] Loading ColModernVBert ({VIS_MODEL_NAME}) on {DEVICE}...")
        self.vis_processor = ColModernVBertProcessor.from_pretrained(VIS_MODEL_NAME)
        self.vis_model = ColModernVBert.from_pretrained(
            VIS_MODEL_NAME,
            trust_remote_code=True,
            dtype=VIS_QUERY_DTYPE           
        ).to(DEVICE)
        self.vis_model.eval()

    def _collapse_to_pages(self, hits: List[Dict[str, Any]], k: int) -> List[Dict[str, Any]]:
        """
        Deduplicates multiple text chunks belonging to the same page.
        Keeps the highest-scoring chunk per (doc_id, page_num) and caps at k.
        """
        best: Dict[Tuple[str, int], Dict[str, Any]] = {}
        for h in hits:
            key = (h["doc_id"], h["page_num"])
            if key not in best or h["score"] > best[key]["score"]:
                best[key] = h
        return sorted(best.values(), key=lambda h: -h["score"])[:k]
    
    def _text_filter(self) -> models.Filter:
        return models.Filter(
            must=[models.FieldCondition(key="page_type", match=models.MatchValue(value="text-dense"))]
        )

    def _layout_filter(self) -> models.Filter:
        return models.Filter(
            must=[models.FieldCondition(key="page_type", match=models.MatchValue(value="layout-heavy"))]
        )
    def _encode_visual_query(self, query: str) -> List[List[float]]:
        """Encodes query text using ColModernVBertProcessor.process_queries."""
        batch_queries = self.vis_processor.process_queries([query]).to(DEVICE)
        with torch.no_grad():
            query_embeddings = self.vis_model(**batch_queries)
        return query_embeddings[0].cpu().float().numpy().tolist()
    
    def _execute_text_query(
        self,
        query_vector: List[float],
        limit: int,
        filter_condition: Optional[models.Filter]
    ) -> List[Dict[str, Any]]:
        hits = self.qdrant.query_points(
            collection_name=TEXT_COLLECTION,
            query=query_vector,
            query_filter=filter_condition,
            limit=limit
        ).points
        return [{
            "source": "text_chunks",
            "chunk_id": h.payload.get("chunk_id"),
            "doc_id": h.payload.get("doc_id"),
            "page_num": int(h.payload.get("page_num")),
            "page_type": h.payload.get("page_type"),
            "content": h.payload.get("text", ""),
            "score": float(h.score)
        } for h in hits]

    def _execute_visual_query(
        self,
        query_vector: List[List[float]],
        limit: int,
        filter_condition: Optional[models.Filter]
    ) -> List[Dict[str, Any]]:
        hits = self.qdrant.query_points(
            collection_name=VISUAL_COLLECTION,
            query=query_vector,
            query_filter=filter_condition,
            limit=limit
        ).points
        return [{
            "source": "visual_pages",
            "chunk_id": h.payload.get("chunk_id"),
            "doc_id": h.payload.get("doc_id"),
            "page_num": int(h.payload.get("page_num")),
            "page_type": h.payload.get("page_type"),
            "content": "",
            "score": float(h.score)
        } for h in hits]
    # ----------------------------------------------------
    # Unified Dispatcher
    # ----------------------------------------------------
    # === MODIFIED / ADDED: 1. Two standardized pool helpers ===
    def _text_pool(self, dense_vec: list, flt: Optional[models.Filter] = None) -> List[Dict[str, Any]]:
            raw = self._execute_text_query(dense_vec, CANDIDATE_POOL * TEXT_OVERFETCH, flt)
            return self._collapse_to_pages(raw, CANDIDATE_POOL)

    def _vis_pool(self, query_multivec: list, flt: Optional[models.Filter] = None) -> List[Dict[str, Any]]:
        return self._execute_visual_query(query_multivec, CANDIDATE_POOL, flt)
    # ----------------------------------------------------
    # Fusion and Query Helpers
    # ----------------------------------------------------
    def _fuse_rrf(
        self,
        text_hits: List[Dict[str, Any]],
        vis_hits: List[Dict[str, Any]],
        k: int
    ) -> List[Dict[str, Any]]:
        """Reciprocal Rank Fusion on page-level candidate pools."""
        doc_scores: Dict[Tuple[str, int], float] = {}
        doc_map: Dict[Tuple[str, int], Dict[str, Any]] = {}

        for rank, doc in enumerate(text_hits, 1):
            key = (doc["doc_id"], doc["page_num"])
            doc_scores[key] = doc_scores.get(key, 0.0) + (1.0 / (RRF_K + rank))
            doc_map[key] = doc

        for rank, doc in enumerate(vis_hits, 1):
            key = (doc["doc_id"], doc["page_num"])
            doc_scores[key] = doc_scores.get(key, 0.0) + (1.0 / (RRF_K + rank))
            if key not in doc_map or doc_map[key]["source"] == "text_chunks":
                doc_map[key] = doc

        ranked_keys = sorted(doc_scores.keys(), key=lambda x: doc_scores[x], reverse=True)
        results = []
        for key in ranked_keys[:k]:
            item = dict(doc_map[key])
            item["score"] = doc_scores[key]
            item["rrf_score"] = doc_scores[key]
            results.append(item)
        return results
    


    # ----------------------------------------------------
    # Mode 1: Full Text (Over-fetched by 4x, then collapsed to pages)
    # ----------------------------------------------------
    def search_full_text(self, query: str, k: int = MAX_K) -> List[Dict[str, Any]]:
        dense_vec = self.text_model.encode(query, normalize_embeddings=True).tolist()
        return self._text_pool(dense_vec, flt=None)[:k]

    # ----------------------------------------------------
    # Mode 2: Full Visual (Natively 1 vector per page, no collapse needed)
    # ----------------------------------------------------
    def search_full_visual(self, query: str, k: int = MAX_K) -> List[Dict[str, Any]]:
        query_multivec = self._encode_visual_query(query)
        return self._vis_pool(query_multivec, flt=None)[:k]

    # ----------------------------------------------------
    # Mode 3: Hybrid Routed (Strictly partitioned by page_type)
    # ----------------------------------------------------
    def search_hybrid_routed(self, query: str, k: int = MAX_K) -> List[Dict[str, Any]]:
        # Text Arm: filter text-dense, overfetch, collapse to CANDIDATE_POOL
        dense_vec = self.text_model.encode(query, normalize_embeddings=True).tolist()
        multivec = self._encode_visual_query(query)
        text_pool = self._text_pool(dense_vec, self._text_filter())
        vis_pool = self._vis_pool(multivec, self._layout_filter())
        return self._fuse_rrf(text_pool, vis_pool, k), {"text": text_pool, "visual": vis_pool}

    # ----------------------------------------------------
    # Mode 4: Fusion Unrouted (Ensemble Baseline)
    # ----------------------------------------------------
    def search_unrouted_fusion(self, query: str, k: int = MAX_K) -> List[Dict[str, Any]]:
        # Text Arm: unfiltered, overfetch, collapse to CANDIDATE_POOL
        dense_vec = self.text_model.encode(query, normalize_embeddings=True).tolist()
        multivec = self._encode_visual_query(query)
        text_pool = self._text_pool(dense_vec, None)
        vis_pool = self._vis_pool(multivec, None)
        return self._fuse_rrf(text_pool, vis_pool, k), {"text": text_pool, "visual": vis_pool}
    
    # === MODIFIED / ADDED: 2. Add hybrid_v2 mode ===

    def search_hybrid_v2(self, query: str, k: int = MAX_K):
        text_pool = self._text_pool(self.text_model.encode(query, normalize_embeddings=True).tolist(), None)
        vis_pool = self._vis_pool(self._encode_visual_query(query), self._layout_filter())
        return self._fuse_rrf(text_pool, vis_pool, k), {"text": text_pool, "visual": vis_pool} 


    # === MODIFIED / ADDED: 4. Dispatcher supporting hybrid_v2 and return_pools ===
    def search(self, mode: str, query: str, k: int = MAX_K, return_pools: bool = False):
        t0 = time.perf_counter()
        pools = None
        if mode == "text":
            hits = self.search_full_text(query, k)
        elif mode == "visual":
            hits = self.search_full_visual(query, k)
        elif mode == "hybrid_routed":
            hits, pools = self.search_hybrid_routed(query, k)
        elif mode == "fusion_unrouted":
            hits, pools = self.search_unrouted_fusion(query, k)
        elif mode == "hybrid_v2":
            hits, pools = self.search_hybrid_v2(query, k)
        else:
            raise ValueError(mode)
        latency = time.perf_counter() - t0
        return (hits, latency, pools) if return_pools else (hits, latency)

    

    # ----------------------------------------------------
    # Harness Validation & Benchmarking Helpers
    # ----------------------------------------------------
    def get_page_type(self, doc_id: str, page_num: int) -> Optional[str]:
        """Scrolls visual_pages by doc_id and page_num without reading heavy vectors."""
        res, _ = self.qdrant.scroll(
            collection_name=VISUAL_COLLECTION,
            scroll_filter=models.Filter(
                must=[
                    models.FieldCondition(key="doc_id", match=models.MatchValue(value=doc_id)),
                    models.FieldCondition(key="page_num", match=models.MatchValue(value=page_num)),
                ]
            ),
            limit=1,
            with_vectors=False,
            with_payload=True
        )
        if res:
            return res[0].payload.get("page_type")
        return None

    def id_exists(self, doc_id: str, page_num: int) -> bool:
        """Confirms that a target gold page exists in the vector store."""
        return self.get_page_type(doc_id, page_num) is not None

    def fetch_page_text(self, doc_id: str, page_num: int) -> str:
        """Fetches and concatenates all text chunks belonging to a document page."""
        hits, _ = self.qdrant.scroll(
            collection_name=TEXT_COLLECTION,
            scroll_filter=models.Filter(
                must=[
                    models.FieldCondition(key="doc_id", match=models.MatchValue(value=doc_id)),
                    models.FieldCondition(key="page_num", match=models.MatchValue(value=page_num)),
                ]
            ),
            limit=20,
            with_vectors=False,
            with_payload=True
        )
        chunks = [h.payload.get("text", "") for h in hits if h.payload.get("text")]
        return "\n".join(chunks)

    def warmup(self):
        """Runs one query through each arm to clear first-run on-disk index latency."""
        print("[Retriever] Warming up vector indices and query encoders...")
        dummy_query = "benchmark system warmup query"
        for mode in ["text", "visual", "hybrid_routed", "fusion_unrouted"]:
            self.search(mode, dummy_query, k=1)
        print("[Retriever] Warmup complete.")