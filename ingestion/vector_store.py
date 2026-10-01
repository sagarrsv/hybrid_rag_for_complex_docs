import os
import time
import uuid
import torch
from typing import List, Dict, Any
from qdrant_client import QdrantClient
from qdrant_client.http import models

class QdrantStorageNode:
    def __init__(
        self,
        url: str = None,
        api_key: str = None,
        path: str = "./qdrant_local_db"
    ):
        """
        Production vector database node for hybrid RAG.
        Enforces idempotency, payload indexing for neighbor-page lookups,
        and retry-resilient upserts.
        """
        if url:
            self.client = QdrantClient(url=url, api_key=api_key)
        else:
            self.client = QdrantClient(path=path)

        self.text_collection = "text_chunks"
        self.visual_collection = "visual_pages"
        self._init_collections()
        self._init_payload_indexes()

    def _init_collections(self):
        # 1. Text Collection: BGE-M3 (1024-dim dense vectors)
        if not self.client.collection_exists(self.text_collection):
            self.client.create_collection(
                collection_name=self.text_collection,
                vectors_config=models.VectorParams(
                    size=1024,
                    distance=models.Distance.COSINE
                )
            )
            print(f"Created Qdrant collection: {self.text_collection} (size=1024, Cosine)")

        # 2. Visual Collection: ColModernVBERT Multi-Vector (128-dim per token, MaxSim)
        if not self.client.collection_exists(self.visual_collection):
            self.client.create_collection(
                collection_name=self.visual_collection,
                vectors_config=models.VectorParams(
                    size=128,
                    distance=models.Distance.COSINE,
                    multivector_config=models.MultiVectorConfig(
                        comparator=models.MultiVectorComparator.MAX_SIM
                    )
                )
            )
            print(f"Created Qdrant collection: {self.visual_collection} (multivector=128, MaxSim)")

    def _init_payload_indexes(self): #creates metadata for pagenum and docid
        """Creates indexes for fast metadata filtering (neighbor-page retrieval)."""
        collections = [self.text_collection, self.visual_collection]
        for col in collections:
            try:
                self.client.create_payload_index(
                    collection_name=col,
                    field_name="doc_id",
                    field_schema=models.PayloadSchemaType.KEYWORD
                )
                self.client.create_payload_index(
                    collection_name=col,
                    field_name="page_num",
                    field_schema=models.PayloadSchemaType.INTEGER
                )
            except Exception:
                # Passes cleanly if indexes are already initialized
                pass

    @staticmethod
    def _deterministic_uuid(key: str) -> str: #Convert chunk id to deterministic uuid
        """Derives a deterministic UUIDv5 so reruns overwrite rather than duplicate."""
        return str(uuid.uuid5(uuid.NAMESPACE_DNS, key))

    def _retry_upsert(self, collection_name: str, points: list, max_retries: int = 3):
        """Wraps upserts in an exponential backoff loop for network stability."""
        for attempt in range(1, max_retries + 1):
            try:
                self.client.upsert(
                    collection_name=collection_name,
                    points=points
                )
                return
            except Exception as e:
                if attempt == max_retries:
                    print(f"❌ Failed to upsert to {collection_name} after {max_retries} attempts: {e}")
                    raise e
                sleep_time = attempt * 2
                print(f"⚠️ Upsert warning: {e}. Retrying in {sleep_time}s (attempt {attempt}/{max_retries})...")
                time.sleep(sleep_time)

    def upsert_text_records(self, records: List[Dict[str, Any]]):
        if not records:
            return

        points = []
        for r in records:
            points.append(
                models.PointStruct(
                    id=self._deterministic_uuid(r["chunk_id"]),
                    vector=r["vector"],
                    payload={
                        "chunk_id": r["chunk_id"],
                        "doc_id": r["doc_id"],
                        "page_num": r["page_num"],
                        "modality": r["modality"],
                        "text": r["text"],
                        "metadata": r.get("metadata", {})
                    }
                )
            )

        self._retry_upsert(self.text_collection, points)
        print(f"   Stored {len(points)} text vectors into '{self.text_collection}' (Deterministic IDs).")

    def upsert_visual_records(self, records: List[Dict[str, Any]]):
        if not records:
            return

        points = []
        for r in records:
            tensor = r["multivector"]
            if isinstance(tensor, torch.Tensor):
                multivec_list = tensor.cpu().float().numpy().tolist()
            else:
                multivec_list = tensor

            points.append(
                models.PointStruct(
                    id=self._deterministic_uuid(r["chunk_id"]),
                    vector=multivec_list,  # List of lists: [[128], [128], ...]
                    payload={
                        "chunk_id": r["chunk_id"],
                        "doc_id": r["doc_id"],
                        "page_num": r["page_num"],
                        "modality": r["modality"],
                        # "doc_path": r.get("doc_path", ""),
                        "n_vectors": r.get("n_vectors", len(multivec_list)),
                        "metadata": r.get("metadata", {})
                    }
                )
            )

        self._retry_upsert(self.visual_collection, points)
        print(f"   Stored {len(points)} multivector page(s) into '{self.visual_collection}' (Deterministic IDs).")