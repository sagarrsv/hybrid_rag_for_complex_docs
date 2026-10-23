import os
import json
import numpy as np
from typing import List, Dict, Any

class DiskStorageExporter:
    def __init__(self, output_dir: str = "./kaggle_vector_dump"):
        self.output_dir = output_dir
        self.text_dir = os.path.join(output_dir, "text_data")
        self.visual_dir = os.path.join(output_dir, "visual_data")
        os.makedirs(self.text_dir, exist_ok=True)
        os.makedirs(self.visual_dir, exist_ok=True)

    def export_text_records(self, doc_id: str, records: List[Dict[str, Any]]):
        """
        Saves text dense vectors (BGE-M3 1024-d) as FP16 .npz and metadata as JSON.
        """
        if not records:
            return

        chunk_ids = [r["chunk_id"] for r in records]
        # Cast 1024-d float32 embeddings to float16
        vectors_fp16 = np.array([r["vector"] for r in records], dtype=np.float16)

        metadata = [
            {
                "chunk_id": r["chunk_id"],
                "doc_id": r["doc_id"],
                "page_num": r["page_num"],
                "page_type": r["page_type"],  # 'text-dense' or 'layout-heavy'
                "text": r["text"],
                "metadata": r.get("metadata", {})
            }
            for r in records
        ]

        # Save vectors as compressed npz
        np.savez_compressed(
            os.path.join(self.text_dir, f"{doc_id}_text_vecs.npz"),
            chunk_ids=np.array(chunk_ids),
            vectors=vectors_fp16
        )

        # Save text payloads
        with open(os.path.join(self.text_dir, f"{doc_id}_text_meta.json"), "w") as f:
            json.dump(metadata, f)

    def export_visual_records(self, doc_id: str, records: List[Dict[str, Any]]):
        """
        Saves ColModernVBERT multivector embeddings (1149 x 128 per page) in FP16 .npz.
        """
        if not records:
            return

        chunk_ids = [r["chunk_id"] for r in records]
        # Tensor/array shape: (num_pages, 1149, 128) in float16
        multivectors_fp16 = np.stack([r["multivector"] for r in records]).astype(np.float16)

        metadata = [
            {
                "chunk_id": r["chunk_id"],
                "doc_id": r["doc_id"],
                "page_num": r["page_num"],
                "page_type": r["page_type"],  # 'text-dense' or 'layout-heavy'
                "n_vectors": r.get("n_vectors", 1149),
                "metadata": r.get("metadata", {})
            }
            for r in records
        ]

        np.savez_compressed(
            os.path.join(self.visual_dir, f"{doc_id}_visual_vecs.npz"),
            chunk_ids=np.array(chunk_ids),
            multivectors=multivectors_fp16
        )

        with open(os.path.join(self.visual_dir, f"{doc_id}_visual_meta.json"), "w") as f:
            json.dump(metadata, f)