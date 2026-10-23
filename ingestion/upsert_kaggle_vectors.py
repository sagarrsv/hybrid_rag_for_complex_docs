import os
import glob
import json
import uuid
import numpy as np
import argparse
from qdrant_client import QdrantClient
from qdrant_client.http import models

TEXT_COLLECTION = "text_chunks"
VISUAL_COLLECTION = "visual_pages"

def deterministic_uuid(key: str) -> str:
    """Derives a deterministic UUIDv5 so re-upserting overwrites rather than duplicates."""
    return str(uuid.uuid5(uuid.NAMESPACE_DNS, key))

def init_collections(client, reset: bool = False):
    # 1. Text Dense Collection: BGE-M3 (1024-dim, Cosine)
    # Standard HNSW is fine here (~8k total chunks, negligible RAM)
    if not client.collection_exists("text_chunks"):
        client.create_collection(
            collection_name="text_chunks",
            vectors_config=models.VectorParams(
                size=1024,
                distance=models.Distance.COSINE,
                on_disk=False
            )
        )
        print("Created collection: text_chunks")

    # 2. Visual Multi-Vector: ColModernVBERT (128-dim per token, MaxSim)
    # Disable HNSW (m=0) -> pure brute-force MaxSim over mmap files
    if not client.collection_exists("visual_pages"):
        client.create_collection(
            collection_name="visual_pages",
            vectors_config=models.VectorParams(
                size=128,
                distance=models.Distance.COSINE,
                datatype=models.Datatype.FLOAT16,  # 50% footprint reduction in Qdrant
                on_disk=True,  # Crucial: memory-map vectors to avoid Docker OOM
                multivector_config=models.MultiVectorConfig(
                    comparator=models.MultiVectorComparator.MAX_SIM
                )
            ),
            hnsw_config=models.HnswConfigDiff(m=0)  # m=0 disables HNSW indexing entirely
        )
        print("Created collection: visual_pages (on_disk=True, brute-force MaxSim)")

    # 3. Payload Indexes
    for col in ["text_chunks", "visual_pages"]:
        for field, schema in [
            ("doc_id", models.PayloadSchemaType.KEYWORD),
            ("page_num", models.PayloadSchemaType.INTEGER),
            ("page_type", models.PayloadSchemaType.KEYWORD)
        ]:
            try:
                client.create_payload_index(collection_name=col, field_name=field, field_schema=schema)
            except Exception:
                pass

def upsert_text_branch(client: QdrantClient, data_dir: str, batch_size: int = 128):
    text_dir = os.path.join(data_dir, "text_data")
    npz_files = sorted(glob.glob(os.path.join(text_dir, "*_text_vecs.npz")))
    total_docs = len(npz_files)

    print(f"\n=======================================================")
    print(f" Starting Text Upsert: {total_docs} Documents")
    print(f"=======================================================")

    for npz_path in npz_files:
        doc_prefix = npz_path.replace("_text_vecs.npz", "")
        meta_path = f"{doc_prefix}_text_meta.json"
        
        if not os.path.exists(meta_path):
            continue

        with np.load(npz_path) as npz_data:
            chunk_ids = npz_data["chunk_ids"].tolist()
            # Cast FP16 -> Float32 -> Python float list for Qdrant payload
            vectors = npz_data["vectors"].astype(np.float32).tolist()

        with open(meta_path, "r") as f:
            metadata_list = json.load(f)

        assert len(chunk_ids) == len(metadata_list) == len(vectors), (
            f"Text length mismatch in {doc_prefix}: IDs={len(chunk_ids)}, Meta={len(metadata_list)}, Vecs={len(vectors)}"
        )

        points = []
        for c_id, vec, meta in zip(chunk_ids, vectors, metadata_list):
            points.append(
                models.PointStruct(
                    id=deterministic_uuid(c_id),
                    vector=vec,
                    payload=meta
                )
            )

        # Batch upsert to Qdrant
        for i in range(0, len(points), batch_size):
            client.upsert(
                collection_name=TEXT_COLLECTION,
                points=points[i : i + batch_size],
                wait=True
            )
        print(f"Stored {len(points)} text chunks for: {os.path.basename(doc_prefix)}")


def upsert_visual_branch(client: QdrantClient, data_dir: str, batch_size: int = 4, log_interval: int = 5):
    vis_dir = os.path.join(data_dir, "visual_data")
    npz_files = sorted(glob.glob(os.path.join(vis_dir, "*_visual_vecs.npz")))
    total_docs = len(npz_files)

    print(f"\n=======================================================")
    print(f"--- Upserting Visual Records: {total_docs} Documents (log every {log_interval} batches) ---")
    print(f"=======================================================")
    
    for doc_idx, npz_path in enumerate(npz_files, 1):
        doc_prefix = npz_path.replace("_visual_vecs.npz", "")
        meta_path = f"{doc_prefix}_visual_meta.json"
        doc_name = os.path.basename(doc_prefix)

        if not os.path.exists(meta_path):
            continue

        with np.load(npz_path) as npz_data:
            chunk_ids = npz_data["chunk_ids"].tolist()
            multivectors = npz_data["multivectors"]

        with open(meta_path, "r") as f:
            metadata_list = json.load(f)

        assert len(chunk_ids) == len(metadata_list) == len(multivectors), "Visual count mismatch!"

        total_pages = len(chunk_ids)
        total_batches = (total_pages + batch_size - 1) // batch_size

        for batch_num, i in enumerate(range(0, total_pages, batch_size), 1):
            batch_ids = chunk_ids[i : i + batch_size]
            batch_vecs = multivectors[i : i + batch_size].astype(np.float32).tolist()
            batch_meta = metadata_list[i : i + batch_size]

            points = [
                models.PointStruct(
                    id=deterministic_uuid(c_id),
                    vector=m_vec,
                    payload=meta
                )
                for c_id, m_vec, meta in zip(batch_ids, batch_vecs, batch_meta)
            ]

            client.upsert(collection_name=VISUAL_COLLECTION, points=points, wait=True)

            # Prints periodically (e.g., every 5 batches) or on the final batch
            if batch_num % log_interval == 0 or batch_num == total_batches:
                uploaded = min(i + batch_size, total_pages)
                print(f"   [Doc {doc_idx}/{total_docs}: {doc_name[:25]}] -> Progress: {uploaded}/{total_pages} pages (Batch {batch_num}/{total_batches})")

        print(f"✓ Completed {doc_name} ({total_pages} pages)")
        print(f"Stored {len(chunk_ids)} visual pages for: {os.path.basename(doc_prefix)}")
def main():
    parser = argparse.ArgumentParser(description="Upsert Kaggle vector dumps into local Docker Qdrant")
    parser.add_argument("--artifacts_dir", type=str, default="./vector_output", help="Path to exported vector folders")
    parser.add_argument("--qdrant_url", type=str, default="http://localhost:6333", help="Local Qdrant endpoint")
    parser.add_argument("--reset", action="store_true", help="Delete and recreate collections before upserting")
    args = parser.parse_args()

    # Client configured with 120s timeout
    client = QdrantClient(url=args.qdrant_url, timeout=120)

    init_collections(client, reset=args.reset)
    upsert_text_branch(client, args.artifacts_dir, batch_size=128)
    upsert_visual_branch(client, args.artifacts_dir, batch_size=4)

    print("\nLocal Qdrant Dual-Store Upsert Complete.")

if __name__ == "__main__":
    main()