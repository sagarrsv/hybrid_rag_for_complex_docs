import os
import glob
import time
import json
import argparse
from ingestion.doc_classifier import PageClassifier
from ingestion.text_ingestion import TextDenseIngestionNode
from ingestion.visual_ingestion import ColModernVBertPipeline
from ingestion.orchestrator import DocumentOrchestrator
from ingestion.vector_store import QdrantStorageNode

def parse_args():
    parser = argparse.ArgumentParser(description="Multi-document batch ingestion for research papers")
    parser.add_argument(
        "--data_dir",
        type=str,
        default="/kaggle/input/datasets/sagarrsv/research-docs",
        help="Path to folder containing PDF papers"
    )
    parser.add_argument(
        "--max_docs",
        type=int,
        default=None,
        help="Optional limit on number of documents to ingest (useful for quick checks)"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./ingestion_output",
        help="Directory to save the processed metadata checkpoints"
    )
    parser.add_argument(
        "--local_db",
        action="store_true",
        help="If set, uses local disk storage path './qdrant_local_db' instead of Qdrant Cloud"
    )
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    # 1. Collect all PDF file paths in the directory
    search_path = os.path.join(args.data_dir, "**/*.pdf")
    pdf_files = sorted(glob.glob(search_path, recursive=True))

    if not pdf_files:
        print(f"No PDF files found in: {args.data_dir}")
        return

    if args.max_docs:
        pdf_files = pdf_files[:args.max_docs]

    print(f"Found {len(pdf_files)} PDF document(s) in {args.data_dir}")

    # 2. Connect to Qdrant (Cloud or Local disk)
    qdrant_url = os.getenv("QDRANT_URL")
    qdrant_api_key = os.getenv("QDRANT_API_KEY")

    if qdrant_url and qdrant_api_key:
        print(f"Connecting to Qdrant Cloud: {qdrant_url}")
        storage_node = QdrantStorageNode(url=qdrant_url, api_key=qdrant_api_key)
    else:
        print("Initializing Qdrant locally at './qdrant_local_db'...")
        storage_node = QdrantStorageNode(path="./qdrant_local_db")

    # 3. Initialize models ONCE (keeps weights resident in GPU memory)
    print("\n[Init] Initializing Classifier and Embedding Pipelines...")
    classifier = PageClassifier()
    text_node = TextDenseIngestionNode()  # BAAI/bge-m3 local
    visual_node = ColModernVBertPipeline()  # ColModernVBERT local

    orchestrator = DocumentOrchestrator(
        classifier=classifier,
        text_node=text_node,
        visual_node=visual_node,
        storage_node=storage_node
    )

    # 3. Batch Loop over each PDF
    total_start = time.time()
    corpus_summary = {
        "total_documents": len(pdf_files),
        "total_pages": 0,
        "text_dense_pages": 0,
        "layout_heavy_pages": 0,
        "text_chunks": 0,
        "docs_processed": []
    }

    for idx, pdf_path in enumerate(pdf_files, 1):
        doc_filename = os.path.basename(pdf_path)
        print(f"\n=======================================================")
        print(f"[{idx}/{len(pdf_files)}] Ingesting Document: {doc_filename}")
        print(f"=======================================================")

        try:
            # Process single PDF through classifier and routed embedding nodes
            result = orchestrator.ingest_document(pdf_path)

            # Extract metrics
            m = result["metrics"]
            total_pages = result["total_pages"]
            text_p = m["text_pages"]
            vis_p = m["visual_pages"]
            n_chunks = len(result["text_records"])
            
            # Aggregate corpus totals
            corpus_summary["total_pages"] += total_pages
            corpus_summary["text_dense_pages"] += text_p
            corpus_summary["layout_heavy_pages"] += vis_p
            corpus_summary["text_chunks"] += n_chunks

            # Record per-document metrics for the ablation / systems table
            corpus_summary["docs_processed"].append({
                "doc_id": result["doc_id"],
                "total_pages": total_pages,
                "text_pages": text_p,
                "visual_pages": vis_p,
                "visual_pct": round((vis_p / max(total_pages, 1)) * 100, 1),
                "text_chunks": n_chunks,
                "text_time_sec": round(m["text_time"], 2),
                "visual_time_sec": round(m["visual_time"], 2),
                "avg_text_sec_per_page": round(m["text_time"] / max(text_p, 1), 3),
                "avg_vis_sec_per_page": round(m["visual_time"] / max(vis_p, 1), 3)
            })

            # Checkpoint metadata to disk (avoid keeping all multivectors in RAM)
            meta_save_path = os.path.join(args.output_dir, f"{result['doc_id']}_meta.json")
            with open(meta_save_path, "w") as f:
                json.dump({
                    "doc_id": result["doc_id"],
                    "total_pages": result["total_pages"],
                    "metrics": m,
                    "visual_pct": result["visual_pct"]
                }, f, indent=2)

        except Exception as e:
            print(f"Error processing {pdf_path}: {e}")

    # 4. Final Corpus Logging
    total_time = time.time() - total_start

    # 1. Save all per-document records into a single central JSON file
    summary_path = os.path.join(args.output_dir, "corpus_ingestion_report.json")
    with open(summary_path, "w") as f:
        json.dump(corpus_summary, f, indent=2)
    print(f"\n📁 Saved per-document metrics report to: {summary_path}")

    # 2. Print Document-by-Document Breakdown Table
    print("\n" + "=" * 95)
    print(f"{'Doc ID':<35} | {'Pages':<5} | {'Txt/Vis':<8} | {'Chunks':<6} | {'Txt Lat (s)':<11} | {'Vis Lat (s)':<11}")
    print("-" * 95)
    for doc in corpus_summary["docs_processed"]:
        txt_vis_ratio = f"{doc['text_pages']}/{doc['visual_pages']}"
        print(f"{doc['doc_id'][:35]:<35} | {doc['total_pages']:<5} | {txt_vis_ratio:<8} | {doc['text_chunks']:<6} | {doc['text_time_sec']:<11} | {doc['visual_time_sec']:<11}")
    print("=" * 95)

    # 3. Final Overall Ingestion Summary
    avg_page_time = total_time / max(corpus_summary["total_pages"], 1)
    print("\n================== CORPUS INGESTION COMPLETE ==================")
    print(f"Total Documents Processed : {len(corpus_summary['docs_processed'])}")
    print(f"Total Pages Processed     : {corpus_summary['total_pages']}")
    print(f"Total Text-Dense Pages    : {corpus_summary['text_dense_pages']} ({corpus_summary['text_chunks']} chunks)")
    print(f"Total Layout-Heavy Pages  : {corpus_summary['layout_heavy_pages']}")
    print(f"Total Wall Time           : {total_time:.2f}s ({avg_page_time:.2f}s/page)")
    print("=" * 95 + "\n")


if __name__ == "__main__":
    main()