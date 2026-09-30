import os
import glob
import time
import json
import argparse
from ingestion.doc_classifier import PageClassifier
from ingestion.text_ingestion import TextDenseIngestionNode
from ingestion.visual_ingestion import ColModernVBertPipeline
from ingestion.orchestrator import DocumentOrchestrator

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

    # 2. Initialize models ONCE (keeps weights resident in GPU memory)
    print("\n[Init] Initializing Classifier and Embedding Pipelines...")
    classifier = PageClassifier()
    text_node = TextDenseIngestionNode()  # BAAI/bge-m3 local
    visual_node = ColModernVBertPipeline()  # ColModernVBERT local

    orchestrator = DocumentOrchestrator(
        classifier=classifier,
        text_node=text_node,
        visual_node=visual_node
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
            corpus_summary["total_pages"] += result["total_pages"]
            corpus_summary["text_dense_pages"] += m["text_pages"]
            corpus_summary["layout_heavy_pages"] += m["visual_pages"]
            corpus_summary["text_chunks"] += len(result["text_records"])
            corpus_summary["docs_processed"].append({
                "doc_id": result["doc_id"],
                "pages": result["total_pages"],
                "text_pages": m["text_pages"],
                "visual_pages": m["visual_pages"]
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
    print("\n================== CORPUS INGESTION COMPLETE ==================")
    print(f"Total Documents Processed : {len(corpus_summary['docs_processed'])}")
    print(f"Total Pages Processed     : {corpus_summary['total_pages']}")
    print(f"Total Text-Dense Pages    : {corpus_summary['text_dense_pages']} (Generated {corpus_summary['text_chunks']} chunks)")
    print(f"Total Layout-Heavy Pages  : {corpus_summary['layout_heavy_pages']}")
    print(f"Total Wall Time           : {total_time:.2f}s ({total_time / max(corpus_summary['total_pages'], 1):.2f}s/page)")
    print("===============================================================\n")

if __name__ == "__main__":
    main()