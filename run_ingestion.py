import os
import glob
import time
import json
import argparse
from ingestion.doc_classifier import PageClassifier
from ingestion.text_ingestion import TextDenseIngestionNode
from ingestion.visual_ingestion import ColModernVBertPipeline
from ingestion.orchestrator import DocumentOrchestrator
from ingestion.file_exporter import DiskStorageExporter

def parse_args():
    parser = argparse.ArgumentParser(description="Multi-document batch vector export for research papers")
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
        help="Optional limit on number of documents to ingest"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./vector_output",
        help="Directory to save the vectors (.npz) and metadata (.json)"
    )
    return parser.parse_args()

def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    search_path = os.path.join(args.data_dir, "**/*.pdf")
    pdf_files = sorted(glob.glob(search_path, recursive=True))
    if not pdf_files:
        print(f"No PDF files found in: {args.data_dir}")
        return

    if args.max_docs:
        pdf_files = pdf_files[:args.max_docs]

    print(f"Found {len(pdf_files)} PDF document(s) in {args.data_dir}")
    print(f"Artifacts will be stored in: {os.path.abspath(args.output_dir)}")

    # 1. Initialize models ONCE in GPU memory
    print("\n[Init] Initializing Classifier and Embedding Pipelines...")
    classifier = PageClassifier()
    text_node = TextDenseIngestionNode()
    visual_node = ColModernVBertPipeline()
    exporter = DiskStorageExporter(output_dir=args.output_dir)

    orchestrator = DocumentOrchestrator(
        classifier=classifier,
        text_node=text_node,
        visual_node=visual_node,
        exporter=exporter
    )

    # 2. Batch Loop over each PDF
    total_start = time.time()
    corpus_summary = {
        "total_documents": len(pdf_files),
        "total_pages": 0,
        "routed_text_pages": 0,
        "routed_layout_pages": 0,
        "total_text_chunks": 0,
        "docs_processed": []
    }

    for idx, pdf_path in enumerate(pdf_files, 1):
        doc_filename = os.path.basename(pdf_path)
        print(f"\n=======================================================")
        print(f"[{idx}/{len(pdf_files)}] Ingesting Document: {doc_filename}")
        print(f"=======================================================")
        try:
            result = orchestrator.ingest_document(pdf_path)
            m = result["metrics"]
            total_p = result["total_pages"]
            txt_routed = result["routed_text_pages"]
            vis_routed = result["routed_visual_pages"]
            n_chunks = result["text_chunks"]
            doc_total_time = m["text_time"] + m["visual_time"] + m["classifier_time"]

            print(f"\n   [Done] {doc_filename}:")
            print(f"      - Text Branch   : {total_p}/{total_p} pages -> {result['text_chunks']} chunks")
            print(f"      - Visual Branch : {total_p}/{total_p} pages -> {total_p} multi-vectors")
            print(f"      - Router Tags   : {result['routed_text_pages']} text-dense | {result['routed_visual_pages']} layout-heavy")

            # Aggregate corpus totals
            corpus_summary["total_pages"] += total_p
            corpus_summary["routed_text_pages"] += txt_routed
            corpus_summary["routed_layout_pages"] += vis_routed
            corpus_summary["total_text_chunks"] += n_chunks

            # Compute dual-branch per-page metrics (all pages embedded on both branches)
            doc_entry = {
                "doc_id": result["doc_id"],
                "total_pages": total_p,
                "routed_txt_vis": f"{txt_routed}/{vis_routed}",
                "text_chunks": n_chunks,
                "text_time_sec": round(m["text_time"], 2),
                "visual_time_sec": round(m["visual_time"], 2),
                "clf_time_sec": round(m["classifier_time"], 2),
                "total_time_sec": round(doc_total_time, 2),
                "avg_text_sec_per_page": round(m["text_time"] / max(total_p, 1), 3),
                "avg_vis_sec_per_page": round(m["visual_time"] / max(total_p, 1), 3),
                "avg_clf_ms_per_page": round((m["classifier_time"] / max(total_p, 1)) * 1000, 1),
                "visual_pct": result["visual_pct"]
            }
            corpus_summary["docs_processed"].append(doc_entry)

            # Per-document metadata checkpoint
            meta_save_path = os.path.join(args.output_dir, f"{result['doc_id']}_checkpoint.json")
            with open(meta_save_path, "w") as f:
                json.dump(doc_entry, f, indent=2)

        except Exception as e:
            print(f"Error processing {pdf_path}: {e}")

    # 3. Final Corpus Report
    total_time = time.time() - total_start
    summary_path = os.path.join(args.output_dir, "corpus_export_report.json")
    with open(summary_path, "w") as f:
        json.dump(corpus_summary, f, indent=2)
    print(f"\nSaved central metrics report to: {summary_path}")

    # 4. Print Breakdown Table
    print("\n" + "=" * 110)
    print(f"{'Doc ID':<35} | {'Pages':<5} | {'Txt/Vis':<8} | {'Chunks':<6} | {'Total(s)':<8} | {'Txt(s/p)':<8} | {'Vis(s/p)':<8} | {'Clf(ms/p)':<9}")
    print("-" * 110)
    for doc in corpus_summary["docs_processed"]:
        print(f"{doc['doc_id'][:35]:<35} | {doc['total_pages']:<5} | {doc['routed_txt_vis']:<8} | {doc['text_chunks']:<6} | {doc['total_time_sec']:<8.2f} | {doc['avg_text_sec_per_page']:<8.3f} | {doc['avg_vis_sec_per_page']:<8.3f} | {doc['avg_clf_ms_per_page']:<9.1f}")
    print("=" * 110)

    # 5. Final Summary Statistics
    total_pages = max(corpus_summary["total_pages"], 1)
    tot_txt_time = sum(d["text_time_sec"] for d in corpus_summary["docs_processed"])
    tot_vis_time = sum(d["visual_time_sec"] for d in corpus_summary["docs_processed"])

    print("\n================== DUAL-BRANCH EXPORT COMPLETE ==================")
    print(f"Total Documents Exported  : {len(corpus_summary['docs_processed'])}")
    print(f"Total Pages Processed     : {corpus_summary['total_pages']}")
    print(f"Total Text Chunks Saved   : {corpus_summary['total_text_chunks']}")
    print(f"Router Breakdown          : {corpus_summary['routed_text_pages']} text-dense / {corpus_summary['routed_layout_pages']} layout-heavy")
    print(f"Mean Text Embedding Time  : {tot_txt_time / total_pages:.3f} s/page")
    print(f"Mean Visual Embedding Time: {tot_vis_time / total_pages:.3f} s/page")
    print(f"Total Ingestion Wall Time : {total_time:.2f}s ({total_time / total_pages:.2f} s/page)")
    print("=================================================================\n")

if __name__ == "__main__":
    main()