import os
import sys
import argparse
from ingestion.doc_classifier import PageClassifier
from ingestion.text_ingestion import TextDenseIngestionNode
from ingestion.visual_ingestion import ColModernVBertPipeline
from ingestion.orchestrator import DocumentOrchestrator

def parse_args():
    parser = argparse.ArgumentParser(description="Stage 1: Dynamic Routed Document Ingestion")
    parser.add_argument(
        "--pdf_path",
        type=str,
        default="/kaggle/input/datasets/sagarrsv/test-paper-1/colmodernvbert_paper.pdf",
        help="Full path to the PDF document to ingest"
    )
    return parser.parse_args()

def main():
    args = parse_args()
    pdf_path = args.pdf_path

    # 1. Initialize components
    print("Loading models and initializing pipelines...")
    classifier = PageClassifier()
    text_node = TextDenseIngestionNode()
    visual_node = ColModernVBertPipeline()

    # 2. Wire orchestrator
    orchestrator = DocumentOrchestrator(classifier, text_node, visual_node)

    # 3. Process test file
    sample_pdf = "/kaggle/input/datasets/sagarrsv/test-paper-2/colmodernvbert_paper.pdf"
    result = orchestrator.ingest_document(pdf_path)

    # 4. Inspect results
    print("\n================== INGESTION RUN SUMMARY ==================")
    print(f"Ingestion complete for {result['doc_id']}:")
    print(f"Total Pages Processed : {result['total_pages']}")
    print(f"- Text Chunks Created: {len(result['text_records'])} (from {result['metrics']['text_pages']} pages)")
    print(f"- Visual Pages Embedded: {len(result['visual_records'])} (from {result['metrics']['visual_pages']} pages)")
    print(f"- Avg Text Latency: {result['metrics']['text_time'] / max(result['metrics']['text_pages'], 1):.2f}s/page")
    print(f"- Avg Visual Latency: {result['metrics']['visual_time'] / max(result['metrics']['visual_pages'], 1):.2f}s/page")

if __name__ == "__main__":
    main()