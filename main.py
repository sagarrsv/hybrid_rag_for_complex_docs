import os
from ingestion.doc_classifier import PageClassifier
from ingestion.text_ingestion import TextDenseIngestionNode
from ingestion.visual_ingestion import ColModernVBertPipeline
from ingestion.orchestrator import DocumentOrchestrator

def main():
    # 1. Initialize components
    classifier = PageClassifier()
    text_node = TextDenseIngestionNode(api_key=os.getenv("GOOGLE_API_KEY", "your_dummy_key"))
    visual_node = ColModernVBertPipeline()

    # 2. Wire orchestrator
    orchestrator = DocumentOrchestrator(classifier, text_node, visual_node)

    # 3. Process test file
    sample_pdf = "/path/to/Apple_financial_filing.pdf"
    result = orchestrator.ingest_document(sample_pdf)

    # 4. Inspect results
    print(f"Ingestion complete for {result['doc_id']}:")
    print(f"- Text Chunks Created: {len(result['text_records'])} (from {result['metrics']['text_pages']} pages)")
    print(f"- Visual Pages Embedded: {len(result['visual_records'])} (from {result['metrics']['visual_pages']} pages)")
    print(f"- Avg Text Latency: {result['metrics']['text_time'] / max(result['metrics']['text_pages'], 1):.2f}s/page")
    print(f"- Avg Visual Latency: {result['metrics']['visual_time'] / max(result['metrics']['visual_pages'], 1):.2f}s/page")

if __name__ == "__main__":
    main()