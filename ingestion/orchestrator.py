import os
import time
import pymupdf
from typing import Dict, Any, List

class DocumentOrchestrator:
    def __init__(self, classifier, text_node, visual_node):
        self.classifier = classifier
        self.text_node = text_node
        self.visual_node = visual_node

    def ingest_document(self, pdf_path: str) -> Dict[str, Any]:
        doc_id = os.path.splitext(os.path.basename(pdf_path))[0]
        doc = pymupdf.open(pdf_path)

        text_records: List[Dict[str, Any]] = []
        visual_records: List[Dict[str, Any]] = []
        timings = {"text_pages": 0, "visual_pages": 0, "text_time": 0.0, "visual_time": 0.0}

        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            modality, meta = self.classifier.classify_page_optimized(page)
            p_display = page_num + 1

            if modality == "text-dense":
                t0 = time.time()
                page_text = page.get_text("text")
                records = self.text_node.process_page(
                    text=page_text, 
                    doc_id=doc_id, 
                    page_num=p_display,
                    metadata=meta
                )
                text_records.extend(records)
                timings["text_time"] += (time.time() - t0)
                timings["text_pages"] += 1
            else:
                t0 = time.time()
                rec = self.visual_node.embed_page(
                    page=page, 
                    doc_id=doc_id, 
                    page_num=p_display
                )
                visual_records.append(rec)
                timings["visual_time"] += (time.time() - t0)
                timings["visual_pages"] += 1

        return {
            "doc_id": doc_id,
            "total_pages": len(doc),
            "text_records": text_records,
            "visual_records": visual_records,
            "metrics": timings
        }