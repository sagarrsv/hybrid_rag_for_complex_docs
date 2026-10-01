import os
import time
import pymupdf
from typing import Dict, Any, List

class DocumentOrchestrator:
    def __init__(self, classifier, text_node, visual_node, storage_node = None):
        self.classifier = classifier
        self.text_node = text_node
        self.visual_node = visual_node
        self.storage_node = storage_node

    def ingest_document(self, pdf_path: str) -> Dict[str, Any]:

        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"Document not found at: {pdf_path}")
        
        doc_id = os.path.splitext(os.path.basename(pdf_path))[0]
        doc = pymupdf.open(pdf_path)

        print(f"\n=======================================================")
        print(f"Starting Routed Ingestion: {doc_id} ({len(doc)} pages)")
        print(f"File Path: {pdf_path}")
        print(f"=======================================================")

        text_records: List[Dict[str, Any]] = []
        visual_records: List[Dict[str, Any]] = []
        timings = {"text_pages": 0, "visual_pages": 0, "text_time": 0.0, "visual_time": 0.0}

        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            modality, meta = self.classifier.classify_page_optimized(page)
            p_display = page_num + 1
            
            print(f"\n[Page {p_display}/{len(doc)}] Decision: --> {modality.upper()} <--")
            print(f"   L_ Heuristics: chars={meta.get('char_count')}, "
                  f"cols={meta.get('columns', 1)}, tables={meta.get('has_table', False)}, "
                  f"large_img={meta.get('has_large_image', False)}")

            if modality == "text-dense":
                t0 = time.time()
                page_text = page.get_text("text")
                records = self.text_node.process_page(
                    text=page_text, 
                    doc_id=doc_id, 
                    page_num=p_display,
                    metadata=meta
                )
                elapsed = time.time() - t0
                text_records.extend(records)
                timings["text_time"] += elapsed
                timings["text_pages"] += 1

                # Inspect Text Bi-Encoder vector dimensionality
                vec_dim = len(records[0]["vector"]) if records else 0
                print(f"   ✓ [Text Ingestion]   Latency: {elapsed:.2f}s | "
                      f"Chunks Created: {len(records)} | Vector Dim: [{len(records)}, {vec_dim}] (1D dense)")
            else:
                t0 = time.time()
                records = self.visual_node.embed_page(
                    page=page, 
                    doc_id=doc_id, 
                    page_num=p_display
                )
                visual_records.append(records)

                elapsed = (time.time() - t0)
                timings["visual_time"] += elapsed
                timings["visual_pages"] += 1

                # Inspect ColModernVBERT multi-vector tensor shape
                emb_shape = tuple(records["multivector"].shape)
                print(f"   ✓ [Visual Ingestion] Latency: {elapsed:.2f}s | "
                      f"Output Tensor Shape: {emb_shape} (ColModernVBERT patch multivectors)")
        
        # Upsert directly to Qdrant if a storage node is configured
        if self.storage_node:
            if text_records:
                print(f"   -> Upserting {len(text_records)} text chunks to Qdrant...")
                self.storage_node.upsert_text_records(text_records)
            if visual_records:
                print(f"   -> Upserting {len(visual_records)} visual pages to Qdrant...")
                self.storage_node.upsert_visual_records(visual_records)
                
        return {
            "doc_id": doc_id,
            "total_pages": len(doc),
            "text_records": text_records,
            "visual_records": visual_records,
            "metrics": timings
            
        }