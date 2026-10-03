import os
import time
import pymupdf
from typing import Dict, Any, List

class DocumentOrchestrator:
    def __init__(self, classifier, text_node, visual_node, exporter=None):
        self.classifier = classifier
        self.text_node = text_node
        self.visual_node = visual_node
        self.exporter = exporter

    def ingest_document(self, pdf_path: str) -> Dict[str, Any]:
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"Document not found at: {pdf_path}")
        
        doc_id = os.path.splitext(os.path.basename(pdf_path))[0]
        doc = pymupdf.open(pdf_path)
        print(f"\n=======================================================")
        print(f"Generating Embeddings for: {doc_id} ({len(doc)} pages)")
        print(f"=======================================================")

        text_records: List[Dict[str, Any]] = []
        visual_records: List[Dict[str, Any]] = []
        timings = {
            "text_pages": 0,
            "visual_pages": 0,
            "text_time": 0.0,
            "visual_time": 0.0,
            "classifier_time": 0.0
        }

        routed_text_count = 0
        routed_vis_count = 0

        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            p_display = page_num + 1

            # 1. Run classifier to detect page_type
            tc0 = time.time()
            page_type, meta = self.classifier.classify_page_optimized(page)
            timings["classifier_time"] += (time.time() - tc0)

            if page_type == "text-dense":
                routed_text_count += 1
            else:
                routed_vis_count += 1

            # 2. Text branch (runs on all pages for ablation rows A & B)
            t_text_0 = time.time()
            page_text = page.get_text("text")
            t_recs = self.text_node.process_page(
                text=page_text,
                doc_id=doc_id,
                page_num=p_display,
                page_type=page_type,
                metadata=meta
            )
            elapsed_txt = (time.time() - t_text_0)
            timings["text_time"] += (time.time() - t_text_0)
            timings["text_pages"] += 1
            text_records.extend(t_recs)

            # 3. Visual branch (runs on all pages for ablation rows C & D)
            t_vis_0 = time.time()
            v_rec = self.visual_node.embed_page(
                page=page,
                doc_id=doc_id,
                page_num=p_display,
                page_type=page_type
            )
            elapsed_vis = (time.time() - t_vis_0)
            timings["visual_time"] += (time.time() - t_vis_0)
            timings["visual_pages"] += 1
            visual_records.append(v_rec)

            print(f"   [P.{p_display}/{len(doc)}] Router: {page_type:<12} | "
                  f"Text: {len(t_recs)} chunks | Visual: {v_rec['n_vectors']} tokens"
                  f"text_latency : {elapsed_txt:.2f}s | Vis_latency : {elapsed_vis:.2f}s")

        # 4. Save vectors to Kaggle output directory
        if self.exporter:
            print(f"   -> Flushing {doc_id} embeddings to disk (FP16)...")
            self.exporter.export_text_records(doc_id, text_records)
            self.exporter.export_visual_records(doc_id, visual_records)

        total_p = max(len(doc), 1)
        return {
            "doc_id": doc_id,
            "total_pages": len(doc),
            "text_chunks": len(text_records),
            "routed_text_pages": routed_text_count,
            "routed_visual_pages": routed_vis_count,
            "metrics": timings,
            "visual_pct": round((routed_vis_count / total_p) * 100, 2)
        }