"""
generator.py
====================================================================
Prompt construction and text generation. Enriches visual page hits
with page text chunks from Qdrant before synthesis.
====================================================================
"""

from typing import List, Dict, Any
from config import GENERATOR_CONTEXT_PAGES


class MultimodalGenerator:
    def __init__(self, retriever=None):
        self.retriever = retriever

    def format_context(self, docs: List[Dict[str, Any]]) -> str:
        """Structures retrieved context chunks into attributable prompt blocks."""
        blocks = []
        for d in docs[:GENERATOR_CONTEXT_PAGES]:
            content = d.get("content", "")
            # If visual hit lacks text, fetch its text chunks from the text collection
            if not content and self.retriever:
                content = self.retriever.fetch_page_text(d["doc_id"], d["page_num"])

            header = f"[{d['source']} | Doc: {d['doc_id']} | Page: {d.get('page_num')} | Type: {d.get('page_type')}]"
            blocks.append(f"{header}\n{content or '[No text content extracted for page]'}")
        return "\n\n".join(blocks)

    def generate(self, query: str, docs: List[Dict[str, Any]]) -> str:
        """Synthesizes an answer using the formatted context blocks."""
        if not docs:
            return "No matching context found."

        context_str = self.format_context(docs)
        top_doc = docs[0]
        score_val = top_doc.get("rrf_score", top_doc.get("score", 0.0))

        simulated_answer = (
            f"Synthesized response using {min(len(docs), GENERATOR_CONTEXT_PAGES)} context pages.\n"
            f"Top Source: {top_doc['doc_id']} (Page {top_doc.get('page_num')}, {top_doc.get('page_type')}) "
            f"via {top_doc['source']} [Score: {score_val:.4f}].\n"
            f"Context Summary Length: {len(context_str)} characters."
        )
        return simulated_answer