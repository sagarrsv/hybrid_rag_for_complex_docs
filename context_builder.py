
"""
context_builder.py
=============================================================
Builds deterministic multimodal context from retrieved pages.

Responsibilities:
1. Resolve Qdrant doc_id to an original PDF.
2. Extract complete page text using PyMuPDF.
3. Render images for layout-heavy pages.
4. Preserve retrieval order and page citations.

This module performs no retrieval, ranking, or LLM inference.
=============================================================
"""

from pathlib import Path
from typing import Any, Dict, List

import pymupdf

from config import (
    PDF_ROOT,
    PT_VISUAL,
    PAGE_IMAGE_DPI,
    GENERATOR_CONTEXT_PAGES,
)


class ContextBuilder:
    def __init__(self, pdf_root: str = PDF_ROOT):
        self.pdf_root = Path(pdf_root).expanduser().resolve()

        if not self.pdf_root.is_dir():
            raise FileNotFoundError(
                f"PDF directory does not exist: {self.pdf_root}"
            )

        # Build a mapping: doc_id -> original PDF path.
        self.pdf_index = self._index_pdfs()

        # Cache within this process to avoid duplicate extraction.
        self.text_cache = {}
        self.image_cache = {}

    def _index_pdfs(self) -> Dict[str, Path]:
        """
        Index PDFs by filename stem.

        Example:
        2407.01449_ColPali.pdf -> 2407.01449_ColPali

        The filename stem must match the Qdrant doc_id.
        """
        index = {}

        for path in self.pdf_root.rglob("*"):
            if not path.is_file() or path.suffix.lower() != ".pdf":
                continue

            doc_id = path.stem

            if doc_id in index:
                raise ValueError(
                    f"Duplicate PDF doc_id: {doc_id}"
                )

            index[doc_id] = path

        if not index:
            raise RuntimeError(
                f"No PDFs found under {self.pdf_root}"
            )

        return index

    def _resolve_pdf(self, doc_id: str) -> Path:
        """Locate the original PDF using its indexed doc_id."""
        path = self.pdf_index.get(doc_id)

        if path is None:
            raise FileNotFoundError(
                f"PDF not found for doc_id: {doc_id}"
            )

        return path

    def _extract_page(
        self,
        doc_id: str,
        page_num: int,
        include_image: bool,
    ) -> tuple[str, bytes | None]:
        """
        Access one 1-indexed PDF page.

        PyMuPDF does not need to extract preceding pages.

        Both text and image are produced from the same
        PDF page to maintain citation consistency.
        """
        text_key = (doc_id, page_num)
        image_key = (doc_id, page_num, PAGE_IMAGE_DPI)

        have_text = text_key in self.text_cache
        have_image = image_key in self.image_cache

        if have_text and (not include_image or have_image):
            return (
                self.text_cache[text_key],
                self.image_cache.get(image_key)
                if include_image else None,
            )

        pdf_path = self._resolve_pdf(doc_id)

        with pymupdf.open(pdf_path) as pdf:
            if not 1 <= page_num <= len(pdf):
                raise ValueError(
                    f"Invalid page {page_num} for {doc_id}; "
                    f"PDF contains {len(pdf)} pages."
                )

            # ingestion stores 1-indexed PDF page numbers.
            page = pdf.load_page(page_num - 1)

            if not have_text:
                # Full page extraction; no chunk overlap.
                self.text_cache[text_key] = page.get_text(
                    "text",
                    sort=True,
                )

            if include_image and not have_image:
                pix = page.get_pixmap(
                    dpi=PAGE_IMAGE_DPI,
                    colorspace=pymupdf.csRGB,
                    alpha=False,
                )

                self.image_cache[image_key] = pix.tobytes(
                    "png"
                )

        return (
            self.text_cache[text_key],
            self.image_cache.get(image_key)
            if include_image else None,
        )

    def build(
        self,
        retrieved_docs: List[Dict[str, Any]],
        max_pages: int | None = GENERATOR_CONTEXT_PAGES,
        include_images: bool = True,
    ) -> Dict[str, Any]:
        """
        Assemble retrieved pages for the generator.

        include_images=False supports the text-only
        generation baseline without rerunning retrieval.

        max_pages=None means use all supplied pages,
        which is useful for oracle-context evaluation.
        """
        pages = []
        citations = []
        seen = set()

        for hit in retrieved_docs:
            doc_id = str(hit["doc_id"])
            page_num = int(hit["page_num"])
            page_type = hit["page_type"]

            page_key = (doc_id, page_num)

            # Defensive deduplication.
            if page_key in seen:
                continue

            seen.add(page_key)

            citation_id = f"P{len(pages) + 1}"

            needs_image = (
                include_images
                and page_type == PT_VISUAL
            )

            page_text, image_bytes = self._extract_page(
                doc_id=doc_id,
                page_num=page_num,
                include_image=needs_image,
            )

            # Each page is one evidence unit regardless
            # of whether it came from text or visual retrieval.
            pages.append({
                "citation_id": citation_id,
                "doc_id": doc_id,
                "page_num": page_num,
                "page_type": page_type,
                "text": page_text,
                "image_bytes": image_bytes,
            })

            citations.append({
                "citation_id": citation_id,
                "doc_id": doc_id,
                "page_num": page_num,
                "page_type": page_type,
            })

            if (
                max_pages is not None
                and len(pages) >= max_pages
            ):
                break

        return {
            "pages": pages,
            "citations": citations,
            "page_count": len(pages),
            "image_count": sum(
                page["image_bytes"] is not None
                for page in pages
            ),
        }
