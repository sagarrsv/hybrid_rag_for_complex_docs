import pymupdf
import re
from typing import Dict, Any, Tuple

# Precompiled regex: matches "Table 1:", "Table 2.", "Table I", etc.
TABLE_CAPTION_REGEX = re.compile(r"^\s*Table\s+(\d+|[IVXLCDM]+)[\.:\s]", re.IGNORECASE | re.MULTILINE)

class PageClassifier:
    def __init__(
        self,
        min_text_len: int = 100,
        min_sparse_drawings: int = 5,
        drawing_path_threshold: int = 15,
    ):
        self.min_text_len = min_text_len
        self.min_sparse_drawings = min_sparse_drawings
        self.drawing_path_threshold = drawing_path_threshold

    def classify_page_optimized(
        self,
        page: pymupdf.Page,
    ) -> Tuple[str, Dict[str, Any]]:
        """
        Layout classifier that avoids raster image extraction/decompression during routing.
        Evaluates PDF structural primitives to route between 'text-dense' and 'layout-heavy'.
        """
        # 1. Fast text extraction
        text_content = page.get_text("text").strip()
        char_count = len(text_content)

        # 2. Defensive raster image metadata inspection (avoid decompression)
        image_list = page.get_images(full=False)
        has_large_image = False
        for img_info in image_list:
            if len(img_info) >= 4:
                width, height = img_info[2], img_info[3]
                if width > 120 and height > 120:
                    has_large_image = True
                    break

        # 3. Vector path count
        drawings = page.get_drawings()
        vector_path_count = len(drawings)

        # --- Early Exit: Scanned page, diagrammatic slide, or graphic cover ---
        if char_count < self.min_text_len and (has_large_image or vector_path_count >= self.min_sparse_drawings):
            return "layout-heavy", {
                "modality": "layout-heavy",
                "reason": "scanned_or_sparse_visual",
                "char_count": char_count,
                "has_large_image": has_large_image,
                "vector_path_count": vector_path_count,
            }

        # 4. Lexical distribution (digits & financial tokens)
        numeric_symbols = sum(c.isdigit() or c in "$%€," for c in text_content)
        numeric_ratio = numeric_symbols / max(char_count, 1)

        # 5. Caption detection (The cleanest academic table detector)
        has_table_caption = bool(TABLE_CAPTION_REGEX.search(text_content))

        # 6. Block Geometry
        blocks = page.get_text("blocks")
        text_blocks = [b for b in blocks if b[6] == 0]
        left_margins = {round(b[0], -1) for b in text_blocks}
        
        total_words = len(text_content.split())
        avg_words_per_block = total_words / max(len(text_blocks), 1)

        # 7. Native table detection (C-level)
        has_table = False
        if len(left_margins) >= 2 or vector_path_count >= 2:
            table_finder = page.find_tables()
            has_table = len(table_finder.tables) > 0

        # --- Structural Layout Triggers ---
        # Trigger A: Explicit tables or complex vector graphics (flowcharts/plots)
        is_vector_dense = vector_path_count >= self.drawing_path_threshold
        
        # Trigger B: Financial borderless statements (high numeric density across columns)
        is_financial_table = (numeric_ratio > 0.18 and len(left_margins) >= 3)

        # Trigger C: Academic open/borderless tables (booktabs rules + Table Caption)
        is_academic_table = (has_table_caption and vector_path_count >= 2)

        # Trigger D: Diagram fragmentation
        is_fragmented = (len(text_blocks) > 16 and avg_words_per_block < 8)

        is_layout_heavy = (
            has_table or
            has_large_image or
            is_vector_dense or
            is_financial_table or
            is_academic_table or
            is_fragmented
        )

        modality = "layout-heavy" if is_layout_heavy else "text-dense"

        return modality, {
            "modality": modality,
            "char_count": char_count,
            "numeric_ratio": round(numeric_ratio, 3),
            "columns": len(left_margins),
            "vector_path_count": vector_path_count,
            "has_table_caption": has_table_caption,
            "has_large_image": has_large_image,
            "has_table": has_table,
        }