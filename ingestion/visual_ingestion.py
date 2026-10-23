import torch
import pymupdf
import gc
from PIL import Image
from typing import Dict, Any, List, Tuple
from colpali_engine.models import ColModernVBert, ColModernVBertProcessor

class ColModernVBertPipeline:
    def __init__(
        self,
        model_name: str = "ModernVBERT/colmodernvbert-merged",
        dpi: int = 150
    ):
        # Fallback to CPU if MPS throws errors on the Mac during ingestion
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

        self.model_name = model_name
        self.dpi = dpi

        self.model = ColModernVBert.from_pretrained(
            self.model_name,
            dtype=torch.bfloat16,  # use torch_dtype=torch.bfloat16 for flash attention
            trust_remote_code=True,
        ).to(self.device).eval()

        self.processor = ColModernVBertProcessor.from_pretrained(self.model_name)
        print("Model loaded successfully.")

    def embed_page(self, page: pymupdf.Page, doc_id: str, page_num: int, page_type: str = "layout-heavy") -> Dict[str, Any]:
        """
        Renders page to image and extracts ColModernVBERT multi-vector embeddings.
        """
        pix = page.get_pixmap(dpi=self.dpi)  # sets the render quality...100= 850 x 1100px long edge~1100
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)

        with torch.no_grad():
            # Use process_images instead of the standard __call__ to ensure dummy text tokens are generated
            inputs = self.processor.process_images([img]).to(self.device)
            emb = self.model(**inputs) ##[Batch, SQ len, embdd dim]-->[1,1149,128]
            
            # Convert bfloat16 -> float16 numpy array
            page_emb_fp16 = emb[0].to(torch.float16).cpu().numpy()

            del inputs, emb
            gc.collect()
            if self.device.type == "cuda":
                torch.cuda.empty_cache()

        return {
            "chunk_id": f"{doc_id}_p{page_num}",
            "doc_id": doc_id,
            "page_num": page_num,
            "page_type": page_type,
            "multivector": page_emb_fp16,  # Shape: torch.Size([1149, 128])
            "n_vectors": page_emb_fp16.shape[0]
        }

    def compute_maxsim(self, queries: List[str], page_embeddings: List[torch.Tensor]) -> List[List[Tuple[float, int]]]:
        """
        Computes MaxSim late-interaction scores across embedded pages.
        """
        with torch.no_grad():
            query_inputs = self.processor(text=queries, return_tensors="pt", padding=True).to(self.device)
            query_embs = self.model(**query_inputs).cpu()

            all_results = []
            for q_idx, _ in enumerate(queries):
                q_emb = query_embs[q_idx]
                scores = []
                for p_idx, p_emb in enumerate(page_embeddings):
                    # Compute MaxSim
                    sim = torch.einsum("qd,pd->qp", q_emb, p_emb)
                    score = sim.max(dim=1).values.sum().item()
                    scores.append((score, p_idx + 1))
                all_results.append(sorted(scores, reverse=True))
            return all_results