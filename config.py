"""
config.py
====================================================================
Centralized configurations, hardware strategies, evaluation constants,
and canonical page-type labels.
====================================================================
"""

import os
from typing import Dict, Any, Set
import torch

# ------------------------------------------------------------------
# 1. Canonical Page-Type Labels (Matches Classifier Output)
# ------------------------------------------------------------------
PT_TEXT = "text-dense"
PT_VISUAL = "layout-heavy"

# ------------------------------------------------------------------
# 2. Qdrant Vector DB Connectivity
# ------------------------------------------------------------------
QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
QDRANT_TIMEOUT = 120

TEXT_COLLECTION = "text_chunks"
VISUAL_COLLECTION = "visual_pages"

# ------------------------------------------------------------------
# 3. Model Checkpoints & Execution Strategy
# ------------------------------------------------------------------
TEXT_MODEL_NAME = "BAAI/bge-m3"
VIS_MODEL_NAME = "ModernVBERT/colmodernvbert-merged"

# Device detection (MPS vs CUDA vs CPU)
if torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
    VIS_QUERY_DTYPE = torch.float32  # MPS requires float32 for ColModernVBERT query processing
elif torch.cuda.is_available():
    DEVICE = torch.device("cuda")
    VIS_QUERY_DTYPE = torch.bfloat16
else:
    DEVICE = torch.device("cpu")
    VIS_QUERY_DTYPE = torch.float32

# Generator Configuration
GENERATOR_MODEL_NAME = "qwen2.5:1.5b"
GENERATOR_TEMPERATURE = 0.0
GENERATOR_MAX_TOKENS = 512
GENERATOR_CONTEXT_PAGES = 3

# ------------------------------------------------------------------
# 4. Retrieval & Evaluation Hyperparameters
# ------------------------------------------------------------------
MAX_K = 10
K_VALUES = [1, 3, 5, 10]
CANDIDATE_POOL = 20
TEXT_OVERFETCH = 6
RRF_K = 60

RETRIEVAL_MODES = ["text", "visual", "hybrid_routed", "fusion_unrouted"]

GOLDEN_PATH = "evaluation/golden_dataset.json"
RESULTS_DIR = "evaluation/results"

# Documents excluded from benchmark calculations (e.g., corrupted inputs or suspect runs)
EXCLUDE_DOCS: Set[str] = {
    "2007.07399_Bringing_the_People_Back_In_Contesting_Bench",
    "2310.12469_Entropy_and_de_Haasvan_Alphen_oscillations_o",
    "2405.01168_Remote_Nucleation_and_Stationary_Domain_Walls",
    "2108.06945_Characterization_of_CSymmetric_Toeplitz_oper",
    "2312.00758_On_absolutely_friendly_measures_on_mathbbQ"
}


def snapshot() -> Dict[str, Any]:
    """Captures runtime configuration snapshot for artifact reproducibility."""
    return {
        "pt_text": PT_TEXT,
        "pt_visual": PT_VISUAL,
        "qdrant_url": QDRANT_URL,
        "qdrant_timeout": QDRANT_TIMEOUT,
        "text_collection": TEXT_COLLECTION,
        "visual_collection": VISUAL_COLLECTION,
        "text_model_name": TEXT_MODEL_NAME,
        "vis_model_name": VIS_MODEL_NAME,
        "device": str(DEVICE),
        "vis_query_dtype": str(VIS_QUERY_DTYPE),
        "generator_model_name": GENERATOR_MODEL_NAME,
        "generator_temperature": GENERATOR_TEMPERATURE,
        "generator_max_tokens": GENERATOR_MAX_TOKENS,
        "generator_context_pages": GENERATOR_CONTEXT_PAGES,
        "max_k": MAX_K,
        "k_values": K_VALUES,
        "candidate_pool": CANDIDATE_POOL,
        "text_overfetch": TEXT_OVERFETCH,
        "rrf_k": RRF_K,
        "retrieval_modes": RETRIEVAL_MODES,
        "exclude_docs": list(EXCLUDE_DOCS),
    }