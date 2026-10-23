import os
from typing import List, Dict, Any
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import torch

class TextDenseIngestionNode:
    def __init__(
        self,
        # api_key: str = "AIzaSyDummyKey_ReplaceWithYourActualKey12345",
        model_name: str = "BAAI/bge-m3",
        chunk_size: int = 1000,
        chunk_overlap: int = 150,
        device: str = None
    ):
        """
        Text ingestion node using BGE M3 embeddings.
        Uses task_type='RETRIEVAL_DOCUMENT' for optimal index representation.
        """
        # Set or fallback to environment variable if available
        # self.api_key = os.getenv("GOOGLE_API_KEY", api_key)

        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        print(f"Loading {model_name} on device: {self.device}...")
        
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
        # Loads BAAI/bge-m3 directly into GPU memory
        self.model = SentenceTransformer(model_name, device=self.device)


        # self.embedder = GoogleGenerativeAIEmbeddings(
        #     model=model_name,
        #     google_api_key=self.api_key,
        #     task_type="RETRIEVAL_DOCUMENT"
        # )

    def process_page(
        self,
        text: str,
        doc_id: str,
        page_num: int,
        page_type: str = "text-dense",
        metadata: Dict[str, Any] = None
    ) -> List[Dict[str, Any]]:
        """
        Chunks page text, queries the embedding API in batch, and formats payloads for Qdrant.
        """
        cleaned_text = text.strip()
        if not cleaned_text:
            return []

        chunks = self.splitter.split_text(cleaned_text)
        if not chunks:
            return []

        # Forward pass on GPU: returns 1024-dimensional normalized vectors
        embeddings = self.model.encode(
            chunks,
            batch_size=16,
            normalize_embeddings=True,
            show_progress_bar=False
        ).tolist()

        # # Single batch API call for all chunks on this page
        # embeddings = self.embedder.embed_documents(chunks)

        records = []
        for idx, (chunk, emb) in enumerate(zip(chunks, embeddings)):
            record = {
                "chunk_id": f"{doc_id}_p{page_num}_c{idx}",
                "doc_id": doc_id,
                "page_num": page_num,
                "page_type": page_type, # Saved as 'text-dense' or 'layout-heavy'
                "modality": "text-dense",
                "text": chunk,
                "vector": emb,  # 1D vector (dim: 768)
            }
            if metadata:
                record["metadata"] = metadata
            records.append(record)

        return records