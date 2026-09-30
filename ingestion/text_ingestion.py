import os
from typing import List, Dict, Any
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings

class TextDenseIngestionNode:
    def __init__(
        self,
        api_key: str = "AIzaSyDummyKey_ReplaceWithYourActualKey12345",
        model_name: str = "models/text-embedding-004",
        chunk_size: int = 800,
        chunk_overlap: int = 100
    ):
        """
        Text ingestion node using Google Generative AI Embeddings.
        Uses task_type='RETRIEVAL_DOCUMENT' for optimal index representation.
        """
        # Set or fallback to environment variable if available
        self.api_key = os.getenv("GOOGLE_API_KEY", api_key)
        
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
        
        self.embedder = GoogleGenerativeAIEmbeddings(
            model=model_name,
            google_api_key=self.api_key,
            task_type="RETRIEVAL_DOCUMENT"
        )

    def process_page(
        self,
        text: str,
        doc_id: str,
        page_num: int,
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

        # Single batch API call for all chunks on this page
        embeddings = self.embedder.embed_documents(chunks)

        records = []
        for idx, (chunk, emb) in enumerate(zip(chunks, embeddings)):
            record = {
                "chunk_id": f"{doc_id}_p{page_num}_c{idx}",
                "doc_id": doc_id,
                "page_num": page_num,
                "modality": "text-dense",
                "text": chunk,
                "vector": emb,  # 1D vector (dim: 768)
            }
            if metadata:
                record["metadata"] = metadata
            records.append(record)

        return records