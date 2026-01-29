# rag/retriever.py

from typing import List, Dict, Any, Tuple
from api.config import settings
from rag.embedder import EmbeddingService
from rag.faiss_store import FaissStore

class Retriever:
    def __init__(self, embedder: EmbeddingService, store: FaissStore):
        self.embedder = embedder
        self.store = store
        self.threshold = settings.SIMILARITY_THRESHOLD

    def search(self, query: str, top_k:int) -> Tuple[str, List[Dict[str, Any]], List[float]]:
        query_vector = (
            self.embedder.embed_texts([query])
            .cpu()
            .numpy()
            .astype("float32")
        )

        scores, indices = self.store.index.search(query_vector, top_k)

        chunks = []
        chunk_scores = []

        for score, idx in zip(scores[0], indices[0]): 
            if idx != -1 and score >= settings.SIMILARITY_THRESHOLD:
                chunks.append(self.store.metadata[idx])
                chunk_scores.append(float(score))

        top_score = chunk_scores[0] if chunk_scores else 0.0

        print(f"[DEBUG] Retrieved {len(chunks)} chunks, top score {top_score:.4f}")
        return "ANSWER", chunks, chunk_scores
