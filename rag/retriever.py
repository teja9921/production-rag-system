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
        self.top_k = settings.RETRIEVAL_K

    def search(self, query: str) -> Tuple[str, List[Dict[str, Any]], List[float]]:
        query_vector = (
            self.embedder.embed_texts([query])
            .cpu()
            .numpy()
            .astype("float32")
        )

        scores, indices = self.store.index.search(query_vector, self.top_k)

        top_score = float(scores[0][0])
        if top_score < self.threshold:
            print(f"[DEBUG] Top score {top_score:.4f} below threshold {self.threshold}")
            return "NO_ANSWER", [], []

        chunks = []
        chunk_scores = []

        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue
            chunks.append(self.store.metadata[idx])
            chunk_scores.append(float(score))

        print(f"[DEBUG] Retrieved {len(chunks)} chunks, top score {top_score:.4f}")
        return "ANSWER", chunks, chunk_scores
