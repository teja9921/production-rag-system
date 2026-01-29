from sentence_transformers import CrossEncoder
from typing import List, Dict, Any

class CrossEncoderReranker:
    """
    Cross-encoder based reranker
    Produces true semantic relevance ordering
    """

    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        self.model = CrossEncoder(model_name, device= "cpu")

    def rerank(self, query: str, chunks: List[Dict[str, Any]], top_k: int =5)-> List[Dict[str, Any]]:

        if not chunks:
            return []
        
        pairs = [(query, c["content"]) for c in chunks]
        scores = self.model.predict(pairs)

        ranked = sorted(
            zip(chunks, scores), 
            key = lambda x: x[1],
            reverse= True
        )

        return [c for c, _ in ranked[:top_k]]
    