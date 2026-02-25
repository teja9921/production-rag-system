from typing import List, Dict, Any, Tuple
import torch
from sentence_transformers import CrossEncoder

from core.logger import get_logger

class CrossEncoderReranker:
    """
    Cross-encoder based reranker.

    Responsibility:
    - Reorder retrieved chunks by semantic relevance
    - Trim to top_k
    - Preserve scores for observability
    """

    def __init__(
        self,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
    ):
        device = "cuda" if torch.cuda.is_available() else "cpu"

        self.model = CrossEncoder(model_name, device=device)
        self.device = device

        self.logger = get_logger("rag.reranker")

        self.logger.info(
            "event=RERANKER_INIT | model=%s | device=%s",
            model_name,
            device,
        )

    def rerank(
        self,
        query: str,
        chunks: List[Dict[str, Any]],
        top_k: int = 5,
    ) -> List[Dict[str, Any]]:

        if not chunks:
            self.logger.info("event=RERANK_SKIPPED | reason=empty_input")
            return []

        pairs = [(query, c["content"]) for c in chunks]

        scores = self.model.predict(pairs)

        ranked = sorted(
            zip(chunks, scores),
            key=lambda x: x[1],
            reverse=True,
        )

        top_chunks = []
        for chunk, score in ranked[:top_k]:
            chunk = dict(chunk)  # shallow copy
            chunk["rerank_score"] = float(score)
            top_chunks.append(chunk)

        self.logger.info(
            "event=RERANK_COMPLETE | candidates=%d | returned=%d",
            len(chunks),
            len(top_chunks),
        )

        return top_chunks
