import time
from typing import Dict, Any

from langchain_core.runnables import Runnable

from rag.hybrid_retriever import HybridRetriever
from rag.reranker import CrossEncoderReranker
from core.logger import get_logger
from core.tracing import traced
from core.metrics import RETRIEVAL_LATENCY, RETRIEVAL_TOP1_SIMILARITY
from api.config import settings

class RetrieverRunnable(Runnable):
    """
    LangGraph adapter for hybrid retrieval + reranking.
    """

    def __init__(
        self,
        retriever: HybridRetriever,
        reranker: CrossEncoderReranker,
    ):
        self.retriever = retriever
        self.reranker = reranker
        self.logger = get_logger("orchestration.retriever")

    @traced("retrieval")
    def invoke(
        self,
        state: Dict[str, Any],
        config=None,
        **kwargs,
    ) -> Dict[str, Any]:
        
        start = time.perf_counter()
        try:
            query = state.get("rewritten_query") or state["query"]

            status, chunks, scores = self.retriever.search(query)
            if scores:
                if isinstance(scores, dict):
                    top1 = max(scores.values())
                else:
                    top1 = scores[0]
                RETRIEVAL_TOP1_SIMILARITY.observe(float(top1))

            if status == "NO_ANSWER":
                self.logger.info(
                    "event=RETRIEVAL_NO_ANSWER | query_len=%d",
                    len(query),
                )
                return {
                    "status": "NO_ANSWER",
                    "retrieved_chunks": [],
                    "retrieval_scores": {},
                }

            reranked_chunks = self.reranker.rerank(
                query=query,
                chunks=chunks,
                top_k=5,
            )

            if not reranked_chunks:
                self.logger.info("event=RETRIEVAL_NO_ANSWER | reason=empty_rerank")
                return {
                    "status": "NO_ANSWER",
                    "retrieved_chunks": [],
                    "retrieval_scores": scores,
                }

            top1_rerank = float(reranked_chunks[0].get("rerank_score", -1e9))
            if top1_rerank < settings.RERANK_SCORE_THRESHOLD:
                self.logger.info(
                    "event=RETRIEVAL_NO_ANSWER | reason=top1_below_rerank_threshold | top1=%.4f | threshold=%.4f",
                    top1_rerank,
                    settings.RERANK_SCORE_THRESHOLD,
                )
                return {
                    "status": "NO_ANSWER",
                    "retrieved_chunks": [],
                    "retrieval_scores": scores,
                }

            gated_chunks = [
                c
                for c in reranked_chunks
                if float(c.get("rerank_score", -1e9)) >= settings.RERANK_SCORE_THRESHOLD
            ]

            self.logger.info(
                "event=RETRIEVAL_COMPLETE | initial=%d | reranked=%d | gated=%d | rerank_threshold=%.4f",
                len(chunks),
                len(reranked_chunks),
                len(gated_chunks),
                settings.RERANK_SCORE_THRESHOLD,
            )

            return {
                "status": "ANSWER",
                "retrieved_chunks": gated_chunks,
                "retrieval_scores": scores,
            }
        
        finally:
            latency_s = time.perf_counter() - start
            RETRIEVAL_LATENCY.observe(latency_s)
            latency_ms = int(latency_s * 1000)
            self.logger.info("event=RETRIEVAL_LATENCY | latency_ms=%d", latency_ms)
