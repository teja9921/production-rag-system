from typing import Dict, Any

from langchain_core.runnables import Runnable

from rag.hybrid_retriever import HybridRetriever
from rag.reranker import CrossEncoderReranker
from core.logger import get_logger

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

    def invoke(
        self,
        state: Dict[str, Any],
        config=None,
        **kwargs,
    ) -> Dict[str, Any]:

        query = state.get("rewritten_query") or state["query"]

        status, chunks, scores = self.retriever.search(query)

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

        self.logger.info(
            "event=RETRIEVAL_COMPLETE | initial=%d | reranked=%d",
            len(chunks),
            len(reranked_chunks),
        )

        return {
            "status": "ANSWER",
            "retrieved_chunks": reranked_chunks,
            "retrieval_scores": scores,
        }
