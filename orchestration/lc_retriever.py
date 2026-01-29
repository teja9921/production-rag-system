# orchestration/lc_retriever.py
from api.config import settings
from langchain_core.runnables import Runnable
from rag.hybrid_retriever import HybridRetriever
from rag.reranker import CrossEncoderReranker
from typing import Dict, Any

class RetrieverRunnable(Runnable):
    def __init__(self, retriever: HybridRetriever):
        self.retriever = retriever
        self.reranker = CrossEncoderReranker()

    def invoke(self, state: Dict[str, Any], config=None, **kwargs) -> Dict[str, Any]:
        query = state["rewritten_query"] if state["rewritten_query"] else state["query"]
        
        chunks = self.retriever.search(query)

        if not chunks:
            return {
                "status": "NO_ANSWER",
                "retrieved_chunks": [],
            }
        reranked = self.reranker.rerank(query, chunks, top_k=5)

        return {
            "status": "ANSWER",
            "retrieved_chunks": reranked,
        }
