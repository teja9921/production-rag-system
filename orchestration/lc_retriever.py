# orchestration/lc_retriever.py

from langchain_core.runnables import Runnable
from rag.hybrid_retriever import HybridRetriever
from typing import Dict, Any

class RetrieverRunnable(Runnable):
    def __init__(self, retriever: HybridRetriever):
        self.retriever = retriever

    def invoke(self, state: Dict[str, Any], config=None, **kwargs) -> Dict[str, Any]:
        query = state["rewritten_query"] if state["rewritten_query"] else state["query"]
        
        chunks = self.retriever.search(query)

        if not chunks:
            return {
                "status": "NO_ANSWER",
                "retrieved_chunks": [],
                "scores": []
            }

        return {
            "status": "ANSWER",
            "retrieved_chunks": chunks,
            "scores": [],
        }
