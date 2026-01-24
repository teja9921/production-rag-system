# orchestration/lc_retriever.py

from langchain_core.runnables import Runnable
from rag.retriever import Retriever
from typing import Dict, Any

class RetrieverRunnable(Runnable):
    def __init__(self, retriever: Retriever):
        self.retriever = retriever

    def invoke(self, state: Dict[str, Any], config=None, **kwargs) -> Dict[str, Any]:
        query = state["rewritten_query"] if state["rewritten_query"] else state["query"]
        status, chunks, scores = self.retriever.search(query)
        return {
            "status": status,
            "retrieved_chunks": chunks,
            "scores": scores,
        }
