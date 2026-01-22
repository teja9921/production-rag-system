# orchestration/lc_retriever.py

from langchain_core.runnables import Runnable
from typing import Dict, Any

class RetrieverRunnable(Runnable):
    def __init__(self, retriever):
        self.retriever = retriever

    def invoke(self, input: Dict[str, Any], config=None, **kwargs) -> Dict[str, Any]:
        query = input["query"]
        status, chunks, scores = self.retriever.search(query)
        return {
            "query": query,
            "status": status,
            "chunks": chunks,
            "scores": scores,
        }
