from typing import List, Any, Dict
from rag.retriever import Retriever
from rag.bm25_store import BM25Store

class HybridRetriever:
    """
    Combines dense vector retrieval and sparse BM25 retrieval.
    """
    
    def __init__(self, dense: Retriever, sparse: BM25Store, k_dense:int =5, k_sparse:int =5):
        self.dense = dense
        self.sparse = sparse
        self.k_dense = k_dense
        self.k_sparse = k_sparse
    
    def search(self, query: str)-> List[Dict[str, Any]]:
        dense_chunks,_ = self._dense_search(query)
        sparse_chunks = self._sparse_search(query)

        combined = {c["chunk_id"]: c for c in dense_chunks}

        for c in sparse_chunks:
            combined[c["chunk_id"]] = c

        return list(combined.values())
    
    def _dense_search(self, query:str):
        status, chunks, scores = self.dense.search(query, self.k_sparse)
        return chunks, scores
    
    def _sparse_search(self, query:str):
        results = self.sparse.search(query, self.k_sparse)
        return [doc for doc, _ in results]
    

