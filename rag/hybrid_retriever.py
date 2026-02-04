from typing import List, Dict, Any, Tuple

from rag.retriever import Retriever
from rag.bm25_store import BM25Store
from core.logger import get_logger

class HybridRetriever:
    """
    Combines dense vector retrieval and sparse BM25 retrieval.

    Return Contract:
    - status: "ANSWER" | "NO_ANSWER"
    - chunks: List[chunk dicts]
    - scores: Dict[chunk_id -> float] (dense + sparse)
    """

    def __init__(
        self,
        dense: Retriever,
        sparse: BM25Store,
        k_dense: int = 5,
        k_sparse: int = 5,
    ):
        self.dense = dense
        self.sparse = sparse
        self.k_dense = k_dense
        self.k_sparse = k_sparse

        self.logger = get_logger("rag.hybrid_retriever")

    def search(
        self, query: str
    ) -> Tuple[str, List[Dict[str, Any]], Dict[str, float]]:

        # ---- Dense retrieval ----
        status, dense_chunks, dense_scores = self.dense.search(
            query, self.k_dense
        )

        # Even if dense fails, sparse can rescue recall
        sparse_results = self.sparse.search(query, self.k_sparse)

        combined_chunks: Dict[str, Dict[str, Any]] = {}
        combined_scores: Dict[str, float] = {}

        # Add dense results
        for chunk, score in zip(dense_chunks, dense_scores):
            cid = chunk["chunk_id"]
            combined_chunks[cid] = chunk
            combined_scores[cid] = float(score)

        # Add sparse results (lower confidence, additive)
        for chunk, bm25_score in sparse_results:
            cid = chunk["chunk_id"]
            if cid not in combined_chunks:
                combined_chunks[cid] = chunk
                combined_scores[cid] = float(bm25_score)

        if not combined_chunks:
            self.logger.info(
                "event=HYBRID_NO_ANSWER | query_len=%d",
                len(query),
            )
            return "NO_ANSWER", [], {}

        self.logger.info(
            "event=HYBRID_RETRIEVAL | dense=%d | sparse=%d | merged=%d",
            len(dense_chunks),
            len(sparse_results),
            len(combined_chunks),
        )

        return "ANSWER", list(combined_chunks.values()), combined_scores
