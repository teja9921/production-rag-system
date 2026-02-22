import json
import time
import csv
from pathlib import Path
from typing import Dict, Any, List, Optional

# ---- IMPORT YOUR PIPELINE ----
from rag.index_manager import build_or_load_index
from rag.embedder import EmbeddingService
from rag.retriever import Retriever
from rag.hybrid_retriever import HybridRetriever
from rag.reranker import CrossEncoderReranker
from api.config import settings

# -----------------------------
# Config
# -----------------------------

EVAL_FILE = Path("evaluation/gale/evaluation_gale_final.json")
OUTPUT_FILE = Path("evaluation/eval_outputs/bm25.csv")

PDFS = ["data/The_GALE_ENCYCLOPEDIA_of_MEDICINE_SECOND.pdf"]

TOP_K = 5
CONFIG_ID = "bm25_v1"

# -----------------------------
# Helpers
# -----------------------------

def compute_ranks(retrieved_ids: List[str], gold_id: str) -> Optional[int]:
    """
    Returns 1-indexed rank of gold chunk, or None if missing.
    """
    try:
        return retrieved_ids.index(gold_id) + 1
    except ValueError:
        return None


def recall_at_k(rank: Optional[int], k: int) -> int:
    return int(rank is not None and rank <= k)


# -----------------------------
# Main
# -----------------------------

def main():
    # Load eval data
    data = json.loads(EVAL_FILE.read_text())

    # Build pipeline
    embedder = EmbeddingService()
    faiss_store, bm25_store = build_or_load_index(PDFS)

    dense = Retriever(embedder, faiss_store)
    hybrid = HybridRetriever(
        dense=dense,
        sparse=bm25_store,
        k_dense=TOP_K,
        k_sparse=TOP_K,
    )

    reranker = CrossEncoderReranker()

    results = []

    for row in data:
        qid = row["id"]
        question = row["question"]
        gold_chunk = row["chunk_id"]
        difficulty = row.get("difficulty", "unknown")

        start = time.perf_counter()

        #status, chunks, _ = dense.search(question,5) #used for dense_search
        results_with_scores = bm25_store.search(question, 5) #used for sparse or keyword search (bm25)
        '''
        if status == "NO_ANSWER":
            latency = (time.perf_counter() - start) * 1000
            results.append({
                "question_id": qid,
                "difficulty": difficulty,
                "gold_rank": None,
                "reciprocal_rank": 0.0,
                "recall@1": 0,
                "recall@3": 0,
                "recall@5": 0,
                "latency_ms": int(latency),
                "config_id": CONFIG_ID,
            })
            continue
            '''
        
        chunks = [doc for doc, score in results_with_scores] #used for bm25
        #reranked = reranker.rerank(question, chunks, top_k=TOP_K)
        retrieved_ids = [c["chunk_id"] for c in chunks]

        latency = (time.perf_counter() - start) * 1000
        rank = compute_ranks(retrieved_ids, gold_chunk)

        results.append({
            "question_id": qid,
            "difficulty": difficulty,
            "gold_rank": rank,
            "reciprocal_rank": 1.0 / rank if rank else 0.0,
            "recall@1": recall_at_k(rank, 1),
            "recall@3": recall_at_k(rank, 3),
            "recall@5": recall_at_k(rank, 5),
            "latency_ms": int(latency),
            "config_id": CONFIG_ID,
        })

    # Write CSV
    with open(OUTPUT_FILE, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)

    print(f"Saved results → {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
