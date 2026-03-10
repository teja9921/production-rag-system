import csv
import json
import time
from pathlib import Path
from typing import Dict, List, Optional

from rag.embedder import EmbeddingService
from rag.hybrid_retriever import HybridRetriever
from rag.index_manager import build_or_load_index
from rag.reranker import CrossEncoderReranker
from rag.retriever import Retriever

# -----------------------------
# Config (myset)
# -----------------------------
EVAL_FILE = Path("evaluation/myset/evaluation_myset_final.json")
OUTPUT_FILE = Path("evaluation/myset/outputs/eval_outputs/myset_hybrid_rerank.csv")
PDFS = ["data/myset_source.pdf"]
TOP_K = 5
CONFIG_ID = "myset_hybrid_rerank_v1"


def compute_ranks(retrieved_ids: List[str], gold_id: str) -> Optional[int]:
    try:
        return retrieved_ids.index(gold_id) + 1
    except ValueError:
        return None


def recall_at_k(rank: Optional[int], k: int) -> int:
    return int(rank is not None and rank <= k)


def main():
    data = json.loads(EVAL_FILE.read_text(encoding="utf-8"))

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

    results: List[Dict] = []

    for row in data:
        qid = row["id"]
        question = row["question"]
        gold_chunk = row["chunk_id"]
        difficulty = row.get("difficulty", "unknown")

        start = time.perf_counter()
        status, chunks, _ = hybrid.search(question)

        if status == "NO_ANSWER":
            latency = int((time.perf_counter() - start) * 1000)
            results.append(
                {
                    "question_id": qid,
                    "question": question,
                    "difficulty": difficulty,
                    "gold_rank": "",
                    "reciprocal_rank": 0.0,
                    "recall@1": 0,
                    "recall@3": 0,
                    "recall@5": 0,
                    "latency_ms": latency,
                    "config_id": CONFIG_ID,
                }
            )
            continue

        reranked = reranker.rerank(question, chunks, top_k=TOP_K)
        retrieved_ids = [c["chunk_id"] for c in reranked]

        latency = int((time.perf_counter() - start) * 1000)
        rank = compute_ranks(retrieved_ids, gold_chunk)

        results.append(
            {
                "question_id": qid,
                "question": question,
                "difficulty": difficulty,
                "gold_rank": rank if rank is not None else "",
                "reciprocal_rank": 1.0 / rank if rank else 0.0,
                "recall@1": recall_at_k(rank, 1),
                "recall@3": recall_at_k(rank, 3),
                "recall@5": recall_at_k(rank, 5),
                "latency_ms": latency,
                "config_id": CONFIG_ID,
            }
        )

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_FILE, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)

    print(f"Saved results -> {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
