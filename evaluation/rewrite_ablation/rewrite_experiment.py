import json
import time
from pathlib import Path

from rag.embedder import EmbeddingService
from rag.faiss_store import FaissStore
from rag.retriever import Retriever
from rag.hybrid_retriever import HybridRetriever
from rag.reranker import CrossEncoderReranker
from rag.index_manager import build_or_load_index
from orchestration.rewrite import QueryWriter
from api.config import settings


PDFS = ["data/The_GALE_ENCYCLOPEDIA_of_MEDICINE_SECOND.pdf"]
EVAL_FILE = "evaluation/gale/evaluation_gale_final.json"
TOP_K = 5


# -----------------------------
# Conditional Rewrite Policy
# -----------------------------
PRONOUNS = {"it", "this", "that", "they", "he", "she", "them", "its"}

def needs_rewrite(query: str):
    tokens = query.lower().split()
    if len(tokens) < 8:
        return True
    if any(p in tokens for p in PRONOUNS):
        return True
    return False


# -----------------------------
# Build Retrieval Stack
# -----------------------------
def build_pipeline():
    embedder = EmbeddingService()
    faiss_store, bm25_store = build_or_load_index(PDFS)

    dense = Retriever(embedder, faiss_store)

    hybrid = HybridRetriever(
        dense=dense,
        sparse=bm25_store,
        k_dense=5,
        k_sparse=5,
    )

    reranker = CrossEncoderReranker()

    return hybrid, reranker

# -------------------------------------------------
# LOAD FILES SAFELY WITHOUT CRASHING
# -------------------------------------------------
def load_json_robust(path: Path):
    raw = path.read_bytes()
    try:
        return json.loads(raw.decode("utf-8"))
    except UnicodeDecodeError:
        return json.loads(raw.decode("cp1252"))
    
# -----------------------------
# Evaluate Single Policy
# -----------------------------
def evaluate(policy_name, hybrid, reranker):
    data = load_json_robust(Path(EVAL_FILE))
    rewriter = QueryWriter()

    hits = 0
    total_latency = []

    for item in data:
        query = item["question"]
        answer_span = item["answer_span"]

        start = time.perf_counter()

        # Rewrite logic
        if policy_name == "rewrite_off":
            final_query = query

        elif policy_name == "rewrite_always":
            state = {"query": query, "history": ""}
            rewritten = rewriter(state, True)
            final_query = rewritten.get("rewritten_query") or query

        elif policy_name == "rewrite_conditional":
            if needs_rewrite(query):
                state = {"query": query, "history": ""}
                rewritten = rewriter(state, True)
                final_query = rewritten.get("rewritten_query") or query
            else:
                final_query = query

        else:
            raise ValueError("Unknown policy")

        status, chunks, scores = hybrid.search(final_query)

        if status == "ANSWER":
            reranked = reranker.rerank(final_query, chunks, top_k=TOP_K)

            for c in reranked:
                if answer_span.lower() in c["content"].lower():
                    hits += 1
                    break

        latency = time.perf_counter() - start
        total_latency.append(latency)

    recall = hits / len(data)
    mean_latency = sum(total_latency) / len(total_latency)

    return {
        "policy": policy_name,
        "recall@5": recall,
        "mean_latency_ms": mean_latency * 1000,
    }


# -----------------------------
# Run All Policies
# -----------------------------
def run():
    hybrid, reranker = build_pipeline()

    policies = [
        "rewrite_off",
        "rewrite_always",
        "rewrite_conditional",
    ]

    results = {}

    for p in policies:
        print("Running:", p)
        results[p] = evaluate(p, hybrid, reranker)

    Path("evaluation/rewrite_ablation/rewrite_ablation_results.json").write_text(
        json.dumps(results, indent=2)
    )


if __name__ == "__main__":
    run()