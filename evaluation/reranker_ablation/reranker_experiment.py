import json
from pathlib import Path

from rag.embedder import EmbeddingService
from rag.index_manager import build_or_load_index
from rag.retriever import Retriever
from rag.hybrid_retriever import HybridRetriever
from orchestration.rewrite import QueryWriter

PDFS = ["data/The_GALE_ENCYCLOPEDIA_of_MEDICINE_SECOND.pdf"]
EVAL_FILE = "evaluation/gale/evaluation_gale_final.json"

PRONOUNS = {"it", "this", "that", "they", "he", "she", "them", "its"}

def needs_rewrite(query):
    tokens = query.lower().split()
    if len(tokens) < 8:
        return True
    if any(p in tokens for p in PRONOUNS):
        return True
    return False

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
    return hybrid

# -------------------------------------------------
# LOAD FILES SAFELY WITHOUT CRASHING
# -------------------------------------------------
def load_json_robust(path: Path):
    raw = path.read_bytes()
    try:
        return json.loads(raw.decode("utf-8"))
    except UnicodeDecodeError:
        return json.loads(raw.decode("cp1252"))

def retrieve_all_chunks():
    data = load_json_robust(Path(EVAL_FILE))
    rewriter = QueryWriter()
    hybrid = build_pipeline()

    retrieval_data = []

    for item in data:
        query = item["question"]
        answer_span = item["answer_span"]

        if needs_rewrite(query):
            state = {"query": query, "history": ""}
            rewritten = rewriter(state)
            final_query = rewritten.get("rewritten_query") or query
        else:
            final_query = query

        status, chunks, scores = hybrid.search(final_query)
        chunk_entries = []

        for i, c in enumerate(chunks):
            score_val = 0.0
            try:
                if isinstance(scores, dict):
                    cid = c.get("chunk_id") if isinstance(c, dict) else None
                    score_val = float(scores.get(cid, 0.0))
                else:
                    score_val = float(scores[i]) if scores and scores[i] is not None else 0.0
            except Exception:
                score_val = 0.0

            chunk_entries.append({"content": c, "score": score_val})

        retrieval_data.append({
            "original_query": query,
            "rewritten_query": final_query,
            "answer_span": answer_span,
            "chunks": chunk_entries,
            "status": status,
        })

    Path("evaluation/reranker_ablation/retrieved_chunks.json").write_text(
        json.dumps(retrieval_data, indent=2)
    )
    print(f"Saved {len(retrieval_data)} retrieval results")

if __name__ == "__main__":
    retrieve_all_chunks()