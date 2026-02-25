import json
import re
from pathlib import Path
import numpy as np
import nltk

from ingestion.loader import load_pdf
from rag.embedder import EmbeddingService
from rag.faiss_store import FaissStore
from rag.retriever import Retriever

from rag.chunking.paragraph_chunker import ParagraphChunker
from rag.chunking.sentence_chunker import SentenceChunker
from rag.chunking.sliding_window_chunker import SlidingWindowChunker
from ingestion.semantic_splitter import SemanticChunker


# -------------------------------------------------
# CONFIG
# -------------------------------------------------
PDF = "data/The_GALE_ENCYCLOPEDIA_of_MEDICINE_SECOND.pdf"
GALE_EVAL = "evaluation/gale/evaluation_gale_final.json"
OUTPUT_FILE = "evaluation/chunking_ablation/chunking_results.json"

TOP_K = 5
SEMANTIC_THRESHOLD = 0.70

nltk.download("punkt")


# -------------------------------------------------
# TEXT NORMALIZATION (medical aware)
# -------------------------------------------------
def normalize(text: str) -> str:
    text = text.lower()

    # normalize common medical numeric forms
    text = re.sub(r"type\s*ii", "type 2", text)
    text = re.sub(r"type\s*i", "type 1", text)

    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


# -------------------------------------------------
# SENTENCE SPLIT
# -------------------------------------------------
from nltk.tokenize import sent_tokenize
def split_sentences(text: str):
    return sent_tokenize(text)


# -------------------------------------------------
# BUILD RETRIEVAL INDEX
# -------------------------------------------------
def build_index(chunks, embedder):

    vecs = embedder.embed_texts(
        [c["content"] for c in chunks]
    ).cpu().numpy()

    store = FaissStore("tmp.index", "tmp.pkl", dimension=vecs.shape[1])
    store.add_chunks(vecs, chunks)

    return Retriever(embedder, store)


# -------------------------------------------------
# LEXICAL CONTAINMENT
# -------------------------------------------------
def lexical_hit(answer, chunks):
    a = normalize(answer)
    for c in chunks:
        if a in normalize(c["content"]):
            return True
    return False


# -------------------------------------------------
# SEMANTIC MATCH (efficient)
# -------------------------------------------------
def semantic_match(answer, chunks, embedder, threshold):

    if not answer:
        return False, 0.0

    a_vec = embedder.embed_texts([answer]).cpu().numpy()[0]

    sentences = []
    for c in chunks:
        sentences.extend([
            s for s in split_sentences(c["content"])
            if s.strip()
        ])

    if not sentences:
        return False, 0.0

    sent_vecs = embedder.embed_texts(sentences).cpu().numpy()

    sims = sent_vecs @ a_vec
    best = float(sims.max())

    return best >= threshold, best


# -------------------------------------------------
# EVALUATE CHUNKER
# -------------------------------------------------
def evaluate_chunker(retriever, embedder, eval_data):

    lexical_hits = 0
    semantic_hits = 0
    similarity_scores = []

    for item in eval_data:

        question = item["question"]
        answer = item.get("answer_span") or item["answer"]

        status, retrieved_chunks, _ = retriever.search(question, TOP_K)

        if status == "NO_ANSWER":
            continue

        if lexical_hit(answer, retrieved_chunks):
            lexical_hits += 1

        sem_hit, best_sim = semantic_match(
            answer,
            retrieved_chunks,
            embedder,
            SEMANTIC_THRESHOLD
        )

        if sem_hit:
            semantic_hits += 1

        similarity_scores.append(best_sim)

    n = len(eval_data)

    return {
        "lexical_recall@5": lexical_hits / n,
        "semantic_recall@5": semantic_hits / n,
        "mean_best_similarity": float(np.mean(similarity_scores))
    }


# -------------------------------------------------
# SAFE CHUNKING CALL
# -------------------------------------------------
def run_chunker(chunker, pages):
    if hasattr(chunker, "split_pages"):
        return chunker.split_pages(pages)
    return chunker.split(pages)

# -------------------------------------------------
# LOAD FILES SAFELY WITHOUT CRASHING
# -------------------------------------------------
def load_json_robust(path: Path):
    raw = path.read_bytes()
    try:
        return json.loads(raw.decode("utf-8"))
    except UnicodeDecodeError:
        return json.loads(raw.decode("cp1252"))

# -------------------------------------------------
# MAIN EXPERIMENT
# -------------------------------------------------
def run():

    print("Loading PDF...")
    pages = load_pdf(PDF)

    print("Loading evaluation set...")
    data = load_json_robust(GALE_EVAL)

    embedder = EmbeddingService()

    experiments = {
        "paragraph_small": ParagraphChunker(800, 150),
        "paragraph_large": ParagraphChunker(1600, 300),
        "semantic_default": SemanticChunker(),
        "sentence_boundary": SentenceChunker(),
        "sliding_window": SlidingWindowChunker(),
    }

    results = {}

    for name, chunker in experiments.items():

        print(f"\nRunning experiment: {name}")

        chunks = run_chunker(chunker, pages)

        print("Chunks created:", len(chunks))

        retriever = build_index(chunks, embedder)

        metrics = evaluate_chunker(
            retriever,
            embedder,
            eval_data
        )

        results[name] = metrics

    Path(OUTPUT_FILE).write_text(json.dumps(results, indent=2))
    print("\nSaved results to:", OUTPUT_FILE)


# -------------------------------------------------
if __name__ == "__main__":
    run()