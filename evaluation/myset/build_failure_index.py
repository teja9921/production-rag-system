import json
from pathlib import Path

import numpy as np

from rag.embedder import EmbeddingService

try:
    import faiss
except ImportError as exc:
    raise ImportError(
        "faiss is not installed in this environment. Install faiss-cpu first."
    ) from exc

FAILURE_FILE = Path("evaluation/myset/myset_failures.json")
INDEX_PATH = Path("evaluation/myset/myset_failure.index")
META_PATH = Path("evaluation/myset/myset_failure_meta.npy")


def run():
    data = json.loads(FAILURE_FILE.read_text(encoding="utf-8"))
    questions = [d["question"] for d in data]
    if not questions:
        raise ValueError("No failed questions found to build failure index.")

    embedder = EmbeddingService()
    vectors = embedder.embed_texts(questions).cpu().numpy().astype(np.float32)

    dim = vectors.shape[1]
    index = faiss.IndexFlatIP(dim)
    faiss.normalize_L2(vectors)
    index.add(vectors)

    INDEX_PATH.parent.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(INDEX_PATH))
    np.save(str(META_PATH), vectors)

    print(f"Failure index built with {len(questions)} vectors.")


if __name__ == "__main__":
    run()
