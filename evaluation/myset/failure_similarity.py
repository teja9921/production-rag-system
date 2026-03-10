from pathlib import Path

from rag.embedder import EmbeddingService

try:
    import faiss
except ImportError as exc:
    raise ImportError(
        "faiss is not installed in this environment. Install faiss-cpu first."
    ) from exc

INDEX_PATH = Path("evaluation/myset/myset_failure.index")


class FailureSimilarityChecker:
    def __init__(self, threshold: float = 0.80):
        self.threshold = threshold
        self.embedder = EmbeddingService()
        if not INDEX_PATH.exists():
            raise FileNotFoundError(
                f"Failure index missing at {INDEX_PATH}. Build it first."
            )
        self.index = faiss.read_index(str(INDEX_PATH))

    def is_similar(self, query: str) -> bool:
        vec = self.embedder.embed_texts([query]).cpu().numpy()
        faiss.normalize_L2(vec)
        distances, _ = self.index.search(vec, 1)
        similarity = float(distances[0][0])
        return similarity >= self.threshold
