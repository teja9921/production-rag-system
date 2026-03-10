from pathlib import Path

try:
    import faiss
except ImportError as exc:
    raise ImportError(
        "faiss is not installed in this environment. Install faiss-cpu first."
    ) from exc

from rag.embedder import EmbeddingService

INDEX_ROOT = Path("evaluation")


class FailureSimilarityChecker:
    def __init__(self, index_path: Path, threshold: float = 0.80):
        self.index_path = index_path
        self.threshold = threshold
        self.embedder = EmbeddingService()
        if not self.index_path.exists():
            raise FileNotFoundError(
                f"Failure index missing at {self.index_path}. Build it first."
            )
        self.index = faiss.read_index(str(self.index_path))

    def is_similar(self, query: str) -> bool:
        vec = self.embedder.embed_texts([query]).cpu().numpy()
        faiss.normalize_L2(vec)
        distances, _ = self.index.search(vec, 1)
        similarity = float(distances[0][0])  # inner product on normalized vectors
        return similarity >= self.threshold


def _dataset_name_from_version(dataset_version: str) -> str:
    # gale_v1 -> gale, myset_v1 -> myset
    if "_v" in dataset_version:
        return dataset_version.split("_v", 1)[0]
    return dataset_version


def resolve_index_path(dataset_version: str) -> Path:
    dataset_name = _dataset_name_from_version(dataset_version)
    return INDEX_ROOT / dataset_name / f"{dataset_name}_failure.index"


def build_failure_checker(dataset_version: str, threshold: float = 0.80):
    index_path = resolve_index_path(dataset_version)
    return FailureSimilarityChecker(index_path=index_path, threshold=threshold)
