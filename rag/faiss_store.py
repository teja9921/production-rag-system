import faiss
import numpy as np
import pickle
from pathlib import Path
from typing import List, Dict, Any, Optional

class FaissStore:
    def __init__(self, index_path: str, metadata_path: str, dimension: Optional[int] = None):
        self.index_path = Path(index_path)
        self.metadata_path = Path(metadata_path)
        self.index = None
        self.dim = dimension
        self.metadata: List[Dict[str, Any]] = []

    def add_chunks(self, embeddings: np.ndarray, chunks: List[Dict[str, Any]]):
        if embeddings.ndim != 2:
            raise ValueError("Embeddings must be 2D")

        if self.dim is None:
            self.dim = embeddings.shape[1]

        if self.index is None:
            self.index = faiss.IndexFlatIP(self.dim)

        if len(embeddings) != len(chunks):
            raise ValueError("Embedding/chunk length mismatch")

        self.index.add(embeddings.astype("float32"))
        self.metadata.extend(chunks)

    def save(self):
        if self.index is None:
            raise RuntimeError("No index to save")

        faiss.write_index(self.index, str(self.index_path))
        with open(self.metadata_path, "wb") as f:
            pickle.dump(self.metadata, f)

    def load(self):
        if not self.index_path.exists() or not self.metadata_path.exists():
            raise FileNotFoundError("Index or metadata missing")

        self.index = faiss.read_index(str(self.index_path))
        with open(self.metadata_path, "rb") as f:
            self.metadata = pickle.load(f)
