import faiss
import numpy as np
import pickle
from pathlib import Path
from typing import List, Dict, Any, Optional

from core.logger import get_logger
from core.exceptions import CustomException

class FaissStore:
    """
    FAISS vector store with strict lifecycle guarantees.

    Lifecycle:
    - Either `build` OR `load`
    - Never both in the same process
    """

    def __init__(
        self,
        index_path: str,
        metadata_path: str,
        dimension: Optional[int] = None,
    ):
        self.index_path = Path(index_path)
        self.metadata_path = Path(metadata_path)

        self.index = None
        self.dim = dimension
        self.metadata: List[Dict[str, Any]] = []

        self._loaded = False
        self._built = False

        self.logger = get_logger("rag.faiss_store")

    def add_chunks(self, embeddings: np.ndarray, chunks: List[Dict[str, Any]]):
        if self._loaded:
            raise RuntimeError("Cannot add chunks to a loaded index")

        if embeddings.ndim != 2:
            raise ValueError("Embeddings must be 2D")

        if len(embeddings) != len(chunks):
            raise ValueError("Embedding/chunk length mismatch")

        if self.dim is None:
            self.dim = embeddings.shape[1]

        if embeddings.shape[1] != self.dim:
            raise ValueError(
                f"Embedding dimension mismatch: expected {self.dim}, got {embeddings.shape[1]}"
            )

        if self.index is None:
            self.logger.info(
                "event=FAISS_INDEX_INIT | dim=%d",
                self.dim,
            )
            self.index = faiss.IndexFlatIP(self.dim)

        self.index.add(embeddings.astype("float32"))
        self.metadata.extend(chunks)
        self._built = True

        self.logger.info(
            "event=FAISS_CHUNKS_ADDED | vectors=%d | total=%d",
            len(embeddings),
            self.index.ntotal,
        )

    def save(self):
        if not self._built or self.index is None:
            raise RuntimeError("Cannot save empty FAISS index")

        if self.index.ntotal != len(self.metadata):
            raise RuntimeError(
                "Index/metadata length mismatch before save"
            )

        self.index_path.parent.mkdir(parents=True, exist_ok=True)

        faiss.write_index(self.index, str(self.index_path))
        with open(self.metadata_path, "wb") as f:
            pickle.dump(self.metadata, f)

        self.logger.info(
            "event=FAISS_INDEX_SAVED | path=%s | vectors=%d",
            self.index_path,
            self.index.ntotal,
        )

    def load(self):
        if self._built:
            raise RuntimeError("Cannot load index after building in same process")

        if not self.index_path.exists() or not self.metadata_path.exists():
            raise FileNotFoundError("FAISS index or metadata missing")

        self.index = faiss.read_index(str(self.index_path))
        with open(self.metadata_path, "rb") as f:
            self.metadata = pickle.load(f)

        if self.index.ntotal != len(self.metadata):
            raise RuntimeError(
                "Loaded index/metadata length mismatch"
            )

        self.dim = self.index.d
        self._loaded = True

        self.logger.info(
            "event=FAISS_INDEX_LOADED | vectors=%d | dim=%d",
            self.index.ntotal,
            self.dim,
        )
