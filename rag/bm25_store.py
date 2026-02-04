import re
import pickle
from pathlib import Path
from typing import List, Dict, Any

from rank_bm25 import BM25Okapi

from core.logger import get_logger
from core.exceptions import CustomException

class BM25Store:
    """
    Sparse lexical retriever (BM25).

    Guarantees: 
    - deterministic preprocessing
    - strict build/load lifecycle
    - aligned document corpus
    """

    def __init__(self, path: str):
        self.path = Path(path)
        self.bm25 = None
        self.documents: List[Dict[str, Any]] = []

        self._built = False
        self._loaded = False

        self.logger = get_logger("rag.bm25_store")

    def _preprocess(self, text: str) -> List[str]:
        """
        Medical-safe preprocessing:
        - lowercase
        - preserve alphanumerics
        - preserve hyphenated medical terms
        - normalize slashes
        """
        text = text.lower()
        text = text.replace("/", " ")

        # keep words, numbers, hyphens (important for medical terms)
        text = re.sub(r"[^a-z0-9\\-\\s]", " ", text)

        tokens = text.split()
        return tokens

    def build(self, chunks: List[Dict[str, Any]]):
        if self._loaded:
            raise RuntimeError("Cannot build BM25 after loading")

        if not chunks:
            raise ValueError("BM25 build received empty chunks")

        corpus = [self._preprocess(c["content"]) for c in chunks]

        self.bm25 = BM25Okapi(corpus)
        self.documents = chunks
        self._built = True

        self.logger.info(
            "event=BM25_BUILT | docs=%d",
            len(chunks),
        )

    def save(self):
        if not self._built or self.bm25 is None:
            raise RuntimeError("Cannot save empty BM25 index")

        with open(self.path, "wb") as f:
            pickle.dump(
                {
                    "bm25": self.bm25,
                    "documents": self.documents,
                },
                f,
            )

        self.logger.info(
            "event=BM25_SAVED | path=%s | docs=%d",
            self.path,
            len(self.documents),
        )

    def load(self):
        if self._built:
            raise RuntimeError("Cannot load BM25 after building")

        if not self.path.exists():
            raise FileNotFoundError("BM25 file not found")

        with open(self.path, "rb") as f:
            data = pickle.load(f)

        self.bm25 = data["bm25"]
        self.documents = data["documents"]
        self._loaded = True

        if self.bm25 is None or not self.documents:
            raise RuntimeError("Loaded BM25 index is invalid")

        self.logger.info(
            "event=BM25_LOADED | docs=%d",
            len(self.documents),
        )

    def search(self, query: str, k: int):
        if self.bm25 is None:
            raise RuntimeError("BM25 index not initialized")

        tokens = self._preprocess(query)
        scores = self.bm25.get_scores(tokens)

        top_indices = sorted(
            range(len(scores)),
            key=lambda i: scores[i],
            reverse=True,
        )[:k]

        results = [
            (self.documents[i], float(scores[i]))
            for i in top_indices
        ]

        self.logger.info(
            "event=BM25_SEARCH | query_len=%d | returned=%d",
            len(tokens),
            len(results),
        )

        return results