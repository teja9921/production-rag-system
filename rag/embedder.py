# rag/embedder.py

import torch
from sentence_transformers import SentenceTransformer
from typing import List
from api.config import settings

class EmbeddingService:
    _model = None
    _device = "cpu"

    def __init__(self):
        if EmbeddingService._model is None:
            EmbeddingService._model = SentenceTransformer(
                settings.EMBEDDING_MODEL,
                device=self._device
            )
        self.model = EmbeddingService._model

    def embed_texts(self, texts: List[str]) -> torch.Tensor:
        if not texts:
            raise ValueError("embed_texts received empty input")

        return self.model.encode(
            texts,
            batch_size=32,
            show_progress_bar=False,
            convert_to_tensor=True,
            normalize_embeddings=True
        )
