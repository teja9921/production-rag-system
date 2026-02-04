import torch
from typing import List

from sentence_transformers import SentenceTransformer

from api.config import settings
from core.logger import get_logger
from core.exceptions import CustomException


logger = get_logger("rag.embedder")


class EmbeddingService:
    """
    Singleton embedding service.

    Guarantees:
    - single model instance per process
    - deterministic embeddings
    - explicit device control
    """

    _model = None
    _device = None

    def __init__(self):
        try:
            if EmbeddingService._model is None:
                device = "cpu"

                if device == "cuda":
                    if not torch.cuda.is_available():
                        logger.warning(
                            "event=EMBEDDER_FALLBACK_CPU | reason=cuda_not_available"
                        )
                        device = "cpu"

                EmbeddingService._device = device

                logger.info(
                    "event=EMBEDDER_INIT | model=%s | device=%s",
                    settings.EMBEDDING_MODEL,
                    device,
                )

                EmbeddingService._model = SentenceTransformer(
                    settings.EMBEDDING_MODEL,
                    device=device,
                )

            self.model = EmbeddingService._model

        except Exception as e:
            logger.exception("event=EMBEDDER_INIT_FAILED")
            raise CustomException(
                "Failed to initialize embedding model",
                error=e,
                context={
                    "model": settings.EMBEDDING_MODEL,
                    "device": settings.EMBEDDING_DEVICE,
                },
            ) from e

    def embed_texts(self, texts: List[str]) -> torch.Tensor:
        if not texts:
            raise ValueError("embed_texts received empty input")

        try:
            embeddings = self.model.encode(
                texts,
                batch_size=32,
                show_progress_bar=False,
                convert_to_tensor=True,
                normalize_embeddings=True,
            )

            logger.info(
                "event=EMBEDDINGS_GENERATED | count=%d | device=%s",
                len(texts),
                EmbeddingService._device,
            )

            return embeddings

        except Exception as e:
            logger.exception("event=EMBEDDINGS_FAILED")
            raise CustomException(
                "Embedding generation failed",
                error=e,
                context={
                    "text_count": len(texts),
                    "device": EmbeddingService._device,
                },
            ) from e
