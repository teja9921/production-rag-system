import pytest
import torch

from core.exceptions import CustomException
from rag.embedder import EmbeddingService


class DummyModel:
    def encode(self, texts, **kwargs):
        return torch.zeros((len(texts), 384))


@pytest.fixture(autouse=True)
def reset_singleton():
    EmbeddingService._model = None
    EmbeddingService._device = None
    yield
    EmbeddingService._model = None
    EmbeddingService._device = None


def test_embedder_initializes_model(monkeypatch):
    called = {}

    def fake_sentence_transformer(model_name, device=None, model_kwargs=None):
        called["model_name"] = model_name
        called["device"] = device
        called["model_kwargs"] = model_kwargs
        return DummyModel()

    monkeypatch.setattr("rag.embedder.SentenceTransformer", fake_sentence_transformer)

    service = EmbeddingService()
    assert service.model is not None
    assert EmbeddingService._model is service.model
    assert called["device"] == "cpu"


def test_embedding_empty_input_raises(monkeypatch):
    monkeypatch.setattr("rag.embedder.SentenceTransformer", lambda *a, **k: DummyModel())
    service = EmbeddingService()
    with pytest.raises(ValueError):
        service.embed_texts([])


def test_model_init_error_wrapped(monkeypatch):
    def raise_init_error(*args, **kwargs):
        raise OSError("init failed")

    monkeypatch.setattr("rag.embedder.SentenceTransformer", raise_init_error)

    with pytest.raises(CustomException):
        EmbeddingService()
