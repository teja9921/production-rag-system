import json
import hashlib
from pathlib import Path
from typing import List, Tuple

from ingestion.loader import load_pdf
from ingestion.semantic_splitter import SemanticChunker
from rag.embedder import EmbeddingService
from rag.faiss_store import FaissStore
from rag.bm25_store import BM25Store
from api.config import settings
from core.logger import get_logger
from core.exceptions import CustomException

logger = get_logger("rag.index_manager")

INDEX_DIR = Path("data/index")
INDEX_DIR.mkdir(parents=True, exist_ok=True)

FAISS_INDEX = INDEX_DIR / "faiss.index"
FAISS_META = INDEX_DIR / "faiss_meta.pkl"
BM25_INDEX = INDEX_DIR / "bm25.pkl"
META_FILE = INDEX_DIR / "index_meta.json"


def compute_fingerprint(pdf_paths: List[str]) -> str:
    hasher = hashlib.sha256()

    for path in sorted(pdf_paths):
        with open(path, "rb") as f:
            while chunk := f.read(8192):
                hasher.update(chunk)

    config_blob = (
        f"{settings.EMBEDDING_MODEL}-"
        f"{settings.MAX_CHARS}-"
        f"{settings.MIN_CHARS}-"
        "semantic"
    )

    hasher.update(config_blob.encode())
    return hasher.hexdigest()


def load_index_metadata():
    if not META_FILE.exists():
        return None
    return json.loads(META_FILE.read_text())


def save_index_metadata(fingerprint: str, pdf_paths: List[str]):
    META_FILE.write_text(
        json.dumps(
            {
                "fingerprint": fingerprint,
                "pdf_files": pdf_paths,
                "embedding_model": settings.EMBEDDING_MODEL,
                "chunking": "semantic",
                "max_chars": settings.MAX_CHARS,
                "min_chars": settings.MIN_CHARS,
            },
            indent=2,
        )
    )


def build_or_load_index(
    pdf_paths: List[str],
) -> Tuple[FaissStore, BM25Store]:

    fingerprint = compute_fingerprint(pdf_paths)
    meta = load_index_metadata()

    if (
        meta
        and meta.get("fingerprint") == fingerprint
        and FAISS_INDEX.exists()
        and BM25_INDEX.exists()
    ):
        logger.info("event=INDEX_LOAD_START")

        faiss_store = FaissStore(FAISS_INDEX, FAISS_META)
        faiss_store.load()

        bm25_store = BM25Store(BM25_INDEX)
        bm25_store.load()

        logger.info(
            "event=INDEX_LOADED | faiss_vectors=%d | bm25_docs=%d",
            len(faiss_store.metadata),
            len(bm25_store.documents),
        )

        return faiss_store, bm25_store

    logger.warning("event=INDEX_REBUILD_START")

    try:
        pages = []
        for pdf in pdf_paths:
            pages.extend(load_pdf(pdf))

        chunker = SemanticChunker(
            max_chars=settings.MAX_CHARS,
            min_chars=settings.MIN_CHARS,
        )
        chunks = chunker.split_pages(pages)

        embedder = EmbeddingService()
        embeddings = embedder.embed_texts(
            [c["content"] for c in chunks]
        ).cpu().numpy()

        faiss_store = FaissStore(
            FAISS_INDEX,
            FAISS_META,
            dimension=embeddings.shape[1],
        )
        faiss_store.add_chunks(embeddings, chunks)
        faiss_store.save()

        bm25_store = BM25Store(BM25_INDEX)
        bm25_store.build(chunks)
        bm25_store.save()

        save_index_metadata(fingerprint, pdf_paths)

        logger.info(
            "event=INDEX_REBUILT | chunks=%d",
            len(chunks),
        )

        return faiss_store, bm25_store

    except Exception as e:
        logger.exception("event=INDEX_BUILD_FAILED")
        raise CustomException(
            "Index build/load failed",
            error=e,
            context={
                "pdf_count": len(pdf_paths),
                "fingerprint": fingerprint,
            },
        ) from e
