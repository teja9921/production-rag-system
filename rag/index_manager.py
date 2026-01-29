import json
import hashlib
from pathlib import Path
from typing import List
from ingestion.loader import load_pdf
from ingestion.semantic_splitter import SemanticChunker
from rag.embedder import EmbeddingService
from rag.faiss_store import FaissStore
from rag.bm25_store import BM25Store
from api.config import settings

INDEX_DIR = Path("data")
INDEX_DIR.mkdir(exist_ok = True)

FAISS_INDEX = INDEX_DIR/"faiss.index"
FAISS_META = INDEX_DIR/"meta.pkl"
BM25_INDEX = INDEX_DIR/"bm25.pkl"
META_FILE = INDEX_DIR/"index_meta.json"


def compute_fingerprint(pdf_paths: List[str]) -> str:
    hasher = hashlib.sha256()

    for path in sorted(pdf_paths):
        with open(path, "rb") as f:
            while chunk := f.read(8192):
                hasher.update(chunk)

    config_blob = f"{settings.CHUNK_SIZE}-{settings.CHUNK_OVERLAP}-{settings.EMBEDDING_MODEL}"
    hasher.update(config_blob.encode())

    return hasher.hexdigest()


def load_index_metadata():
    if not META_FILE.exists():
        return None
    return json.loads(META_FILE.read_text())

def save_index_metadata(fingerprint: str, pdf_paths: List[str]):
    META_FILE.write_text(json.dumps({
        "fingerprint": fingerprint,
        "pdf_files": pdf_paths,
        "max_chars": settings.MAX_CHARS,
        "min_chars": settings.MIN_CHARS,
        "embedding_model": settings.EMBEDDING_MODEL,
        "chunking": "semantic"
    }, indent=2))

def build_or_load_index(pdf_paths: List[str]) -> FaissStore:
    fingerprint = compute_fingerprint(pdf_paths)
    meta = load_index_metadata()

    if meta and meta.get("fingerprint") == fingerprint \
        and FAISS_INDEX.exists() and BM25_INDEX.exists():

        print("[INDEX] Loading existing FAISS index + BM25 index")
        faiss_store = FaissStore(FAISS_INDEX, FAISS_META)
        faiss_store.load()

        bm25_store = BM25Store(BM25_INDEX)
        bm25_store.load()

        return faiss_store, bm25_store

    print("[INDEX] Rebuilding FAISS index + BM25 indxes...")

    pages = []
    for pdf in pdf_paths:
        pages.extend(load_pdf(pdf))

    # chunker = SimpleChunker(
    #     chunk_size=settings.CHUNK_SIZE,
    #     chunk_overlap=settings.CHUNK_OVERLAP
    # )
    # chunks = chunker.split_pages(pages)
    
    chunker = SemanticChunker(
        max_chars = settings.MAX_CHARS,
        min_chars = settings.MIN_CHARS
    )
    chunks = chunker._split_pages(pages)

    embedder = EmbeddingService()
    embeddings = embedder.embed_texts(
        [c["content"] for c in chunks]
    ).cpu().numpy()

    faiss_store = FaissStore(FAISS_INDEX, FAISS_META, dimension=embeddings.shape[1])
    faiss_store.add_chunks(embeddings, chunks)
    faiss_store.save()

    bm25_store = BM25Store(BM25_INDEX)
    bm25_store.build(chunks)
    bm25_store.save()

    save_index_metadata(fingerprint, pdf_paths)

    return faiss_store, bm25_store
