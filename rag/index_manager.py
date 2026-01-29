import json
import hashlib
from pathlib import Path
from typing import List
from ingestion.loader import load_pdf
from ingestion.semantic_splitter import SemanticChunker
from rag.embedder import EmbeddingService
from rag.faiss_store import FaissStore
from api.config import settings

INDEX_DIR = Path("data")
META_FILE = INDEX_DIR/"index_meta.pkl"
INDEX_FILE = INDEX_DIR/"faiss.index"
META_STORE = INDEX_DIR/"meta.pkl"

INDEX_DIR.mkdir(exist_ok = True)

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
        "chunk_size": settings.CHUNK_SIZE,
        "chunk_overlap": settings.CHUNK_OVERLAP,
        "embedding_model": settings.EMBEDDING_MODEL,
    }, indent=2))

def build_or_load_index(pdf_paths: List[str]) -> FaissStore:
    fingerprint = compute_fingerprint(pdf_paths)
    meta = load_index_metadata()

    if meta and meta.get("fingerprint") == fingerprint and INDEX_FILE.exists():
        store = FaissStore(INDEX_FILE, META_STORE)
        store.load()
        print("[INDEX] Loaded existing FAISS index")
        return store

    print("[INDEX] Building FAISS index from scratch...")

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

    store = FaissStore(INDEX_FILE, META_STORE, dimension=embeddings.shape[1])
    store.add_chunks(embeddings, chunks)
    store.save()

    save_index_metadata(fingerprint, pdf_paths)

    return store
