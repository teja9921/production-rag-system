from ingestion.loader import load_pdf
from ingestion.splitter import SimpleChunker
from rag.embedder import EmbeddingService
from rag.faiss_store import FaissStore
from rag.retriever import Retriever
from orchestration.lc_retriever import RetrieverRunnable
from orchestration.lc_llm import LLMRunnable
from orchestration.graph import build_graph
from api.config import settings

PDF_FILE = "data/LLM_Interview_Questions.pdf"
INDEX_PATH = "data/faiss.index"
META_PATH = "data/faiss_meta.pkl"

_pages = load_pdf(PDF_FILE)
_chunker = SimpleChunker(
    settings.CHUNK_SIZE, 
    settings.CHUNK_OVERLAP
    )
_chunks = _chunker.split_pages(_pages)

_embedder = EmbeddingService()
_embeddings = _embedder.embed_texts([c["content"] for c in _chunks]).cpu().numpy()

_store = FaissStore(INDEX_PATH, META_PATH)
_store.add_chunks(_embeddings, _chunks)

_retriever = Retriever(_embedder, _store)

_retriever_runnable = RetrieverRunnable(_retriever)
_llm_runnable = LLMRunnable()

GRAPH = build_graph(_retriever_runnable, _llm_runnable)

