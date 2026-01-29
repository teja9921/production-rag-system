from rag.embedder import EmbeddingService
from rag.faiss_store import FaissStore
from rag.retriever import Retriever
from rag.index_manager import build_or_load_index
from orchestration.lc_retriever import RetrieverRunnable
from orchestration.lc_llm import LLMRunnable
from orchestration.graph import build_graph
from api.config import settings

PDFs = ["data/The_GALE_ENCYCLOPEDIA_of_MEDICINE_SECOND.pdf"]

_embedder = EmbeddingService()
_store = build_or_load_index(PDFs)

_retriever = Retriever(_embedder, _store)
_retriever_runnable = RetrieverRunnable(_retriever)
_llm_runnable = LLMRunnable()

GRAPH = build_graph(_retriever_runnable, _llm_runnable)

