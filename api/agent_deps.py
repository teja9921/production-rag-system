from rag.embedder import EmbeddingService
from rag.retriever import Retriever
from rag.reranker import CrossEncoderReranker
from rag.hybrid_retriever import HybridRetriever
from orchestration.lc_retriever import RetrieverRunnable
from orchestration.lc_llm import LLMRunnable
from orchestration.rewrite import QueryWriter
from orchestration.reasoning_graph import build_reasoning_graph
from rag.index_manager import build_or_load_index
from api.config import settings 

PDFs = ["data/The_GALE_ENCYCLOPEDIA_of_MEDICINE_SECOND.pdf"]

# NOTE:
# Objects below are initialized once at process startup.
# Do NOT import agent_deps inside request handlers.

_embedder = EmbeddingService()

_faiss_store, _bm25_store = build_or_load_index(PDFs)

_dense_retriever = Retriever(_embedder, _faiss_store)

_hybrid_retriever = HybridRetriever(
    dense = _dense_retriever, 
    sparse = _bm25_store,
    k_dense = 5,
    k_sparse= 5
    )

reranker = CrossEncoderReranker()
retriever_runnable = RetrieverRunnable(_hybrid_retriever, reranker)

llm_runnable = LLMRunnable()

rewriter = QueryWriter()

REASONING_GRAPH = build_reasoning_graph(
    rewrite_node = rewriter,
    retriever_node = retriever_runnable,
)
