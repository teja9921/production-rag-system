from rag.embedder import EmbeddingService
from rag.retriever import Retriever
from orchestration.lc_retriever import RetrieverRunnable
from orchestration.lc_llm import LLMRunnable
from orchestration.memory import memory_reader, memory_writer
from orchestration.rewrite import QueryWriter
from orchestration.agent_graph import build_agentic_graph
from orchestration.retrieval_graph import build_retrieval_graph
from rag.index_manager import build_or_load_index
from api.config import settings 

PDFs = ["data/The_GALE_ENCYCLOPEDIA_of_MEDICINE_SECOND.pdf"]

_embedder = EmbeddingService()
_store = build_or_load_index(PDFs)

_retriever = Retriever(_embedder, _store)
retriever_runnable = RetrieverRunnable(_retriever)

llm_runnable = LLMRunnable()

rewriter = QueryWriter()

GRAPH = build_agentic_graph(
    memory_node = memory_reader,
    rewrite_node = rewriter,
    retriever_node = retriever_runnable,
    llm_node = llm_runnable,
    memory_write_node = memory_writer
)

RETRIEVAL_GRAPH = build_retrieval_graph(
    memory_node = memory_reader,
    rewrite_node= rewriter,
    retriever_node= retriever_runnable
)