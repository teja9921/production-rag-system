from ingestion.loader import load_pdf
from ingestion.splitter import SimpleChunker
from rag.embedder import EmbeddingService
from rag.faiss_store import FaissStore
from rag.retriever import Retriever
from orchestration.lc_retriever import RetrieverRunnable
from orchestration.lc_llm import LLMRunnable
from orchestration.memory import memory_reader, memory_writer
from orchestration.rewrite import QueryWriter
from orchestration.agent_graph import build_agentic_graph
from api.config import settings 

PDF_FILE = "data/LLM_Interview_Questions.pdf"
INDEX_PATH = "data/faiss.index"
META_PATH = "data/meta.pkl"

_pages = load_pdf(PDF_FILE)

_chunker = SimpleChunker(
    chunk_size=settings.CHUNK_SIZE, 
    chunk_overlap=settings.CHUNK_OVERLAP
    )
_chunks = _chunker.split_pages(_pages)

_embedder = EmbeddingService()
_embeddings = _embedder.embed_texts(
    [c["content"]for c in _chunks]
    ).cpu().numpy()

_store = FaissStore(INDEX_PATH, META_PATH)
_store.add_chunks(_embeddings, _chunks)

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

