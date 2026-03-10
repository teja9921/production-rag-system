from prometheus_client import Counter, Histogram, Gauge

# -----------------------------
# Request Metrics
# -----------------------------

REQUEST_COUNT = Counter(
    "rag_requests_total",
    "Total number of RAG requests",
    ["endpoint"]
)

REQUEST_LATENCY = Histogram(
    "rag_request_latency_seconds",
    "Latency of RAG requests",
    ["endpoint"]
)

# -----------------------------
# Retrieval Metrics
# -----------------------------

RETRIEVAL_LATENCY = Histogram(
    "rag_retrieval_latency_seconds",
    "Latency of retrieval step"
)

# -----------------------------
# Generation Metrics
# -----------------------------

GENERATION_LATENCY = Histogram(
    "rag_generation_latency_seconds",
    "Latency of LLM generation"
)

# -----------------------------
# Quality Metrics
# -----------------------------

NO_ANSWER_COUNT = Counter(
    "rag_no_answer_total",
    "Total NO_ANSWER responses"
)

# -----------------------------
# Failure Metrics
# -----------------------------

MODEL_FAILURE_COUNT = Counter(
    "rag_model_failure_total",
    "LLM model failures"
)

TIMEOUT_COUNT = Counter(
    "rag_timeout_total",
    "LLM timeout occurrences"
)

# -----------------------------
# Tokens Count
# -----------------------------
TOKEN_USAGE = Counter(
    "rag_tokens_used_total",
    "Total tokens used"
)

FIRST_TOKEN_LATENCY = Histogram(
    "rag_first_token_latency_seconds",
    "Time to first token for streaming responses"
)

FAILURE_SIMILARITY_COUNT = Counter(
    "rag_queries_similar_to_eval_failures_total",
    "Queries similar to known evaluation failures"
)

RETRIEVAL_TOP1_SIMILARITY = Histogram(
    "rag_top1_similarity",
    "Top1 retrieval similarity score"
)
