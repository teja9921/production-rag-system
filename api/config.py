from pydantic_settings import BaseSettings
from pydantic import Field

class Settings(BaseSettings):
    # ===== Secrets (required) =====

    HF_TOKEN: str = Field(..., env="HF_TOKEN")
    LANGSMITH_TRACING: bool | None = Field(False, env="LANGSMITH_TRACING")
    LANGSMITH_ENDPOINT: str | None = Field(None, env="LANGSMITH_ENDPOINT")
    LANGSMITH_API_KEY: str | None = Field(None, env="LANGSMITH_API_KEY")
    LANGSMITH_PROJECT: str | None = Field(None, env="LANGSMITH_PROJECT")

    # ===== Models =====
    EMBEDDING_MODEL: str = "sentence-transformers/all-MiniLM-L6-v2"
    LLM_MODEL_ID: str = "Qwen/Qwen3-Coder-480B-A35B-Instruct"

    # ===== Chunking =====
    CHUNK_SIZE: int = 500
    CHUNK_OVERLAP: int = 50
    MAX_CHARS: int = 1000
    MIN_CHARS: int = 150

    # ===== Retrieval =====
    SIMILARITY_THRESHOLD: float = 0.45
    RERANK_SCORE_THRESHOLD: float = 0.2
    FAILURE_SIMILARITY_THRESHOLD: float = 0.8
    DATASET_VERSION: str = "gale_v1"
    CONFIG_ID: str = "hybrid_conditional_minilm"

    # ===== Limits =====
    MAX_PROMPT_TOKENS: int = 3000
    LLM_TIMEOUT_SECONDS: int = 15

    # ===== Device =======
    EMBEDDING_DEVICE: str = "cpu"  # allowed: "cpu", "cuda"

    # ===== Database URL ======
    DATABASE_URL: str = "sqlite:///./rag_app.db"

    # ===== CHAT HISTORY ======
    MAX_HISTORY_MESSAGES: int = 6
    class Config:
        env_file = ".env"
        case_sensitive = True

settings = Settings()
