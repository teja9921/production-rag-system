from pydantic_settings import BaseSettings
from pydantic import Field

class Settings(BaseSettings):
    # ===== Secrets (required) =====
    HF_TOKEN: str = Field(..., env="HF_TOKEN")

    # ===== Models =====
    EMBEDDING_MODEL: str = "sentence-transformers/all-MiniLM-L6-v2"
    LLM_MODEL_ID: str = "MiniMaxAI/MiniMax-M2.5" #"meta-llama/Meta-Llama-3-8B-Instruct" #"mistralai/Mistral-7B-Instruct-v0.2" 

    # ===== Chunking =====
    CHUNK_SIZE: int = 500
    CHUNK_OVERLAP: int = 50
    MAX_CHARS: int = 1000
    MIN_CHARS: int = 150

    # ===== Retrieval =====
    SIMILARITY_THRESHOLD: float = 0.45

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