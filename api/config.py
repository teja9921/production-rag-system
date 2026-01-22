# api/config.py

from pydantic_settings import BaseSettings
from pydantic import Field

class Settings(BaseSettings):
    # ===== Secrets (required) =====
    HF_TOKEN: str = Field(..., env="HF_TOKEN")

    # ===== Models =====
    EMBEDDING_MODEL: str = "sentence-transformers/all-MiniLM-L6-v2"
    LLM_MODEL_ID: str = "meta-llama/Meta-Llama-3-8B-Instruct"

    # ===== Chunking =====
    CHUNK_SIZE: int = 500
    CHUNK_OVERLAP: int = 50

    # ===== Retrieval =====
    RETRIEVAL_K: int = 3
    SIMILARITY_THRESHOLD: float = 0.45

    # ===== Limits =====
    MAX_PROMPT_TOKENS: int = 3000
    LLM_TIMEOUT_SECONDS: int = 15

    class Config:
        env_file = None  # dotenv handled outside
        case_sensitive = True


settings = Settings()
