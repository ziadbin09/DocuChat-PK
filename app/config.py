"""Runtime configuration. Every value can be overridden in `.env` or the environment."""
from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # --- storage ---
    data_dir: Path = Path("data")
    qdrant_url: str | None = None  # e.g. http://qdrant:6333; unset = embedded on-disk Qdrant
    qdrant_api_key: str | None = None
    collection: str = "docuchat"

    # --- ingestion ---
    chunk_chars: int = 1100
    chunk_overlap_chars: int = 180
    max_upload_mb: int = 25

    # --- retrieval ---
    embed_provider: str = "fastembed"  # fastembed | hash (hash = no model download, tests/dev only)
    embed_model: str = "BAAI/bge-small-en-v1.5"
    rerank_enabled: bool = True
    rerank_model: str = "Xenova/ms-marco-MiniLM-L-6-v2"
    dense_candidates: int = 30
    bm25_candidates: int = 30
    rerank_candidates: int = 20
    top_k: int = 6
    rrf_k: int = 60

    # --- generation ---
    llm_provider: str = "anthropic"  # anthropic | openai_compat | extractive (no LLM, demo)
    anthropic_api_key: str | None = None
    anthropic_model: str = "claude-opus-5-5"
    anthropic_rewrite_model: str = "claude-haiku-5-5"
    anthropic_effort: str = "medium"  # low | medium | high | xhigh | max
    anthropic_refusal_fallback: bool = True
    max_output_tokens: int = 64000
    history_turns: int = 6

    # OpenAI-compatible endpoint (NVIDIA NIM, Groq, Together, Ollama, ...)
    openai_base_url: str = "https://integrate.api.nvidia.com/v1"
    openai_api_key: str | None = None
    openai_model: str = "meta/llama-3.3-70b-instruct"


@lru_cache
def get_settings() -> Settings:
    return Settings()
