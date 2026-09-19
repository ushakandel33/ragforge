"""
Central configuration. All values are overridable via environment variables
or a `.env` file (see .env.example).
"""
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # Primary provider (Gemini - free tier)
    gemini_api_key: str = ""
    gemini_model: str = "gemini-2.0-flash"

    # Fallback provider (Groq - free tier, OpenAI-compatible)
    groq_api_key: str = ""
    groq_model: str = "llama-3.1-8b-instant"

    # Optional third-tier: locally served open-source model via vLLM
    local_model_enabled: bool = False
    local_model_base_url: str = "http://localhost:8001/v1"
    local_model_name: str = "mistralai/Mistral-7B-Instruct-v0.2"

    # Generation parameters
    app_temperature: float = 0.4
    app_top_p: float = 0.9

    # Reliability
    rate_limit_per_minute: int = 20
    cache_ttl_seconds: int = 3600
    max_retries: int = 3

    # Agentic RAG
    max_agent_steps: int = 5
    agent_evidence_limit: int = 6
    agent_history_limit: int = 8

    # RAG
    vector_db_path: str = "./data/chroma_db"
    chunk_size: int = 500
    chunk_overlap: int = 50
    embedding_model: str = "all-MiniLM-L6-v2"
    top_k_retrieval: int = 4


settings = Settings()
