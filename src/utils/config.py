from pydantic_settings import BaseSettings
from pydantic import Field
from typing import Optional

class Settings(BaseSettings):
    openai_api_key: str = Field(..., env="OPENAI_API_KEY")
    chroma_persist_dir: str = Field("./chroma_db", env="CHROMA_PERSIST_DIR")
    embedding_model: str = "text-embedding-3-small"
    llm_model: str = "gpt-4o-mini"
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    
    # Retrieval parameters
    top_k: int = 10
    fusion_alpha: float = 0.7  # Weight for semantic vs BM25
    cluster_n: int = 5
    
    class Config:
        env_file = ".env"

settings = Settings()