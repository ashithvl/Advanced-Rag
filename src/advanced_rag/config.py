"""Centralized runtime configuration loaded from environment / .env."""

from __future__ import annotations

from functools import lru_cache
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

EmbeddingProvider = Literal["voyage", "bge", "openai"]
RerankerProvider = Literal["cohere", "bge"]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # --- LLM ---
    openai_api_key: str | None = None
    llm_model: str = "gpt-4o"
    llm_temperature: float = 0.0

    # --- Text embeddings ---
    embedding_provider: EmbeddingProvider = "voyage"
    voyage_api_key: str | None = None
    voyage_text_model: str = "voyage-3-large"
    voyage_code_model: str = "voyage-code-3"
    bge_model: str = "BAAI/bge-m3"
    openai_embedding_model: str = "text-embedding-3-large"

    # --- Vision / multimodal embeddings ---
    enable_vision: bool = True
    colpali_model: str = "vidore/colpali-v1.3"

    # --- Reranker ---
    reranker_provider: RerankerProvider = "bge"
    cohere_api_key: str | None = None
    cohere_rerank_model: str = "rerank-v3.5"
    bge_reranker_model: str = "BAAI/bge-reranker-v2-m3"

    # --- Parsing ---
    llama_cloud_api_key: str | None = None
    llamaparse_result_type: Literal["markdown", "text"] = "markdown"

    # --- Vector store: Weaviate ---
    weaviate_url: str = "http://localhost:8080"
    weaviate_api_key: str | None = None
    weaviate_grpc_port: int = 50051
    weaviate_text_collection: str = "DocChunks"
    weaviate_vision_collection: str = "DocPages"

    # --- Graph store: Neo4j ---
    neo4j_uri: str = "bolt://localhost:7687"
    neo4j_username: str = "neo4j"
    neo4j_password: str = "neo4j_password"
    neo4j_database: str = "neo4j"

    # --- Chunking ---
    chunk_size: int = 512
    chunk_overlap: int = 64
    parent_chunk_size: int = 2048

    # --- Retrieval ---
    top_k_dense: int = 20
    top_k_sparse: int = 20
    top_k_rerank: int = 6
    confidence_threshold: float = 0.30

    # --- Observability ---
    langsmith_api_key: str | None = None
    langsmith_project: str = "advanced-rag"
    langchain_tracing_v2: bool = Field(default=False, alias="LANGCHAIN_TRACING_V2")


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
