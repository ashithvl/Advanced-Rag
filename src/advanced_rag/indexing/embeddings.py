"""Text embedding factory.

Selects between Voyage AI, BGE-M3 (local), and OpenAI based on settings.
"""

from __future__ import annotations

from typing import Any

from advanced_rag.config import EmbeddingProvider, get_settings


def get_text_embedder(provider: EmbeddingProvider | None = None) -> Any:
    """Return a LlamaIndex-compatible BaseEmbedding instance."""
    settings = get_settings()
    provider = provider or settings.embedding_provider

    if provider == "voyage":
        from llama_index.embeddings.voyageai import VoyageEmbedding

        if not settings.voyage_api_key:
            raise RuntimeError("VOYAGE_API_KEY is required for Voyage embeddings.")
        return VoyageEmbedding(
            model_name=settings.voyage_text_model,
            voyage_api_key=settings.voyage_api_key,
        )

    if provider == "bge":
        from llama_index.embeddings.huggingface import HuggingFaceEmbedding

        return HuggingFaceEmbedding(model_name=settings.bge_model)

    if provider == "openai":
        from llama_index.embeddings.openai import OpenAIEmbedding

        if not settings.openai_api_key:
            raise RuntimeError("OPENAI_API_KEY is required for OpenAI embeddings.")
        return OpenAIEmbedding(
            model=settings.openai_embedding_model,
            api_key=settings.openai_api_key,
        )

    raise ValueError(f"Unknown embedding provider: {provider}")
