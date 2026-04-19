"""Reranker factory: BGE-reranker-v2 (local) or Cohere Rerank (API)."""

from __future__ import annotations

from typing import Any

from advanced_rag.config import RerankerProvider, get_settings


def get_reranker(provider: RerankerProvider | None = None, top_n: int | None = None) -> Any:
    """Return a LlamaIndex postprocessor that reranks NodeWithScore items."""
    settings = get_settings()
    provider = provider or settings.reranker_provider
    top_n = top_n or settings.top_k_rerank

    if provider == "bge":
        from llama_index.core.postprocessor import SentenceTransformerRerank

        return SentenceTransformerRerank(
            model=settings.bge_reranker_model,
            top_n=top_n,
        )

    if provider == "cohere":
        from llama_index.postprocessor.cohere_rerank import CohereRerank

        if not settings.cohere_api_key:
            raise RuntimeError("COHERE_API_KEY is required for Cohere Rerank.")
        return CohereRerank(
            api_key=settings.cohere_api_key,
            model=settings.cohere_rerank_model,
            top_n=top_n,
        )

    raise ValueError(f"Unknown reranker provider: {provider}")
