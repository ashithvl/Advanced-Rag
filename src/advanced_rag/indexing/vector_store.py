"""Weaviate client + vector store helpers."""

from __future__ import annotations

from functools import lru_cache
from typing import Any

from advanced_rag.config import get_settings


@lru_cache(maxsize=1)
def get_weaviate_client() -> Any:
    """Connect to a Weaviate instance (local docker-compose by default)."""
    import weaviate

    settings = get_settings()
    if settings.weaviate_url.startswith("http://localhost") or settings.weaviate_url.startswith(
        "http://127.0.0.1"
    ):
        return weaviate.connect_to_local(
            host=settings.weaviate_url.split("://", 1)[1].split(":")[0],
            port=int(settings.weaviate_url.rsplit(":", 1)[1]),
            grpc_port=settings.weaviate_grpc_port,
        )

    headers: dict[str, str] | None = None
    if settings.openai_api_key:
        headers = {"X-OpenAI-Api-Key": settings.openai_api_key}
    return weaviate.connect_to_wcs(
        cluster_url=settings.weaviate_url,
        auth_credentials=weaviate.auth.AuthApiKey(settings.weaviate_api_key)
        if settings.weaviate_api_key
        else None,
        headers=headers,
    )


def get_weaviate_vector_store(collection: str | None = None) -> Any:
    """Return a LlamaIndex WeaviateVectorStore bound to a collection."""
    from llama_index.vector_stores.weaviate import WeaviateVectorStore

    settings = get_settings()
    return WeaviateVectorStore(
        weaviate_client=get_weaviate_client(),
        index_name=collection or settings.weaviate_text_collection,
    )
