"""Hybrid (dense + BM25) retriever with small-to-big parent injection."""

from __future__ import annotations

from typing import Any

from llama_index.core import VectorStoreIndex
from llama_index.core.retrievers import QueryFusionRetriever
from llama_index.core.retrievers.fusion_retriever import FUSION_MODES

from advanced_rag.config import get_settings
from advanced_rag.indexing.embeddings import get_text_embedder
from advanced_rag.indexing.vector_store import get_weaviate_vector_store


def build_hybrid_retriever(collection: str | None = None) -> Any:
    """Dense (Weaviate) + sparse (BM25) fusion retriever."""
    from llama_index.core.retrievers import BM25Retriever

    settings = get_settings()
    vector_store = get_weaviate_vector_store(collection)
    index = VectorStoreIndex.from_vector_store(
        vector_store=vector_store,
        embed_model=get_text_embedder(),
    )

    dense = index.as_retriever(similarity_top_k=settings.top_k_dense)
    sparse = BM25Retriever.from_defaults(
        docstore=index.docstore,
        similarity_top_k=settings.top_k_sparse,
    )

    return QueryFusionRetriever(
        retrievers=[dense, sparse],
        similarity_top_k=settings.top_k_dense,
        num_queries=1,
        mode=FUSION_MODES.RECIPROCAL_RANK,
        use_async=False,
    )
