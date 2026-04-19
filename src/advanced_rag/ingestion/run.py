"""End-to-end ingestion: parse -> chunk -> embed -> index (text + optional vision)."""

from __future__ import annotations

from pathlib import Path

from llama_index.core import StorageContext, VectorStoreIndex
from llama_index.core.node_parser import get_leaf_nodes

from advanced_rag.config import get_settings
from advanced_rag.indexing.embeddings import get_text_embedder
from advanced_rag.indexing.vector_store import get_weaviate_vector_store

from .chunking import build_hierarchical_nodes
from .parser import parse_document


def ingest_path(path: str | Path, *, collection: str | None = None) -> dict[str, int]:
    """Parse a document or directory and write chunks into Weaviate.

    Returns a small summary of what was indexed.
    """
    settings = get_settings()
    p = Path(path)
    files = [p] if p.is_file() else sorted(p.rglob("*"))
    files = [f for f in files if f.is_file()]

    documents = []
    for f in files:
        documents.extend(parse_document(f))

    nodes = build_hierarchical_nodes(documents)
    leaf_nodes = get_leaf_nodes(nodes)

    vector_store = get_weaviate_vector_store(collection or settings.weaviate_text_collection)
    storage = StorageContext.from_defaults(vector_store=vector_store)
    storage.docstore.add_documents(nodes)

    VectorStoreIndex(
        nodes=leaf_nodes,
        storage_context=storage,
        embed_model=get_text_embedder(),
        show_progress=True,
    )
    return {"files": len(files), "nodes": len(nodes), "leaves": len(leaf_nodes)}
