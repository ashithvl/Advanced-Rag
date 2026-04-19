"""Hierarchical + semantic chunking for small-to-big retrieval.

Produces parent (large) and child (atomic) nodes with explicit relationships
so retrievers can fetch atomic chunks and inject parent context at generation time.
"""

from __future__ import annotations

from collections.abc import Sequence

from llama_index.core import Document
from llama_index.core.node_parser import HierarchicalNodeParser
from llama_index.core.schema import BaseNode

from advanced_rag.config import get_settings


def build_hierarchical_nodes(documents: Sequence[Document]) -> list[BaseNode]:
    """Split documents into a hierarchy of parent / child nodes.

    Chunk sizes are configured in `Settings` (default 2048 / 512 / 128 tokens).
    """
    settings = get_settings()
    parser = HierarchicalNodeParser.from_defaults(
        chunk_sizes=[
            settings.parent_chunk_size,
            settings.chunk_size,
            max(128, settings.chunk_size // 4),
        ],
        chunk_overlap=settings.chunk_overlap,
    )
    return parser.get_nodes_from_documents(list(documents))
