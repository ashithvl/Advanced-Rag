"""Neo4j property-graph store for relational / multi-hop retrieval."""

from __future__ import annotations

from functools import lru_cache
from typing import Any

from advanced_rag.config import get_settings


@lru_cache(maxsize=1)
def get_neo4j_property_graph_store() -> Any:
    """Return a LlamaIndex Neo4j PropertyGraphStore."""
    from llama_index.graph_stores.neo4j import Neo4jPropertyGraphStore

    settings = get_settings()
    return Neo4jPropertyGraphStore(
        username=settings.neo4j_username,
        password=settings.neo4j_password,
        url=settings.neo4j_uri,
        database=settings.neo4j_database,
    )
