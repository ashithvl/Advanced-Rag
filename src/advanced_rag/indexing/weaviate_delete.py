"""Delete all rows tied to a given ``source_doc_id`` from the page collection."""

from __future__ import annotations

from advanced_rag.config import get_settings
from advanced_rag.indexing.vector_store import get_weaviate_client
from advanced_rag.pretty_log import get_logger

_log = get_logger("weaviate")


def delete_by_source_doc_id(doc_id: str) -> None:
    try:
        from weaviate.classes.query import Filter
    except ImportError:
        return

    s = get_settings()
    if not (s.weaviate_url or "").strip():
        return
    try:
        col = get_weaviate_client().collections.get(s.weaviate_collection)
        col.data.delete_many(where=Filter.by_property("source_doc_id").equal(doc_id))
        _log.info("Deleted all objects for source_doc_id=%r in %r", doc_id, s.weaviate_collection)
    except Exception as exc:
        _log.warning("delete_many skipped or failed for doc_id=%r: %s", doc_id, exc)
