"""Weaviate Cloud client + one collection with two named vectors (txt_vec + img_vec)."""

from __future__ import annotations

from functools import lru_cache
from typing import Any

import weaviate
from weaviate.auth import AuthApiKey
from weaviate.classes.config import Configure, DataType, Property, VectorDistances
from weaviate.exceptions import UnexpectedStatusCodeError

from advanced_rag.config import get_settings
from advanced_rag.pretty_log import get_logger, kv_lines

_log = get_logger("weaviate")


@lru_cache(maxsize=1)
def get_weaviate_client() -> Any:
    s = get_settings()
    url = (s.weaviate_url or "").strip().rstrip("/")
    if not url:
        raise RuntimeError(
            "WEAVIATE_URL is not set. Use your Weaviate Cloud cluster URL "
            "(https://console.weaviate.cloud/)."
        )
    if not s.weaviate_api_key:
        raise RuntimeError("WEAVIATE_API_KEY is required.")
    _log.info(
        "Connecting to Weaviate Cloud\n%s",
        kv_lines([("cluster_url", url), ("collection", s.weaviate_collection)]),
    )
    return weaviate.connect_to_weaviate_cloud(
        cluster_url=url,
        auth_credentials=AuthApiKey(s.weaviate_api_key),
    )


def _create_collection(client: Any, name: str) -> Any:
    """One collection, two named vectors (bring-your-own vectors)."""
    _log.info(
        "Creating collection %r  named vectors txt_vec + img_vec  (self_provided)",
        name,
    )
    client.collections.create(
        name=name,
        vector_config=[
            Configure.Vectors.self_provided(
                name="txt_vec",
                vector_index_config=Configure.VectorIndex.hnsw(
                    distance_metric=VectorDistances.COSINE,
                ),
            ),
            Configure.Vectors.self_provided(
                name="img_vec",
                vector_index_config=Configure.VectorIndex.hnsw(
                    distance_metric=VectorDistances.COSINE,
                ),
            ),
        ],
        properties=[
            Property(name="source_doc_id", data_type=DataType.TEXT),
            Property(name="source_filename", data_type=DataType.TEXT),
            Property(name="page_number", data_type=DataType.INT),
            Property(name="page_image_path", data_type=DataType.TEXT),
            Property(name="is_visual", data_type=DataType.BOOL),
            Property(name="title", data_type=DataType.TEXT),
            Property(name="super_text", data_type=DataType.TEXT),  # BM25 + txt_vec source
            Property(name="visual_text", data_type=DataType.TEXT),
            Property(name="ingested_at", data_type=DataType.TEXT),
        ],
    )
    _log.info("Collection %r created successfully.", name)
    return client.collections.get(name)


def get_collection() -> Any:
    """Return the page collection, creating it lazily on first use."""
    s = get_settings()
    client = get_weaviate_client()
    name = s.weaviate_collection
    try:
        if client.collections.exists(name):
            _log.info("Using existing collection %r", name)
            return client.collections.get(name)
    except UnexpectedStatusCodeError:
        _log.warning("exists() check failed; will try to create collection %r", name)
    return _create_collection(client, name)
