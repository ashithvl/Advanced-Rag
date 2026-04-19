"""Cohere rerank via `langchain_cohere.CohereRerank`.

We wrap our candidate dicts into `langchain_core.documents.Document` instances
so the LangChain compressor can score them, then unpack the results (with
`relevance_score` preserved on `metadata`) back into plain dicts. This keeps
the public signature identical to the direct-Cohere version.
"""

from __future__ import annotations

import time
from functools import lru_cache
from typing import Any

from langchain_cohere import CohereRerank
from langchain_core.documents import Document

from advanced_rag.config import get_settings
from advanced_rag.pretty_log import get_logger, kv_lines

_log = get_logger("rerank")


@lru_cache(maxsize=1)
def _compressor() -> CohereRerank:
    s = get_settings()
    if not s.cohere_api_key:
        raise RuntimeError("COHERE_API_KEY is required for reranking.")
    return CohereRerank(
        model=s.cohere_rerank_model,
        cohere_api_key=s.cohere_api_key,
        top_n=100,  # real top_n is enforced below by slicing after the LC call
    )


def cohere_rerank(query: str, docs: list[dict[str, Any]], *, top_n: int) -> list[dict[str, Any]]:
    """`docs` items must each have a `"text"` field; returns top-N with a `score` added."""
    if not docs:
        _log.info("Cohere rerank skipped (empty candidate list).")
        return []

    s = get_settings()
    t0 = time.perf_counter()

    lc_docs = [
        Document(page_content=d["text"], metadata={"_index": i})
        for i, d in enumerate(docs)
    ]
    compressor = _compressor()
    compressor.top_n = min(top_n, len(docs))

    reranked = list(compressor.compress_documents(lc_docs, query))

    out: list[dict[str, Any]] = []
    rows: list[tuple[str, Any]] = []
    for rank, rd in enumerate(reranked, start=1):
        meta = rd.metadata or {}
        src_idx = int(meta.get("_index", 0))
        d = dict(docs[src_idx])
        d["score"] = float(meta.get("relevance_score", 0.0))
        out.append(d)
        rows.append(
            (
                f"#{rank}",
                f"p.{d.get('page_number')} {d.get('source_filename', '')[:40]} "
                f"score={d['score']:.3f}",
            )
        )

    ms = (time.perf_counter() - t0) * 1000
    _log.info(
        "Cohere rerank (LangChain)  model=%s  in=%d  out=%d  ms=%.0f\n%s",
        s.cohere_rerank_model,
        len(docs),
        len(out),
        ms,
        kv_lines(rows, max_str=200, max_json=400),
    )
    return out
