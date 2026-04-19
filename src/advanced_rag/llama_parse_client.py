"""LlamaIndex `LlamaParse` (via `llama-cloud-services`) for structured PDF extraction.

LlamaParse is the one place LlamaIndex genuinely beats a hand-rolled approach
for technical PDFs: it natively handles complex layout, tables, figure
captions, equations, and multi-column flow, and returns clean per-page
markdown without us having to send each page to a vision model ourselves.

We keep the call surface tiny and *optional*: if `LLAMA_CLOUD_API_KEY` is
unset, `parse_pages` returns `None` and the ingest path falls back to the
legacy PyMuPDF + GPT-4o-mini vision JSON flow.

Returned shape on success:
    list[dict]  # one entry per PDF page, in source order
        {
            "page_number": int,        # 1-based
            "markdown": str,           # primary content
            "title": str,              # best-effort first heading on the page
        }
"""

from __future__ import annotations

import re
from functools import lru_cache
from pathlib import Path
from typing import Any

from advanced_rag.config import get_settings
from advanced_rag.pretty_log import get_logger, kv_lines

_log = get_logger("llamaparse")

_HEADING_RE = re.compile(r"^\s{0,3}#{1,6}\s+(.+?)\s*$", re.MULTILINE)


@lru_cache(maxsize=1)
def _parser() -> Any | None:
    s = get_settings()
    if not s.llama_cloud_api_key:
        return None
    try:
        from llama_cloud_services import LlamaParse  # noqa: PLC0415  - optional import
    except ImportError as exc:
        _log.warning("llama-cloud-services not installed (%s)", exc)
        return None

    return LlamaParse(
        api_key=s.llama_cloud_api_key,
        result_type=s.llama_parse_result_type,
        premium_mode=s.llama_parse_premium,
        verbose=False,
    )


def is_enabled() -> bool:
    return _parser() is not None


def _extract_title(markdown: str) -> str:
    """Best-effort: first markdown heading on the page, else empty."""
    if not markdown:
        return ""
    m = _HEADING_RE.search(markdown)
    return (m.group(1).strip() if m else "")[:200]


def parse_pages(pdf_path: Path) -> list[dict[str, Any]] | None:
    """Parse a PDF with LlamaParse, returning per-page dicts (or None if disabled)."""
    parser = _parser()
    if parser is None:
        return None

    s = get_settings()
    _log.info(
        "LlamaParse start\n%s",
        kv_lines(
            [
                ("path", str(pdf_path)),
                ("result_type", s.llama_parse_result_type),
                ("premium", s.llama_parse_premium),
            ]
        ),
    )

    try:
        documents = parser.load_data(str(pdf_path))
    except Exception as exc:  # noqa: BLE001  - LlamaParse can raise many things
        _log.warning("LlamaParse failed (%s) — falling back to PyMuPDF + GPT-4o-mini", exc)
        return None

    pages: list[dict[str, Any]] = []
    for idx, doc in enumerate(documents):
        md = (getattr(doc, "text", "") or "").strip()
        meta = getattr(doc, "metadata", {}) or {}
        page_number = int(meta.get("page_number") or meta.get("page") or (idx + 1))
        pages.append(
            {
                "page_number": page_number,
                "markdown": md,
                "title": _extract_title(md),
            }
        )

    _log.info(
        "LlamaParse done\n%s",
        kv_lines(
            [
                ("pages_returned", len(pages)),
                ("first_page_chars", len(pages[0]["markdown"]) if pages else 0),
            ]
        ),
    )
    return pages
