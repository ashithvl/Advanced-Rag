"""Local catalog of uploaded PDFs (paths + metadata) for list/delete/reindex UX."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

REGISTRY_PATH = Path("data/registry.json")


@dataclass
class CatalogEntry:
    doc_id: str
    original_filename: str
    size_bytes: int
    uploaded_at: str
    pdf_path: str  # relative to cwd, e.g. data/raw/pdfs/<doc_id>.pdf
    pages_indexed: int


def _ensure_parent() -> None:
    REGISTRY_PATH.parent.mkdir(parents=True, exist_ok=True)


def _load_raw() -> dict[str, Any]:
    if not REGISTRY_PATH.is_file():
        return {"documents": []}
    return json.loads(REGISTRY_PATH.read_text(encoding="utf-8"))


def _save_raw(data: dict[str, Any]) -> None:
    _ensure_parent()
    REGISTRY_PATH.write_text(json.dumps(data, indent=2), encoding="utf-8")


def _row_to_entry(row: dict[str, Any]) -> CatalogEntry:
    # Backward-compat: previous schema used `images_extracted`.
    if "pages_indexed" not in row and "images_extracted" in row:
        row = {**row, "pages_indexed": row.pop("images_extracted")}
    return CatalogEntry(**row)


def list_documents() -> list[CatalogEntry]:
    rows = [_row_to_entry(r) for r in _load_raw().get("documents", [])]
    rows.sort(key=lambda e: e.uploaded_at, reverse=True)
    return rows


def get_document(doc_id: str) -> CatalogEntry | None:
    for row in _load_raw().get("documents", []):
        if row["doc_id"] == doc_id:
            return _row_to_entry(row)
    return None


def add_document(entry: CatalogEntry) -> None:
    data = _load_raw()
    docs = data.setdefault("documents", [])
    docs[:] = [d for d in docs if d["doc_id"] != entry.doc_id]
    docs.append(asdict(entry))
    _save_raw(data)


def remove_document(doc_id: str) -> CatalogEntry | None:
    """Drop Weaviate rows, registry entry, the PDF and the rendered page images."""
    entry = get_document(doc_id)
    if entry is None:
        return None

    try:
        from advanced_rag.indexing.weaviate_delete import delete_by_source_doc_id

        delete_by_source_doc_id(doc_id)
    except Exception:
        pass

    data = _load_raw()
    data["documents"] = [d for d in data.get("documents", []) if d["doc_id"] != doc_id]
    _save_raw(data)

    pdf = Path(entry.pdf_path)
    if pdf.is_file():
        pdf.unlink()

    pages_dir = Path("data/raw/pages")
    if pages_dir.is_dir():
        for p in pages_dir.glob(f"{doc_id}_p*.png"):
            if p.is_file():
                p.unlink()
    return entry


def now_iso() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat()
