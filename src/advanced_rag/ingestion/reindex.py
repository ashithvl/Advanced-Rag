"""Reindex a cataloged PDF: drop its vectors + page images, then re-ingest from disk."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from advanced_rag.indexing.weaviate_delete import delete_by_source_doc_id
from advanced_rag.ingestion.pdf_ingest import ingest_pdf_for_doc
from advanced_rag.ingestion.registry import CatalogEntry, add_document, get_document, now_iso


def reindex_cataloged_document(doc_id: str) -> dict[str, Any]:
    """Raises ``KeyError`` if unknown, ``FileNotFoundError`` if the PDF was deleted."""
    entry = get_document(doc_id)
    if entry is None:
        raise KeyError(doc_id)
    pdf = Path(entry.pdf_path)
    if not pdf.is_file():
        raise FileNotFoundError(str(pdf))

    delete_by_source_doc_id(doc_id)
    pages_dir = Path("data/raw/pages")
    if pages_dir.is_dir():
        for p in pages_dir.glob(f"{doc_id}_p*.png"):
            if p.is_file():
                p.unlink()

    summary = ingest_pdf_for_doc(pdf.resolve(), doc_id, entry.original_filename)
    size_bytes = pdf.stat().st_size
    add_document(
        CatalogEntry(
            doc_id=doc_id,
            original_filename=entry.original_filename,
            size_bytes=size_bytes,
            uploaded_at=now_iso(),
            pdf_path=entry.pdf_path,
            pages_indexed=int(summary.get("pages", 0)),
        )
    )
    return {"doc_id": doc_id, "size_bytes": size_bytes, **summary}
