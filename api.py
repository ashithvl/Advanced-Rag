"""FastAPI surface for the Advanced RAG pipeline."""

from __future__ import annotations

import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel

from advanced_rag.config import get_settings
from advanced_rag.ingestion.pdf_ingest import ingest_pdf_for_doc
from advanced_rag.ingestion.registry import (
    CatalogEntry,
    add_document,
    list_documents,
    now_iso,
    remove_document,
)
from advanced_rag.ingestion.reindex import reindex_cataloged_document
from advanced_rag.pretty_log import get_logger, setup_logging

PDF_DIR = Path("data/raw/pdfs")
_api_log = get_logger("api")


@asynccontextmanager
async def lifespan(app: FastAPI):
    setup_logging(get_settings().log_level)
    PDF_DIR.mkdir(parents=True, exist_ok=True)
    _api_log.info("API startup  log_level=%s", get_settings().log_level)
    yield


app = FastAPI(title="Advanced RAG API", version="0.1.0", lifespan=lifespan)


class QueryRequest(BaseModel):
    question: str


class QueryResponse(BaseModel):
    answer: str
    confidence: float
    citations: list[dict] = []


class DocumentInfo(BaseModel):
    doc_id: str
    original_filename: str
    size_bytes: int
    uploaded_at: str
    pdf_path: str
    pages_indexed: int


_UPLOAD_PDF = File(...)


@app.post("/ingest")
async def ingest_pdf_only(file: UploadFile = _UPLOAD_PDF) -> dict[str, str | int]:
    """Ingest one PDF (classifier + structured vision on visual pages + dual vectors)."""
    s = get_settings()
    max_bytes = s.max_pdf_upload_mb * 1024 * 1024
    name = (file.filename or "").strip()
    if not name.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only .pdf files are accepted.")
    ct = (file.content_type or "").lower()
    if ct and "pdf" not in ct and ct != "application/octet-stream":
        raise HTTPException(status_code=400, detail="File must be a PDF (application/pdf).")

    chunks: list[bytes] = []
    total = 0
    while True:
        block = await file.read(8 * 1024 * 1024)
        if not block:
            break
        total += len(block)
        if total > max_bytes:
            raise HTTPException(
                status_code=413,
                detail=f"PDF exceeds maximum size of {s.max_pdf_upload_mb} MB.",
            )
        chunks.append(block)

    raw = b"".join(chunks)
    if not raw:
        raise HTTPException(status_code=400, detail="Empty file.")

    doc_id = str(uuid.uuid4())
    dest = PDF_DIR / f"{doc_id}.pdf"
    dest.write_bytes(raw)
    original = Path(name).name
    _api_log.info(
        "POST /ingest  filename=%r  doc_id=%s  size_mb=%.2f",
        original,
        doc_id,
        len(raw) / (1024 * 1024),
    )

    try:
        summary = ingest_pdf_for_doc(dest, doc_id, original)
    except Exception as exc:
        dest.unlink(missing_ok=True)
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    try:
        pdf_rel = str(dest.relative_to(Path.cwd()))
    except ValueError:
        pdf_rel = dest.as_posix()

    add_document(
        CatalogEntry(
            doc_id=doc_id,
            original_filename=original,
            size_bytes=len(raw),
            uploaded_at=now_iso(),
            pdf_path=pdf_rel,
            pages_indexed=int(summary.get("pages", 0)),
        )
    )
    _api_log.info("POST /ingest done  %s", summary)
    return {
        "doc_id": doc_id,
        "original_filename": original,
        "size_bytes": len(raw),
        **summary,
    }


@app.get("/documents", response_model=list[DocumentInfo])
def list_uploaded_documents() -> list[DocumentInfo]:
    return [
        DocumentInfo(
            doc_id=e.doc_id,
            original_filename=e.original_filename,
            size_bytes=e.size_bytes,
            uploaded_at=e.uploaded_at,
            pdf_path=e.pdf_path,
            pages_indexed=e.pages_indexed,
        )
        for e in list_documents()
    ]


@app.delete("/documents/{doc_id}")
def delete_document(doc_id: str) -> dict[str, str]:
    if remove_document(doc_id) is None:
        raise HTTPException(status_code=404, detail="Unknown doc_id")
    return {"status": "deleted", "doc_id": doc_id}


@app.post("/documents/{doc_id}/reindex")
def reindex_document(doc_id: str) -> dict[str, Any]:
    """Drop vectors for ``doc_id`` and re-run ingestion from the saved PDF."""
    try:
        return reindex_cataloged_document(doc_id)
    except KeyError:
        raise HTTPException(status_code=404, detail="Unknown doc_id") from None
    except FileNotFoundError:
        raise HTTPException(
            status_code=404,
            detail="PDF file is no longer on disk. Remove the stale catalog entry or upload again.",
        ) from None
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/query", response_model=QueryResponse)
def query(req: QueryRequest) -> QueryResponse:
    from advanced_rag.pipeline import run

    _api_log.info("POST /query  chars=%d", len(req.question))
    try:
        state = run(req.question)
    except Exception as exc:
        _api_log.exception("POST /query failed")
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    s = get_settings()
    contexts = state.get("contexts", [])
    min_score = s.citation_min_score

    shown = [c for c in contexts if float(c.get("score", 0.0)) >= min_score]

    _api_log.info(
        "POST /query done  confidence=%.4f  contexts=%d  citations_shown=%d  (min_score=%.2f)",
        float(state.get("confidence", 0.0)),
        len(contexts),
        len(shown),
        min_score,
    )
    return QueryResponse(
        answer=state.get("answer", ""),
        confidence=float(state.get("confidence", 0.0)),
        citations=[
            {
                "id": c["id"],
                "score": c["score"],
                "source_filename": c["source_filename"],
                "page_number": c["page_number"],
                "page_image_path": c["page_image_path"],
                "is_visual": c["is_visual"],
                "title": c.get("title", ""),
            }
            for c in shown
        ],
    )
