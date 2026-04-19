"""FastAPI surface for the Advanced RAG pipeline."""

from __future__ import annotations

import shutil
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel

from advanced_rag.config import get_settings
from advanced_rag.observability import enable_langsmith

app = FastAPI(title="Advanced RAG API", version="0.1.0")
enable_langsmith()

UPLOAD_DIR = Path("data/raw")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)


class QueryRequest(BaseModel):
    question: str


class QueryResponse(BaseModel):
    answer: str
    confidence: float
    faithful: bool | None = None
    unsupported: list[str] = []
    citations: list[dict] = []


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "embedding_provider": get_settings().embedding_provider}


_UPLOAD_FILE = File(...)


@app.post("/ingest")
async def ingest(file: UploadFile = _UPLOAD_FILE) -> dict[str, int | str]:
    from advanced_rag.ingestion import ingest_path

    dest = UPLOAD_DIR / file.filename
    with dest.open("wb") as f:
        shutil.copyfileobj(file.file, f)
    try:
        summary = ingest_path(dest)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    return {"file": file.filename, **summary}


@app.post("/query", response_model=QueryResponse)
def query(req: QueryRequest) -> QueryResponse:
    from advanced_rag.pipeline import run

    try:
        state = run(req.question)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return QueryResponse(
        answer=state.get("answer", ""),
        confidence=float(state.get("confidence", 0.0)),
        faithful=state.get("faithful"),
        unsupported=list(state.get("unsupported", [])),
        citations=[
            {
                "id": c["id"],
                "score": c["score"],
                "metadata": c["metadata"],
            }
            for c in state.get("contexts", [])
        ],
    )
