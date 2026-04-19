"""Document parsing via LlamaParse.

Produces LLM-ready Markdown (preserving headings, tables as Markdown,
figure captions, layout) plus extracted images.
"""

from __future__ import annotations

from pathlib import Path

from llama_index.core import Document

from advanced_rag.config import get_settings


def parse_document(path: str | Path) -> list[Document]:
    """Parse a single document (PDF / image / office doc) into LlamaIndex Documents.

    LlamaParse runs an agentic OCR + layout pass and returns one Document per
    logical page with rich metadata (page number, source, etc.).
    """
    from llama_parse import LlamaParse

    settings = get_settings()
    if not settings.llama_cloud_api_key:
        raise RuntimeError(
            "LLAMA_CLOUD_API_KEY is required for LlamaParse. "
            "Set it in .env or environment."
        )

    parser = LlamaParse(
        api_key=settings.llama_cloud_api_key,
        result_type=settings.llamaparse_result_type,
        verbose=False,
    )
    return parser.load_data(str(path))
