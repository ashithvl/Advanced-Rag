"""Per-page PDF ingestion.

Page classifier (no paid call):
    text-only  -> chars >= MIN_CHARS_TEXT_ONLY, no drawings, no raster images
    visual     -> any vector drawing OR any embedded raster image OR very low text
    skip       -> nothing meaningful (chars < MIN_CHARS_KEEP_PAGE and not visual)

Visual pages are rendered to PNG and sent to GPT-4o-mini (the OpenAI equivalent of
Gemini Flash) which returns structured JSON: title, markdown, figures[], visual_summary.
We build TWO texts per page:
    super_text  = title + raw text + structured markdown + figure summaries
    visual_text = visual_summary + figure summaries + labels
…then write ONE row to Weaviate with TWO named vectors (txt_vec + img_vec).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import fitz  # PyMuPDF

from advanced_rag.config import get_settings
from advanced_rag.gemini_client import embed_image
from advanced_rag.indexing.vector_store import get_collection
from advanced_rag.ingestion.registry import now_iso
from advanced_rag.llama_parse_client import is_enabled as llama_parse_enabled
from advanced_rag.llama_parse_client import parse_pages as llama_parse_pages
from advanced_rag.openai_client import chat_vision_json, embed
from advanced_rag.pretty_log import banner, footer, get_logger, kv_lines, timed_step

PAGES_IMG_DIR = Path("data/raw/pages")

_log = get_logger("ingest")

EXTRACT_SYSTEM = (
    "You read ONE page from a technical PDF (mechanical / engineering manuals, datasheets). "
    "Return STRICT JSON with these fields:\n"
    '{"title": "section or page title (string, may be empty)",\n'
    ' "markdown": "all readable text and procedures as clean markdown (string)",\n'
    ' "figures": [\n'
    '   {"id": "figure or table label, e.g. Fig 1 / Table 2",\n'
    '    "kind": "diagram|table|chart|photo|schematic|exploded_view|other",\n'
    '    "summary": "one-sentence summary",\n'
    '    "labels": ["all callouts, part numbers, dimensions, units, axis ticks as strings"]}\n'
    " ],\n"
    ' "visual_summary": "concise paragraph: layout, arrows, colors, spatial relations,'
    ' visual cues that define this page"}\n'
    "Be exhaustive on numbers, units, part numbers. Output JSON only — no preamble."
)


def _classify_page(page: fitz.Page) -> tuple[str, str]:
    """Return (page_class, raw_text). page_class is 'text', 'visual' or 'skip'."""
    s = get_settings()
    raw = (page.get_text("text") or "").strip()
    n_chars = len(raw)
    has_raster = bool(page.get_images(full=True))
    try:
        has_drawings = bool(page.get_drawings())
    except Exception:
        has_drawings = False

    if not has_raster and not has_drawings and n_chars >= s.min_chars_text_only:
        return "text", raw
    if has_raster or has_drawings or n_chars < s.min_chars_text_only:
        if n_chars < s.min_chars_keep_page and not (has_raster or has_drawings):
            return "skip", raw
        return "visual", raw
    return "skip", raw


def _render_page_png(page: fitz.Page, out_path: Path, dpi: int) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    page.get_pixmap(dpi=dpi).save(str(out_path))


def _build_texts(raw_text: str, structured: dict) -> tuple[str, str, str]:
    """Return (title, super_text, visual_text) from raw text + structured JSON.

    `structured` may come from LlamaParse (primary) or GPT-4o-mini vision JSON
    (fallback). Both feed this same shape: optional `title`, `markdown`,
    `visual_summary`, `figures[]`.
    """
    title = (structured.get("title") or "").strip()
    md = (structured.get("markdown") or "").strip()
    visual_summary = (structured.get("visual_summary") or "").strip()
    figures = structured.get("figures") or []

    fig_lines: list[str] = []
    label_lines: list[str] = []
    for f in figures:
        fid = (f.get("id") or "").strip() or "Figure"
        kind = (f.get("kind") or "").strip()
        summary = (f.get("summary") or "").strip()
        labels = [str(x) for x in (f.get("labels") or [])]
        fig_lines.append(f"- {fid} ({kind}): {summary}")
        if labels:
            label_lines.append(f"{fid}: " + ", ".join(labels))

    super_parts = [p for p in [title, raw_text, md, "\n".join(fig_lines)] if p]
    super_text = "\n\n".join(super_parts).strip()

    visual_parts = [p for p in [visual_summary, "\n".join(fig_lines), "\n".join(label_lines)] if p]
    visual_text = "\n\n".join(visual_parts).strip() or super_text  # fallback
    return title, super_text, visual_text


def _build_texts_from_llamaparse(
    raw_text: str,
    parsed: dict,
) -> tuple[str, str, str]:
    """Return (title, super_text, visual_text) when the primary parse is LlamaParse.

    LlamaParse already returns clean markdown with tables / figure captions
    inline, so `super_text` is mostly the markdown. We still fold the raw
    PyMuPDF text in as a safety net (sometimes LlamaParse strips tiny labels
    that matter for technical retrieval).
    """
    title = (parsed.get("title") or "").strip()
    md = (parsed.get("markdown") or "").strip()

    super_parts = [p for p in [title, md, raw_text] if p]
    super_text = "\n\n".join(super_parts).strip()
    visual_text = md or super_text
    return title, super_text, visual_text


def ingest_pdf_for_doc(
    pdf_path: Path,
    doc_id: str,
    original_filename: str,
) -> dict[str, Any]:
    """Index one PDF page-by-page into the named-vector collection."""
    s = get_settings()
    pdf_path = pdf_path.resolve()

    lp_enabled = llama_parse_enabled()
    banner(_log, f" INGEST  {original_filename} ")
    _log.info(
        "Start\n%s",
        kv_lines(
            [
                ("doc_id", doc_id),
                ("path", str(pdf_path)),
                ("extractor", "LlamaParse" if lp_enabled else "GPT-4o-mini vision"),
                ("extract_model", s.extract_model),
                ("text_embedding_model", s.embedding_model),
                ("image_embedding_model", s.gemini_image_embedding_model),
                ("image_embedding_dim", s.gemini_image_embedding_dim),
                ("min_chars_text_only", s.min_chars_text_only),
                ("page_render_dpi", s.page_render_dpi),
            ]
        ),
    )

    collection = get_collection()
    timestamp = now_iso()

    # Primary extractor: LlamaParse (one call for the whole PDF).
    lp_pages_by_num: dict[int, dict[str, Any]] = {}
    if lp_enabled:
        with timed_step(_log, "LlamaParse whole PDF"):
            lp_pages = llama_parse_pages(pdf_path) or []
        for p in lp_pages:
            lp_pages_by_num[int(p["page_number"])] = p

    pages_kept = 0
    pages_visual = 0
    pages_skipped = 0
    pages_llamaparse = 0
    pages_vision_fallback = 0

    doc = fitz.open(pdf_path)
    try:
        total = len(doc)
        _log.info("Opened PDF  pages=%d", total)
        for page_idx in range(total):
            page = doc[page_idx]
            page_number = page_idx + 1
            page_class, raw_text = _classify_page(page)

            if page_class == "skip":
                pages_skipped += 1
                _log.info(
                    "Page %d/%d  SKIP  (blank / negligible)\n%s",
                    page_number,
                    total,
                    kv_lines([("chars", len(raw_text.strip()))]),
                )
                continue

            page_image_path: str | None = None
            structured: dict = {}
            is_visual = page_class == "visual"
            n_chars = len(raw_text.strip())

            # Preferred structured content from LlamaParse (whole-PDF call).
            lp_page = lp_pages_by_num.get(page_number)
            if lp_page and lp_page.get("markdown"):
                structured = {
                    "title": lp_page.get("title", ""),
                    "markdown": lp_page["markdown"],
                }
                pages_llamaparse += 1

            # Render PNG for visual pages (needed for Gemini image embedding).
            if is_visual:
                img_path = (PAGES_IMG_DIR / f"{doc_id}_p{page_number}.png").resolve()
                _render_page_png(page, img_path, s.page_render_dpi)
                page_image_path = str(img_path)
                pages_visual += 1

                # Fallback vision JSON only when LlamaParse didn't give us
                # structured content for this page.
                if not structured:
                    _log.info(
                        "Page %d/%d  VISUAL  chars=%d  → render PNG + %s (fallback)",
                        page_number,
                        total,
                        n_chars,
                        s.extract_model,
                    )
                    with timed_step(_log, f"vision extract p.{page_number}"):
                        structured = chat_vision_json(
                            EXTRACT_SYSTEM,
                            "Extract the page exhaustively as JSON, per the system instructions.",
                            img_path,
                            model=s.extract_model,
                        )
                    pages_vision_fallback += 1
                else:
                    _log.info(
                        "Page %d/%d  VISUAL  chars=%d  → render PNG (LlamaParse markdown used)",
                        page_number,
                        total,
                        n_chars,
                    )
            else:
                _log.info(
                    "Page %d/%d  TEXT-ONLY  chars=%d  (no vision call)",
                    page_number,
                    total,
                    n_chars,
                )

            title, super_text, visual_text = _build_texts(raw_text, structured)
            if not super_text:
                pages_skipped += 1
                _log.warning("Page %d/%d  SKIP  (empty super_text after build)", page_number, total)
                continue

            with timed_step(_log, f"embed + insert p.{page_number}"):
                txt_vec = embed([super_text])[0]
                vectors: dict[str, list[float]] = {"txt_vec": txt_vec}
                if is_visual and page_image_path:
                    # Native multimodal embedding of the rendered page PNG.
                    vectors["img_vec"] = embed_image(Path(page_image_path))

                collection.data.insert(
                    properties={
                        "source_doc_id": doc_id,
                        "source_filename": original_filename,
                        "page_number": page_number,
                        "page_image_path": page_image_path or "",
                        "is_visual": is_visual,
                        "title": title,
                        "super_text": super_text,
                        "visual_text": visual_text,
                        "ingested_at": timestamp,
                    },
                    vector=vectors,
                )
            pages_kept += 1
            _log.info(
                "Page %d/%d  INDEXED\n%s",
                page_number,
                total,
                kv_lines(
                    [
                        ("title", title or "—"),
                        ("super_text_chars", len(super_text)),
                        ("visual_text_chars", len(visual_text)),
                        ("vectors", list(vectors.keys())),
                        ("png", page_image_path or "—"),
                    ],
                    max_str=120,
                ),
            )
    finally:
        doc.close()

    _log.info(
        "Ingest summary\n%s",
        kv_lines(
            [
                ("pages_indexed", pages_kept),
                ("pages_visual_api", pages_visual),
                ("pages_llamaparse", pages_llamaparse),
                ("pages_vision_fallback", pages_vision_fallback),
                ("pages_skipped", pages_skipped),
            ]
        ),
    )
    footer(_log)

    return {
        "doc_id": doc_id,
        "pages": pages_kept,
        "pages_visual": pages_visual,
        "pages_llamaparse": pages_llamaparse,
        "pages_vision_fallback": pages_vision_fallback,
        "pages_skipped": pages_skipped,
    }
