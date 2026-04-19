"""End-to-end RAG pipeline orchestrated with LangGraph.

    hyde     -> GPT-4o-mini writes one paragraph describing what the answering page
                would visually look like (used to query the image vector)
    retrieve -> Weaviate hybrid (BM25 + txt_vec) on the user question, in parallel with
                near_vector(img_vec) on the HyDE embedding; merge the two ranked lists
                with Reciprocal Rank Fusion
    rerank   -> Cohere Rerank picks the top-N pages
    answer   -> GPT-4o answers from the reranked super-text and re-shows the top page
                images (image is the source of truth on conflicts)
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, TypedDict

from langchain_core.outputs import Generation
from langgraph.graph import END, StateGraph
from weaviate.classes.query import MetadataQuery

from advanced_rag.config import get_settings
from advanced_rag.gemini_client import embed_text as embed_text_multimodal
from advanced_rag.indexing.vector_store import get_collection
from advanced_rag.openai_client import chat, chat_with_images, embed
from advanced_rag.pretty_log import banner, footer, get_logger, kv_lines
from advanced_rag.query_cache import LLM_STRING, get_cache
from advanced_rag.retrieval.rerank import cohere_rerank
from advanced_rag.smalltalk import is_smalltalk

_log = get_logger("pipeline")


class RagState(TypedDict, total=False):
    question: str
    hyde: str
    contexts: list[dict[str, Any]]   # reranked page records
    answer: str
    confidence: float
    cache_hit: bool                  # True if served from semantic query cache
    smalltalk: bool                  # True if served from small-talk gate
    cache_similarity: float          # cosine score of the cache hit
    cache_matched_question: str      # original cached question we matched


HYDE_SYSTEM = (
    "You help search a technical PDF library. Given a user question, write ONE short "
    "paragraph (3-5 sentences) describing what an answering page would LOOK LIKE: "
    "diagram type, table, callouts, the specific labels / numbers / units expected. "
    "No preamble, no markdown, no bullet points."
)

ANSWER_SYSTEM = (
    "You are a strict assistant for technical documentation. "
    "Use ONLY the provided context (text snippets and page images). "
    "If the context does not answer the question, reply exactly: "
    "'Not specified in the manual.' "
    "Cite sources inline as [source: <file>, p.<page>]. "
    "Preserve units, part numbers and step order verbatim. "
    "When text and image disagree on a number, the image is the source of truth."
)

RETURN_PROPS = [
    "source_doc_id",
    "source_filename",
    "page_number",
    "page_image_path",
    "is_visual",
    "title",
    "super_text",
]


# ---------- nodes ----------

def _hyde(state: RagState) -> RagState:
    s = get_settings()
    _log.info("HyDE step  model=%s", s.hyde_model)
    out = chat(HYDE_SYSTEM, state["question"], model=s.hyde_model, temperature=0.2)
    if _log.isEnabledFor(logging.DEBUG):
        _log.debug("HyDE paragraph (full):\n%s", out)
    else:
        preview = (out[:320] + "…") if len(out) > 320 else out
        _log.info("HyDE preview (first 320 chars):\n%s", preview)
    return {"hyde": out}


def _retrieve(state: RagState) -> RagState:
    s = get_settings()
    col = get_collection()

    _log.info(
        "Retrieve  hybrid_alpha=%s  top_k_text=%d  top_k_image=%d",
        s.hybrid_alpha,
        s.top_k_text,
        s.top_k_image,
    )

    # txt_vec is OpenAI-space; img_vec is Gemini multimodal-space.
    # Embed each query side with the matching model so vector spaces align.
    q_vec = embed([state["question"]])[0]
    h_vec = embed_text_multimodal(state.get("hyde") or state["question"])

    text_hits = col.query.hybrid(
        query=state["question"],
        vector=q_vec,
        target_vector="txt_vec",
        alpha=s.hybrid_alpha,
        limit=s.top_k_text,
        return_metadata=MetadataQuery(score=True),
        return_properties=RETURN_PROPS,
    ).objects

    image_hits = col.query.near_vector(
        near_vector=h_vec,
        target_vector="img_vec",
        limit=s.top_k_image,
        return_metadata=MetadataQuery(distance=True),
        return_properties=RETURN_PROPS,
    ).objects

    _log.info(
        "Weaviate raw hits\n%s",
        kv_lines(
            [
                ("text_hybrid_count", len(text_hits)),
                ("image_near_vector_count", len(image_hits)),
                ("text_top_5", _summarize_hits(text_hits, 5)),
                ("image_top_5", _summarize_hits(image_hits, 5)),
            ],
            max_json=2500,
        ),
    )

    text_ids = _ids_in_order(text_hits)
    image_ids = _ids_in_order(image_hits)
    rrf_scores = _rrf_scores([text_ids, image_ids])
    fused_ids = sorted(rrf_scores, key=lambda i: rrf_scores[i], reverse=True)

    top_rrf = [(uid[:8] + "…", round(rrf_scores[uid], 5)) for uid in fused_ids[:12]]
    _log.info(
        "RRF fusion (k=60)  unique_ids=%d\n%s",
        len(fused_ids),
        kv_lines([("top_rrf", top_rrf)]),
    )

    by_id: dict[str, Any] = {str(o.uuid): o for o in (*text_hits, *image_hits)}
    candidates = [_to_record(by_id[uid]) for uid in fused_ids if uid in by_id]

    reranked = cohere_rerank(state["question"], candidates, top_n=s.top_k_rerank)
    confidence = max((c["score"] for c in reranked), default=0.0)
    return {"contexts": reranked, "confidence": confidence}


def _gate(state: RagState) -> str:
    threshold = get_settings().confidence_threshold
    conf = state.get("confidence", 0.0)
    route = "answer" if conf >= threshold else "fallback"
    _log.info(
        "Gate  confidence=%.4f  threshold=%.4f  → %s",
        conf,
        threshold,
        route,
    )
    return route


def _answer(state: RagState) -> RagState:
    s = get_settings()
    contexts = state.get("contexts") or []

    text_block = "\n\n---\n\n".join(
        f"[source: {c['source_filename']}, p.{c['page_number']}]\n"
        f"{c.get('title') or ''}\n{c['text']}"
        for c in contexts
    )
    user_text = f"Question:\n{state['question']}\n\nContext:\n{text_block}"

    image_paths: list[Path] = []
    for c in contexts:
        if len(image_paths) >= s.top_k_vision_in_answer:
            break
        p = c.get("page_image_path") or ""
        if p and Path(p).is_file():
            image_paths.append(Path(p))

    _log.info(
        "Answer step  model=%s  context_pages=%d  images_attached=%d",
        s.answer_model,
        len(contexts),
        len(image_paths),
    )
    answer = chat_with_images(
        ANSWER_SYSTEM,
        user_text,
        image_paths,
        model=s.answer_model,
        temperature=0.0,
    )
    preview = (answer[:400] + "…") if len(answer) > 400 else answer
    _log.info("Answer preview:\n%s", preview)
    return {"answer": answer}


def _fallback(_: RagState) -> RagState:
    _log.warning("Fallback answer (below confidence threshold).")
    return {"answer": "Not specified in the manual."}


# ---------- helpers ----------

def _ids_in_order(objects: list[Any]) -> list[str]:
    return [str(o.uuid) for o in objects]


def _rrf_scores(rankings: list[list[str]], k: int = 60) -> dict[str, float]:
    scores: dict[str, float] = {}
    for ranked in rankings:
        for rank, uid in enumerate(ranked):
            scores[uid] = scores.get(uid, 0.0) + 1.0 / (k + rank + 1)
    return scores


def _summarize_hits(objects: list[Any], n: int) -> list[str]:
    lines: list[str] = []
    for o in objects[:n]:
        p = o.properties or {}
        meta = o.metadata
        score = getattr(meta, "score", None)
        dist = getattr(meta, "distance", None)
        if score is not None:
            extra = f"score={score:.4f}"
        elif dist is not None:
            extra = f"distance={dist:.4f}"
        else:
            extra = "—"
        lines.append(
            f"p.{p.get('page_number')} {p.get('source_filename', '')[:36]}  {extra}  "
            f"visual={p.get('is_visual')}"
        )
    return lines


def _to_record(obj: Any) -> dict[str, Any]:
    p = obj.properties or {}
    return {
        "id": str(obj.uuid),
        "text": p.get("super_text", ""),
        "source_filename": p.get("source_filename", "?"),
        "page_number": p.get("page_number", "?"),
        "page_image_path": p.get("page_image_path", ""),
        "is_visual": bool(p.get("is_visual", False)),
        "title": p.get("title", ""),
    }


# ---------- graph ----------

def build_graph() -> Any:
    g = StateGraph(RagState)
    g.add_node("hyde", _hyde)
    g.add_node("retrieve", _retrieve)
    g.add_node("answer", _answer)
    g.add_node("fallback", _fallback)

    g.set_entry_point("hyde")
    g.add_edge("hyde", "retrieve")
    g.add_conditional_edges("retrieve", _gate, {"answer": "answer", "fallback": "fallback"})
    g.add_edge("answer", END)
    g.add_edge("fallback", END)
    return g.compile()


def run(question: str) -> RagState:
    """Convenience entry point for FastAPI / Streamlit / scripts.

    Order of operations:
        1. Small-talk gate     — fixed reply, zero API calls.
        2. Semantic cache hit  — replay previous answer; one OpenAI embed only.
        3. Full LangGraph      — HyDE → retrieve → gate → answer | fallback.
        4. Cache store         — write the fresh answer for next time.
    """
    s = get_settings()
    banner(_log, " QUERY PIPELINE ")
    _log.info(
        "Question\n%s",
        kv_lines([("text", question), ("chars", len(question))]),
    )

    try:
        # 1) Small-talk gate.
        if s.enable_smalltalk and is_smalltalk(question):
            _log.info("Small-talk detected → canned reply (no API calls).")
            return {
                "question": question,
                "answer": s.smalltalk_reply,
                "confidence": 1.0,
                "contexts": [],
                "cache_hit": False,
                "smalltalk": True,
            }

        # 2) Semantic cache lookup.
        cache = get_cache()
        if cache is not None:
            cached = cache.lookup(question, LLM_STRING)
            if cached:
                gen = cached[0]
                info = gen.generation_info or {}
                return {
                    "question": question,
                    "answer": gen.text,
                    "confidence": float(info.get("confidence", 0.0)),
                    "contexts": list(info.get("contexts", [])),
                    "cache_hit": True,
                    "smalltalk": False,
                    "cache_similarity": float(info.get("similarity", 0.0)),
                    "cache_matched_question": info.get("matched_question", ""),
                }

        # 3) Full pipeline.
        out: RagState = build_graph().invoke({"question": question})
        out["cache_hit"] = False
        out["smalltalk"] = False

        # 4) Store on miss (only if we actually answered, not on fallback).
        if cache is not None and out.get("answer") and not _is_fallback(out, s):
            cache.update(
                question,
                LLM_STRING,
                [
                    Generation(
                        text=out["answer"],
                        generation_info={
                            "confidence": out.get("confidence", 0.0),
                            "contexts": out.get("contexts", []),
                        },
                    )
                ],
            )
        return out
    finally:
        footer(_log)


def _is_fallback(state: RagState, s: Any) -> bool:
    """Heuristic: don't cache the fixed fallback string."""
    return state.get("answer", "").strip() == "Not specified in the manual."
