"""Thin wrappers around LangChain OpenAI primitives.

We keep the same public surface (`embed`, `chat`, `chat_vision_json`,
`chat_with_images`) so the rest of the codebase doesn't change, but the
implementations now go through:

  * `langchain_openai.OpenAIEmbeddings` for text embeddings (txt_vec + the
    semantic query cache).
  * `langchain_openai.ChatOpenAI` for chat completions (HyDE + final
    multimodal answer + the legacy vision JSON fallback when LlamaParse is
    not configured).

Multimodal Gemini embeddings still live in `gemini_client.py` because
LangChain's Google wrapper doesn't expose Gemini Embedding 2's image input.
"""

from __future__ import annotations

import base64
import json
import time
from functools import lru_cache
from pathlib import Path
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from advanced_rag.config import get_settings
from advanced_rag.pretty_log import get_logger, kv_lines

_log = get_logger("openai")


@lru_cache(maxsize=1)
def _embedder() -> OpenAIEmbeddings:
    s = get_settings()
    if not s.openai_api_key:
        raise RuntimeError("OPENAI_API_KEY is required.")
    return OpenAIEmbeddings(model=s.embedding_model, api_key=s.openai_api_key)


@lru_cache(maxsize=8)
def _chat(model: str, temperature: float) -> ChatOpenAI:
    s = get_settings()
    if not s.openai_api_key:
        raise RuntimeError("OPENAI_API_KEY is required.")
    return ChatOpenAI(model=model, temperature=temperature, api_key=s.openai_api_key)


def embed(texts: list[str]) -> list[list[float]]:
    """Embed a list of strings with the configured embedding model."""
    s = get_settings()
    t0 = time.perf_counter()
    vecs = _embedder().embed_documents(texts)
    ms = (time.perf_counter() - t0) * 1000
    dim = len(vecs[0]) if vecs else 0
    _log.info(
        "OpenAI embeddings (LangChain)\n%s",
        kv_lines(
            [
                ("model", s.embedding_model),
                ("strings", len(texts)),
                ("dim", dim),
                ("ms", f"{ms:.0f}"),
            ]
        ),
    )
    return vecs


def chat(system: str, user: str, *, model: str, temperature: float = 0.0) -> str:
    """Plain text chat completion via LangChain ChatOpenAI."""
    t0 = time.perf_counter()
    llm = _chat(model, temperature)
    resp = llm.invoke(
        [
            SystemMessage(content=system),
            HumanMessage(content=user),
        ]
    )
    out = (resp.content or "").strip() if isinstance(resp.content, str) else str(resp.content)
    ms = (time.perf_counter() - t0) * 1000
    _log.info(
        "OpenAI chat (LangChain)\n%s",
        kv_lines(
            [
                ("model", model),
                ("temperature", temperature),
                ("user_chars", len(user)),
                ("reply_chars", len(out)),
                ("ms", f"{ms:.0f}"),
            ]
        ),
    )
    return out


def chat_vision_json(system: str, user_text: str, image_path: Path, *, model: str) -> dict:
    """Vision chat completion with JSON-mode output. Returns a dict (or {} on parse fail).

    Used as the *fallback* extractor when LlamaParse is not configured (legacy
    path); LlamaParse handles structured PDF extraction in the primary path.
    """
    t0 = time.perf_counter()
    b64 = base64.b64encode(image_path.read_bytes()).decode("ascii")
    llm = _chat(model, 0.0).bind(response_format={"type": "json_object"})
    resp = llm.invoke(
        [
            SystemMessage(content=system),
            HumanMessage(
                content=[
                    {"type": "text", "text": user_text},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}},
                ]
            ),
        ]
    )
    raw = (resp.content or "{}") if isinstance(resp.content, str) else "{}"
    ms = (time.perf_counter() - t0) * 1000
    try:
        parsed: dict = json.loads(raw)
    except json.JSONDecodeError:
        parsed = {}
        _log.warning("Vision JSON parse failed; raw length=%d", len(raw))
    keys = list(parsed.keys()) if parsed else []
    _log.info(
        "OpenAI vision JSON (LangChain)\n%s",
        kv_lines(
            [
                ("model", model),
                ("image", str(image_path)),
                ("image_bytes", image_path.stat().st_size),
                ("top_level_keys", keys),
                ("ms", f"{ms:.0f}"),
            ]
        ),
    )
    return parsed


def chat_with_images(
    system: str,
    user_text: str,
    image_paths: list[Path],
    *,
    model: str,
    temperature: float = 0.0,
) -> str:
    """Multimodal chat completion: one text part plus N inline page images."""
    t0 = time.perf_counter()
    user_parts: list[dict[str, Any]] = [{"type": "text", "text": user_text}]
    for p in image_paths:
        b64 = base64.b64encode(p.read_bytes()).decode("ascii")
        user_parts.append(
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}}
        )
    llm = _chat(model, temperature)
    resp = llm.invoke(
        [
            SystemMessage(content=system),
            HumanMessage(content=user_parts),
        ]
    )
    out = (resp.content or "").strip() if isinstance(resp.content, str) else str(resp.content)
    ms = (time.perf_counter() - t0) * 1000
    _log.info(
        "OpenAI multimodal answer (LangChain)\n%s",
        kv_lines(
            [
                ("model", model),
                ("images_attached", len(image_paths)),
                ("context_chars", len(user_text)),
                ("reply_chars", len(out)),
                ("ms", f"{ms:.0f}"),
            ]
        ),
    )
    return out
