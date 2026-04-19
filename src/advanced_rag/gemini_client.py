"""Tiny wrapper around google-genai for native multimodal embeddings.

We use Gemini Embedding 2 (`gemini-embedding-2-preview`) for the visual side
of the index:
  * `embed_image(png_path)` -> stored as `img_vec` for visual pages.
  * `embed_text(text)`      -> used to embed the HyDE paragraph at query time
                               so the question and the page images live in the
                               same shared vector space.

Text-only pages keep using OpenAI `text-embedding-3-small` for `txt_vec`,
which is independent (Weaviate named vectors don't need to share a dim).
"""

from __future__ import annotations

import time
from functools import lru_cache
from pathlib import Path
from typing import Any

from google import genai
from google.genai import types

from advanced_rag.config import get_settings
from advanced_rag.pretty_log import get_logger, kv_lines

_log = get_logger("gemini")


@lru_cache(maxsize=1)
def client() -> genai.Client:
    s = get_settings()
    if not s.gemini_api_key:
        raise RuntimeError(
            "GEMINI_API_KEY is required for image embeddings "
            "(used by gemini-embedding-2-preview)."
        )
    return genai.Client(api_key=s.gemini_api_key)


def _embed(contents: list[Any], *, label: str) -> list[float]:
    s = get_settings()
    t0 = time.perf_counter()
    cfg = types.EmbedContentConfig(output_dimensionality=s.gemini_image_embedding_dim)
    resp = client().models.embed_content(
        model=s.gemini_image_embedding_model,
        contents=contents,
        config=cfg,
    )
    vec = list(resp.embeddings[0].values)
    ms = (time.perf_counter() - t0) * 1000
    _log.info(
        "Gemini embed (%s)\n%s",
        label,
        kv_lines(
            [
                ("model", s.gemini_image_embedding_model),
                ("dim", len(vec)),
                ("ms", f"{ms:.0f}"),
            ]
        ),
    )
    return vec


def embed_image(image_path: Path) -> list[float]:
    """Embed a single PNG page render into the multimodal vector space."""
    data = image_path.read_bytes()
    return _embed(
        [types.Part.from_bytes(data=data, mime_type="image/png")],
        label=f"image {image_path.name}",
    )


def embed_text(text: str) -> list[float]:
    """Embed text into the *same* multimodal vector space as `embed_image`."""
    return _embed([text], label=f"text ({len(text)} chars)")
