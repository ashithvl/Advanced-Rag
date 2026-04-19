"""Semantic query cache for the full RAG response.

We use LangChain's `BaseCache` interface so this module is swappable with
LangChain's built-in caches (Redis / GPTCache / SQLiteVSS / etc.) later.
For zero-infra simplicity we back it with:
  * OpenAI `text-embedding-3-small` (already configured) for the question
    embedding,
  * an in-memory list of `(embedding, question, payload)` tuples,
  * a JSON file on disk for persistence between process restarts.

Cache hits skip the entire pipeline (HyDE / Weaviate / Cohere / GPT-4o).
The cached payload is the **full** RAG response: answer text, confidence,
and citations — so a hit costs one OpenAI embedding call and zero else.

On `lookup(prompt, llm_string)` we embed the prompt and return a one-element
`[Generation]` list whose `.text` is the answer and whose `.generation_info`
carries `{"confidence": float, "contexts": list[dict], "cache_hit": True,
"matched_question": str, "similarity": float}`. The pipeline unwraps that.
"""

from __future__ import annotations

import json
import math
import threading
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from langchain_core.caches import BaseCache
from langchain_core.outputs import Generation

from advanced_rag.config import get_settings
from advanced_rag.openai_client import embed
from advanced_rag.pretty_log import get_logger, kv_lines

_log = get_logger("cache")

LLM_STRING = "advanced_rag.pipeline"  # cache namespace


def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b, strict=False))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return dot / (na * nb)


class SemanticQueryCache(BaseCache):
    """LangChain BaseCache subclass with cosine similarity over OpenAI embeds."""

    def __init__(
        self,
        path: Path,
        threshold: float,
        max_entries: int,
    ) -> None:
        self._path = path
        self._threshold = threshold
        self._max_entries = max_entries
        self._lock = threading.Lock()
        self._entries: list[dict[str, Any]] = []
        self._load()

    # ---- persistence ----

    def _load(self) -> None:
        if not self._path.is_file():
            self._entries = []
            return
        try:
            data = json.loads(self._path.read_text(encoding="utf-8"))
            self._entries = list(data.get("entries", []))
            _log.info(
                "Loaded semantic cache\n%s",
                kv_lines(
                    [
                        ("path", str(self._path)),
                        ("entries", len(self._entries)),
                        ("threshold", self._threshold),
                    ]
                ),
            )
        except (OSError, json.JSONDecodeError) as exc:
            _log.warning("Failed to load query cache (%s); starting empty.", exc)
            self._entries = []

    def _persist(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self._path.with_suffix(self._path.suffix + ".tmp")
        tmp.write_text(
            json.dumps({"entries": self._entries}, ensure_ascii=False),
            encoding="utf-8",
        )
        tmp.replace(self._path)

    # ---- BaseCache interface ----

    def lookup(self, prompt: str, llm_string: str) -> Sequence[Generation] | None:
        if llm_string != LLM_STRING:
            return None
        if not prompt or not prompt.strip():
            return None
        if not self._entries:
            _log.info("Semantic cache lookup  empty cache  → MISS")
            return None

        q_vec = embed([prompt])[0]

        best_score = -1.0
        best_entry: dict[str, Any] | None = None
        with self._lock:
            for entry in self._entries:
                score = _cosine(q_vec, entry["embedding"])
                if score > best_score:
                    best_score = score
                    best_entry = entry

        if best_entry is None or best_score < self._threshold:
            _log.info(
                "Semantic cache MISS\n%s",
                kv_lines(
                    [
                        ("entries_scanned", len(self._entries)),
                        ("best_similarity", round(best_score, 4)),
                        ("threshold", self._threshold),
                    ]
                ),
            )
            return None

        payload = best_entry["payload"]
        gen = Generation(
            text=payload.get("answer", ""),
            generation_info={
                "confidence": payload.get("confidence", 0.0),
                "contexts": payload.get("contexts", []),
                "cache_hit": True,
                "matched_question": best_entry["question"],
                "similarity": best_score,
            },
        )
        _log.info(
            "Semantic cache HIT\n%s",
            kv_lines(
                [
                    ("similarity", round(best_score, 4)),
                    ("threshold", self._threshold),
                    ("matched_question", best_entry["question"]),
                ]
            ),
        )
        return [gen]

    def update(
        self,
        prompt: str,
        llm_string: str,
        return_val: Sequence[Generation],
    ) -> None:
        if llm_string != LLM_STRING:
            return
        if not prompt or not prompt.strip() or not return_val:
            return
        gen = return_val[0]
        info = gen.generation_info or {}
        if info.get("cache_hit"):
            return  # don't re-store a replay

        q_vec = embed([prompt])[0]
        entry = {
            "question": prompt,
            "embedding": q_vec,
            "payload": {
                "answer": gen.text,
                "confidence": float(info.get("confidence", 0.0)),
                "contexts": info.get("contexts", []),
            },
            "created_at": time.time(),
        }
        with self._lock:
            self._entries.append(entry)
            if len(self._entries) > self._max_entries:
                self._entries = self._entries[-self._max_entries :]
            self._persist()
        _log.info(
            "Semantic cache STORE\n%s",
            kv_lines(
                [
                    ("question", prompt),
                    ("total_entries", len(self._entries)),
                    ("path", str(self._path)),
                ]
            ),
        )

    def clear(self, **_: Any) -> None:
        with self._lock:
            self._entries = []
            self._persist()
        _log.info("Semantic cache CLEARED  path=%s", self._path)


# ---- module-level singleton ----

_INSTANCE: SemanticQueryCache | None = None


def get_cache() -> SemanticQueryCache | None:
    """Return the singleton cache (or None if disabled in config)."""
    global _INSTANCE
    s = get_settings()
    if not s.enable_query_cache:
        return None
    if _INSTANCE is None:
        _INSTANCE = SemanticQueryCache(
            path=Path(s.query_cache_path),
            threshold=s.query_cache_threshold,
            max_entries=s.query_cache_max_entries,
        )
    return _INSTANCE
