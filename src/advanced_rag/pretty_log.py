"""Readable logs for the RAG stack (one handler on ``advanced_rag``).

Set ``LOG_LEVEL=DEBUG`` for longer excerpts (HyDE text, super_text previews).
"""

from __future__ import annotations

import json
import logging
import os
import sys
import time
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

_NS = "advanced_rag"
_configured = False


class _PrettyFormatter(logging.Formatter):
    """``2026-04-19 12:34:56 │ INFO  │ pipeline   │ message``"""

    def format(self, record: logging.LogRecord) -> str:
        ts = self.formatTime(record, "%Y-%m-%d %H:%M:%S")
        short = record.name.removeprefix(f"{_NS}.") if record.name.startswith(_NS) else record.name
        short = (short + " " * 12)[:12]
        return f"{ts} │ {record.levelname:5} │ {short} │ {record.getMessage()}"


def setup_logging(level: str | None = None, *, force: bool = False) -> None:
    """Attach a pretty handler to the ``advanced_rag`` logger only (no root hijack)."""
    global _configured
    if _configured and not force:
        return
    lvl = (level or os.getenv("LOG_LEVEL", "INFO")).upper()
    log = logging.getLogger(_NS)
    log.handlers.clear()
    h = logging.StreamHandler(sys.stderr)
    h.setFormatter(_PrettyFormatter())
    log.addHandler(h)
    log.setLevel(lvl)
    log.propagate = False  # avoid duplicate lines on the root / uvicorn default handler
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("openai").setLevel(logging.WARNING)
    logging.getLogger("weaviate").setLevel(logging.WARNING)
    _configured = True


def get_logger(suffix: str) -> logging.Logger:
    """``suffix`` like ``ingest`` → ``advanced_rag.ingest``."""
    name = suffix if suffix.startswith(_NS) else f"{_NS}.{suffix}"
    return logging.getLogger(name)


def banner(log: logging.Logger, title: str, *, width: int = 70) -> None:
    inner = f" {title} "
    pad = max(0, width - 2 - len(inner))
    left, right = pad // 2, pad - pad // 2
    log.info("┏%s%s%s┓", "━" * left, inner, "━" * right)


def footer(log: logging.Logger, *, width: int = 70) -> None:
    log.info("┗%s┛", "━" * (width - 2))


def kv_lines(rows: list[tuple[str, Any]], *, max_str: int = 600, max_json: int = 1200) -> str:
    """Turn rows into an indented block for one log message."""
    out: list[str] = []
    for key, val in rows:
        out.append(f"  • {key}: {_format_val(val, max_str=max_str, max_json=max_json)}")
    return "\n".join(out)


def _format_val(val: Any, *, max_str: int, max_json: int) -> str:
    if val is None:
        return "—"
    if isinstance(val, float):
        return f"{val:.4f}"
    if isinstance(val, bool):
        return "yes" if val else "no"
    if isinstance(val, dict | list):
        s = json.dumps(val, ensure_ascii=False, indent=2)
        if len(s) > max_json:
            return s[:max_json] + "\n  … (truncated)"
        return s
    s = str(val)
    if len(s) > max_str:
        return s[:max_str] + " …"
    return s


@contextmanager
def timed_step(log: logging.Logger, label: str) -> Iterator[None]:
    t0 = time.perf_counter()
    log.info("… start: %s", label)
    try:
        yield
    finally:
        ms = (time.perf_counter() - t0) * 1000
        log.info("… done:  %s  (%.0f ms)", label, ms)
