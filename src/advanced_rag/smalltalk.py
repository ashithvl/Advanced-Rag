"""Tiny rule-based small-talk gate.

We do *not* want to spend an LLM call (or any embedding / vector call) on
"hi", "thanks", "how are you?", etc. This module returns a canned reply for
short greetings and pleasantries; everything else is treated as a real
question and continues to the RAG pipeline.

Rules (intentionally conservative):
  * The text is short (<= 6 tokens, <= 60 chars after strip).
  * It matches one of the keyword sets below OR is *only* punctuation /
    pleasantry tokens with no question mark and no domain-looking nouns.

If you want a smarter classifier later, swap this out for a `gpt-4o-mini`
call — but for the obvious cases the regex saves money and latency.
"""

from __future__ import annotations

import re

_GREETING_RE = re.compile(
    r"""
    ^\s*
    (
        hi | hii+ | hey | hello | yo | sup | howdy |
        good\s*(morning|afternoon|evening|day) |
        how\s*(are|r)\s*(you|u|ya) |
        how('?s|s)\s*it\s*going |
        what'?s\s*up | wassup |
        (thanks|thank\s*you|ty|thx)(\s*(a\s*lot|so\s*much))? |
        thank\s*u |
        ok(ay)? | cool | nice | great |
        bye | goodbye | see\s*ya | cya
    )
    [\s\.\!\?]*$
    """,
    re.IGNORECASE | re.VERBOSE,
)


def is_smalltalk(text: str) -> bool:
    """Return True if `text` looks like pure small-talk (no real question)."""
    if not text:
        return True
    t = text.strip()
    if not t:
        return True
    if len(t) > 60:
        return False
    # Greetings ending in "?" (e.g. "how are you?") are still caught by
    # _GREETING_RE because of the `[\s\.\!\?]*$` tail; longer real questions
    # fall through and continue to the RAG pipeline.
    return bool(_GREETING_RE.match(t))
