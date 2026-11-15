"""Cap spoken lines at a number of sentences (TALKSHOW_MAX_SENTENCES=<n>, default off).

Larger panel models write long turns (median 57 words with Qwen3.6-35B-A3B in T6), which run 20+ s on air. With the
flag set, every Think & Action ``content`` is cut after the n-th sentence when the think output is parsed. The model is
not told the limit: in T6 a prompt rule ("at most 2 short sentences, one point") made lines thinner, and the plain
lines cut to their first 2 sentences won 80% of pairwise comparisons against them (eval/DIALOGUE_QUALITY.md).
"""

from __future__ import annotations

import os
import re

# A sentence ends at . ! ? (optionally followed by a closing quote or bracket) before whitespace and a new sentence.
# Single capital letters and common abbreviations ("U.S.", "Dr.", "e.g.") do not end a sentence.
_END = re.compile(r"(?<=[.!?])[\"'’”)\]]?\s+(?=[\"'‘“(\[]?[A-Z0-9])")
_ABBREV = re.compile(r"(?:\b[A-Z]|\b(?:Mr|Mrs|Ms|Dr|Prof|St|vs|etc|e\.g|i\.e|U\.S|U\.K|No))\.$")


def max_sentences() -> int:
    """The cap from TALKSHOW_MAX_SENTENCES (0 = off)."""
    try:
        return max(0, int(os.environ.get("TALKSHOW_MAX_SENTENCES", "0").strip() or 0))
    except ValueError:
        return 0


def split_sentences(text: str) -> list[str]:
    parts: list[str] = []
    start = 0
    for m in _END.finditer(text):
        head = text[start:m.start()].rstrip()
        if _ABBREV.search(head):
            continue
        parts.append(text[start:m.start()].strip())
        start = m.end()
    tail = text[start:].strip()
    if tail:
        parts.append(tail)
    return [p for p in parts if p]


def cap_sentences(text: str | None, n: int | None = None) -> str | None:
    """``text`` cut after its n-th sentence (n from the flag when None; unchanged when the cap is off)."""
    n = max_sentences() if n is None else n
    if not text or n <= 0:
        return text
    parts = split_sentences(text)
    return text if len(parts) <= n else " ".join(parts[:n])

