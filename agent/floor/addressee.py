"""Addressee-aware floor (behind TALKSHOW_ADDRESSEE_FLOOR, default off).

The benchmark showed the floor never follows the human's addressee: when the human asked one
panelist by name, the other one answered first. This module detects the vocative; the floor
hook that uses it lives with the free-floor choice.
"""

from __future__ import annotations

import os
import re


def addressee_floor_enabled() -> bool:
    return os.environ.get("TALKSHOW_ADDRESSEE_FLOOR", "0").strip().lower() in ("1", "true", "yes", "on")


def addressed_role(text: str, names: dict[str, str]) -> str | None:
    """Role whose name the human used as a vocative, else None.

    Vocative = the name opens the line ("Amy, …", "Amy — …", "Amy: …"), or closes a question
    (", Amy?" / "… Amy?"). A name merely mentioned mid-sentence ("Ryan said …") does not count.
    Ambiguous (two different panelists addressed) → None.
    """
    text = (text or "").strip()
    hits: set[str] = set()
    for role, name in names.items():
        n = re.escape(name)
        if re.match(rf"^(?:(?:hey|ok|okay|so|but|and|well),?\s+)?{n}\s*[,—–:\-]", text, re.IGNORECASE):
            hits.add(role)
        elif re.search(rf"(?:,|\b)\s*{n}\s*\?\s*$", text, re.IGNORECASE):
            hits.add(role)
    return next(iter(hits)) if len(hits) == 1 else None

