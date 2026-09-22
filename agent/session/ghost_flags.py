"""Ghost-session guards (no LiveKit / adapter imports)."""

from __future__ import annotations

import json
import os


def ghost_turns_allowed() -> bool:
    """Kill switch. Default on — sender must still be a ghost participant."""
    raw = os.environ.get("TALKSHOW_GHOST_TURNS", "1").strip().lower()
    return raw not in ("0", "false", "no", "off")


def is_ghost_participant(participant: object | None) -> bool:
    """True for identity ``ghost-*`` or token metadata ``{"ghost": true}``."""
    if participant is None:
        return False
    identity = (getattr(participant, "identity", None) or "").strip()
    if identity.startswith("ghost-"):
        return True
    raw = getattr(participant, "metadata", None) or ""
    if not raw:
        return False
    try:
        meta = json.loads(raw)
    except json.JSONDecodeError:
        return False
    return bool(isinstance(meta, dict) and meta.get("ghost"))
