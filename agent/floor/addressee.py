"""Addressee-aware floor (behind TALKSHOW_ADDRESSEE_FLOOR, default off).

The benchmark showed the floor never follows the human's addressee: when the human asked one
panelist by name, the other one answered first.

Think & Action: every agent thinks on the human's line (silent / speak) and the free-floor
controller (``ThinkActionInteraction.arbitrate_free``) picks a speaker. With the flag on, when the
latest human turn names one panelist (no panelist has spoken since; host lines skipped) and that
panelist chose "speak", they get the floor without asking the controller. If they chose silent, the
choice is unchanged (nobody is forced to speak) and the miss is logged.
"""

from __future__ import annotations

import os
import re
from typing import Any


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


def addressed_after_human(
    lines: list[Any],
    panel_roles: list[str],
    *,
    host_role: str = "host",
    names: dict[str, str] | None = None,
) -> str | None:
    """Panelist the latest human turn named, if no panelist has spoken since it (host lines skipped)."""
    for line in reversed(lines):
        if line.role_id == host_role:
            continue
        if line.role_id != "human":
            return None  # a panelist already answered this human turn
        if names is None:
            from agent.config import load_persona_name

            names = {r: load_persona_name(r) for r in panel_roles}
        return addressed_role(line.text, names)
    return None


def host_invitee(text: str, names: dict[str, str]) -> str | None:
    """Panelist the host hands the floor to: the only one named in the line's last sentence
    that names anyone ("Amy, good point. Ryan, what do you think?" → Ryan). None if that
    sentence names two panelists or the line names nobody."""
    sentences = [s for s in re.split(r"(?<=[.!?])\s+", (text or "").strip()) if s]
    for sentence in reversed(sentences):
        hits = {
            role
            for role, name in names.items()
            if re.search(rf"\b{re.escape(name)}\b", sentence, re.IGNORECASE)
        }
        if hits:
            return next(iter(hits)) if len(hits) == 1 else None
    return None
