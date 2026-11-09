"""Addressee-aware floor (behind TALKSHOW_ADDRESSEE_FLOOR, default off).

The benchmark showed the floor never follows the human's addressee: after a human turn both
panelists raise their hands and the ``panel_priority`` tie-break picks the same panelist every
time, even when the human asked the other one by name. With the flag on, the first pick after a
human turn goes to the panelist the human addressed by name (vocative), whether or not they raised
their hand; everything else is unchanged.
"""

from __future__ import annotations

import os
import re
from dataclasses import replace

from agent.floor.floor_parser import HandRaiseResult


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


def apply_addressee(
    raises: dict[str, HandRaiseResult],
    panel_priority: list[str],
    addressed: str | None,
) -> tuple[dict[str, HandRaiseResult], list[str]]:
    """Return (raises, priority) that make ``addressed`` win the poll; unchanged if None/unknown."""
    if not addressed or addressed not in raises:
        return raises, panel_priority
    out = dict(raises)
    if not out[addressed].raised:
        out[addressed] = replace(out[addressed], raised=True, reason="addressed by the guest")
    priority = [addressed, *[r for r in panel_priority if r != addressed]]
    return out, priority


def pick_with_addressee(queue, raises: dict[str, HandRaiseResult], panel_priority: list[str], addressed: str | None):
    """One poll pick: (winner, yes_roles, had_tie, new_priority).

    Same as the production tie-break when ``addressed`` is None. With an addressee, they win, and the
    priority is rotated afterwards exactly as after a tie — so the OTHER panelist gets the next turn
    (without the rotation the addressee also won the next tie: same speaker twice in 86% of beats).
    """
    priority = list(panel_priority)
    if addressed:
        raises, priority = apply_addressee(raises, priority, addressed)
    winner, yes_roles, had_tie = queue.apply_poll_batch(raises, priority)
    new_priority = list(panel_priority)
    if winner and (had_tie or addressed) and winner in new_priority:
        new_priority = [r for r in new_priority if r != winner] + [winner]
    return winner, yes_roles, had_tie, new_priority
