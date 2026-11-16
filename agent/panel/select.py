"""Several think candidates per panelist line, the best one picked by the decision model (TALKSHOW_PANEL_SELECT=<n>,
default off).

With the flag set to n >= 2, a panelist's think call on a free floor is sampled n times. The first sample is the
usual call (same prompt and sampling as without the flag), so the speak / silent decision does not change. The other
n - 1 samples use the same prompt at SAMPLE_TEMPERATURE, so they say something different. If the first sample chose
speak, the candidate lines are its content plus the content of every other sample that also chose speak
(``candidate_contents``).

When that panelist is selected to speak, one decision model call (Jev, agent/floor/jev_floor.py) picks the best
candidate (``pick_best``): a choice question over the candidates, with the recent transcript and a one-line persona
of the speaker as the state. On a missing key, an API error or an invalid answer the first candidate is kept.

Used by the live floor (agent/interaction/floor_interaction.py: ``_think_and_act`` samples, ``finish_proposed_reply``
picks) and by the offline full-show benchmark (eval/bench/show.py), with the same functions.
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
from typing import Any

logger = logging.getLogger(__name__)

SAMPLE_TEMPERATURE = 0.8  # the extra samples (the first sample keeps the think call's own temperature)
MAX_N = 8
INSTRUCTIONS = (
    "These are candidate lines {name} could say next on a live talk show. Which line would make the best live "
    "talk-show moment as {name}? Prefer the line that answers what was just said, adds something new rather than "
    "repeating {name}'s earlier points, sounds like {name}, and is short and punchy."
)


def select_n() -> int:
    """Samples per free-floor panelist think call (1 = off)."""
    try:
        n = int(os.environ.get("TALKSHOW_PANEL_SELECT", "0").strip() or 0)
    except ValueError:
        return 1
    return min(n, MAX_N) if n >= 2 else 1


def select_enabled() -> bool:
    return select_n() > 1


def candidate_contents(first: Any, extras: list[Any]) -> list[str]:
    """The candidate lines: the first sample's content (only if it chose speak), then every other sample that chose
    speak with a different content. ``first`` and ``extras`` are parsed think Actions (or None)."""
    if first is None or first.decision != "speak" or not (first.content or "").strip():
        return []
    out = [first.content]
    seen = {first.content.strip()}
    for a in extras:
        if a is None or a.decision != "speak" or not (a.content or "").strip():
            continue
        if a.content.strip() in seen:
            continue
        seen.add(a.content.strip())
        out.append(a.content)
    return out


def persona_line(role: str) -> str:
    """One line on who the speaker is: the library persona (role on the show, length, tone, register), else the first
    sentence of the persona card."""
    import re

    from agent.config import content_seat, load_persona_name
    from agent.persona_library import LIBRARY_SEATS, library_enabled, persona_for_seat

    name = load_persona_name(role)
    if library_enabled() and role in LIBRARY_SEATS:
        p = persona_for_seat(content_seat(role))
        s = p.style
        return f"{name}: {p.role} ({s['length']}, {s['tone']}, {s['register']})."
    from agent.panel.panel_prompts import panelist_personality_block

    card = " ".join(panelist_personality_block(role).split())
    first = re.split(r"(?<=[.!?])\s+", card)[0] if card else ""
    return f"{name}: {first}" if first else name


def recent_transcript(history: Any, max_lines: int = 12) -> str:
    """The public transcript (procedural host lines left out), latest last."""
    if history is None:
        return ""
    return "\n".join(f"{line.speaker}: {line.text}" for line in history.lines[-max_lines:] if not line.procedural)


def option_id(i: int) -> str:
    return f"line_{i + 1}"


def build_request(transcript: str, name: str, persona: str, candidates: list[str]) -> dict[str, Any]:
    """The Jev request: state = transcript + speaker persona; one choice question over the candidate lines."""
    from agent.floor.jev_floor import MODEL

    state = f"Transcript (latest last):\n{transcript}\n\nNext speaker: {persona}"
    criteria = {option_id(i): f'{name} says: "{c}"' for i, c in enumerate(candidates)}
    return {"model": MODEL, "state": state,
            "questions": {"best": {"type": "choice", "instructions": INSTRUCTIONS.format(name=name),
                                   "criteria": criteria}}}


async def pick_best(transcript: str, role: str, candidates: list[str]) -> tuple[int, dict[str, Any]]:
    """(index of the chosen candidate, log fields). Index 0 on a single candidate, a missing key or any failure."""
    from agent.config import load_persona_name
    from agent.floor import jev_floor

    info: dict[str, Any] = {"select_n": len(candidates)}
    if len(candidates) < 2:
        return 0, {**info, "select": "single"}
    key = os.environ.get("TYPESAFE_API_KEY", "").strip()
    if not key:
        return 0, {**info, "select": "skipped"}
    body = build_request(transcript, load_persona_name(role), persona_line(role), candidates)
    t = time.monotonic()
    try:
        resp = await asyncio.to_thread(jev_floor._post, body, key)
        ans = resp["answers"]["best"]
        choice, conf = ans["choice"], float(ans.get("confidence", 0.0))
    except Exception as e:  # noqa: BLE001  (any failure keeps the first candidate)
        logger.warning("jev line pick failed: %s", type(e).__name__)
        return 0, {**info, "select": "error", "select_error": type(e).__name__,
                   "select_latency_s": round(time.monotonic() - t, 3)}
    info.update(select_latency_s=round(time.monotonic() - t, 3), select_conf=round(conf, 3),
                select_probs=ans.get("probabilities"))
    ids = [option_id(i) for i in range(len(candidates))]
    if choice not in ids:
        return 0, {**info, "select": "invalid"}
    return ids.index(choice), {**info, "select": "ok", "select_choice": ids.index(choice)}
