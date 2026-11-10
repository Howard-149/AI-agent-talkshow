"""Panel uptake of the human's concrete points (all flags default OFF).

Our responsiveness tests showed panelists echo a figure when
pointed at it but rarely reason with it, and that the panel scaffold steers them away from a human
who just spoke. Two prompt-side fixes, applied only to the FIRST panel line after a human turn:

TALKSHOW_UPTAKE_PLANNER=1|lite a planner call (same Gemma) extracts the human turn's figures,
                               names, stance and addressee; the panelist must output a hidden
                               ``[engages]: <item> — <how it changes the argument>`` line (never spoken)
                               and engage that item. The planner starts when the human turn is
                               committed, so it overlaps the work that follows (recording the line;
                               the host reply in the older hand-raise modes).
                               ``lite`` (planner-lite): same plan in the prompt, NO [engages] output line —
                               keeps the uptake gain without decoding ~120 extra chars per line.
TALKSHOW_PANEL_SCAFFOLD_FIX=1  neutralise the "human floor FROZEN / NOT your conversation partner /
                               respond to the latest speaker" steering in the panelist prompt.

Think & Action speaks the panelist's think ``content`` and drafts with the worker only as a fallback.
The think prompt never contained the scaffold phrases above (its system prompt is the persona
personality and its history is the spoken lines), so there the scaffold fix becomes a positive
instruction to answer the human's point, and the planner adds the same plan (agent.panel.think_prompt).

(The floor fix is separate: TALKSHOW_ADDRESSEE_FLOOR, agent/floor/addressee.py.)
"""

from __future__ import annotations

import asyncio
import logging
import os
import re
import time
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from agent.data import TalkShowData

logger = logging.getLogger(__name__)

PLAN_WAIT_TIMEOUT_S = 3.0


def _flag(name: str) -> bool:
    return os.environ.get(name, "0").strip().lower() in ("1", "true", "yes", "on")


def _planner_mode() -> str:
    raw = os.environ.get("TALKSHOW_UPTAKE_PLANNER", "0").strip().lower()
    if raw == "lite":
        return "lite"
    return "full" if raw in ("1", "true", "yes", "on", "full") else ""


def uptake_planner_enabled() -> bool:
    return _planner_mode() != ""


def uptake_planner_lite() -> bool:
    return _planner_mode() == "lite"


def scaffold_fix_enabled() -> bool:
    return _flag("TALKSHOW_PANEL_SCAFFOLD_FIX")


def first_answer_to_human(data: TalkShowData) -> str | None:
    """Latest human text if no panelist has spoken since it (host lines skipped), else None."""
    return latest_human_unanswered(data.show_history.lines)


def latest_human_unanswered(lines: list[Any], host_role: str = "host") -> str | None:
    """Latest human text in ``lines`` if no panelist has spoken since it (host lines skipped), else None."""
    for line in reversed(lines):
        if line.role_id == host_role:
            continue
        return line.text if line.role_id == "human" else None
    return None


# --- planner ------------------------------------------------------------------------

PLANNER_SYSTEM = "You extract the concrete commitments in one line a guest said on a talk show. Be literal and short."


def planner_user_prompt(human_text: str) -> str:
    return (
        f'Guest line: "{human_text}"\n\n'
        "List up to 4 concrete items the guest committed to: every figure (with its unit), named people/places/"
        'organisations, the stance they took, and who they addressed. One item per line, starting with "- ". '
        "No commentary."
    )


def plan_block(plan: str, lite: bool | None = None, field: str = "[reply]") -> str:
    """Plan text for a panel prompt. ``field`` names where the spoken words go: ``[reply]`` in the
    worker prompt, ``content`` in the Think & Action think prompt (always lite there: its JSON has no
    place for the hidden [engages] line)."""
    if lite is None:
        lite = uptake_planner_lite()
    head = "\n\nThe human guest's concrete points (extracted from their latest line):\n" + plan.strip() + "\n"
    if lite:
        return head + f"Make your {field} engage the item that matters most for your argument explicitly, and let it change what you argue."
    return head + (
        "Before [reply], output one line:\n[engages]: <the item you engage> — <how it changes your argument>\n"
        "Then make your [reply] engage that item explicitly."
    )


_ENGAGES_RE = re.compile(r"(?im)^\s*\[engages\]\s*:.*$")


def strip_engages(raw: str) -> str:
    """The planning line is never spoken (it can appear before or after [reply])."""
    return _ENGAGES_RE.sub("", raw or "")


def engages_chars(raw: str) -> int:
    """Length of the hidden planning line(s) — its decode time is the planner's per-line overhead."""
    return sum(len(m.group(0)) for m in _ENGAGES_RE.finditer(raw or ""))


def start_uptake_plan(data: TalkShowData, human_text: str) -> None:
    """Kick off the planner as soon as the human turn is committed (runs during the host reply)."""
    if not uptake_planner_enabled() or not (human_text or "").strip():
        return
    if getattr(data, "uptake_plan_for", None) == human_text:
        return
    data.uptake_plan_for = human_text  # type: ignore[attr-defined]
    data.uptake_plan_task = asyncio.create_task(_make_plan(data, human_text))  # type: ignore[attr-defined]


async def _make_plan(data: TalkShowData, human_text: str) -> str:
    t0 = time.monotonic()
    try:
        plan = await data.runtime.gemma_client.complete_text(
            planner_user_prompt(human_text),
            system_prompt=PLANNER_SYSTEM,
            history_messages=None,
            raw=True,
            temperature=0.0,
            max_tokens=160,
        )
    except Exception:
        logger.exception("uptake planner failed")
        plan = ""
    turn_log = getattr(data, "turn_log", None)
    if turn_log is not None:
        turn_log.log("uptake_plan", room=getattr(data, "room_name", ""), plan=plan[:500],
                     plan_latency_s=round(time.monotonic() - t0, 3))
    return plan.strip()


async def await_uptake_plan(data: TalkShowData, human_text: str) -> tuple[str, float]:
    """(plan, seconds the panel line waited for it). Empty plan if missing / late / for another turn."""
    task = getattr(data, "uptake_plan_task", None)
    if task is None or getattr(data, "uptake_plan_for", None) != human_text:
        start_uptake_plan(data, human_text)  # human turn arrived by another path: plan now
        task = getattr(data, "uptake_plan_task", None)
        if task is None:
            return "", 0.0
    t0 = time.monotonic()
    try:
        plan = await asyncio.wait_for(asyncio.shield(task), timeout=PLAN_WAIT_TIMEOUT_S)
    except asyncio.TimeoutError:
        logger.warning("uptake planner not ready after %.1fs — drafting without a plan", PLAN_WAIT_TIMEOUT_S)
        plan = ""
    return plan, time.monotonic() - t0


# --- scaffold fix (live wording: the host's acknowledgement sits between human and panel) ---

SCAFFOLD_FIX: list[tuple[str, str]] = [
    (r"- Who just spoke: the host or an earlier speaker \(see transcript\)",
     "- Who just spoke: the human guest (the host only acknowledged them; see transcript)"),
    (r"- Human guest floor: FROZEN — the human is listening; they are NOT your conversation partner this beat\.\n", ""),
    (r"; address whoever last spoke in the transcript\.", "."),
    (r"- If the human is FROZEN, talking TO them \(especially questions\) is unnatural — they cannot answer yet\.\n", ""),
    (r"- If the human is OPEN, you may invite them back\.\n", ""),
    (r"respond to the host or an earlier speaker \(see transcript\), (\w+), or the guest's take as fits the thread\.",
     r"respond to the human guest's latest point (you may also build on \1)."),
    (r"The panel is mid-discussion — respond to the latest speaker or the thread under debate;",
     "The human guest just made a point — respond to it;"),
    (r"- If the human's floor is FROZEN, do not ask them direct questions — they are listening\.",
     "- Do not ask the human direct questions."),
]


def apply_scaffold_fix(text: str) -> tuple[str, int]:
    n = 0
    for pat, rep in SCAFFOLD_FIX:
        text, k = re.subn(pat, rep, text)
        n += k
    return text, n


def adapt_panel_prompt(
    data: TalkShowData, system_prompt: str, prompt: str
) -> tuple[str, str, dict[str, Any]]:
    """Synchronous part of the uptake fix (scaffold fix). Returns (system, user, log fields)."""
    info: dict[str, Any] = {}
    if scaffold_fix_enabled() and first_answer_to_human(data) is not None:
        system_prompt, n1 = apply_scaffold_fix(system_prompt)
        prompt, n2 = apply_scaffold_fix(prompt)
        info["scaffold_fix_subs"] = n1 + n2
    return system_prompt, prompt, info
