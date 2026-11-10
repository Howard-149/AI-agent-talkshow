"""Dialogue quality fixes for the Think & Action think prompt (all flags default OFF).

Think & Action speaks the selected panelist's think ``content`` directly; the worker draft is only
the fallback when there is no proposed reply. So the fixes that shape the worker prompt must also
reach the think prompt, or they never touch what the panel says. ``think_prompt_additions`` returns
the text appended to a panelist's think prompt; with every flag off it returns "" and the think
prompt is unchanged. The host's think gets nothing.

Order follows the worker prompt:
- TALKSHOW_PANEL_SCAFFOLD_FIX: when the latest public turn is the human's (host lines skipped, as in
  the worker), tell the panelist that their content should answer the human's point.
- TALKSHOW_UPTAKE_PLANNER (lite or full): same condition; add the planner's list of the human's
  figures, names, stance and addressee (same planner and plan text as the worker; always the lite
  instruction, because the think JSON has no place for the hidden [engages] line).
- TALKSHOW_DIALOGUE_LIBRARY: one move example from that library, rotating per spoken line of this
  panelist (the worker injects the catalog; the think prompt is long already).
- TALKSHOW_PANEL_CONCRETE: the one-fact instruction (same text as the worker).
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)

HEADER = 'About "content" (only if you speak or interrupt):'
RESPOND_TO_HUMAN = (
    "The human guest made the latest point. Make your content answer that point directly "
    "(you may also build on a panelist)."
)


def _is_panelist(data: Any, role: str) -> bool:
    return role not in {"human", data.scenario.turn_control.listen_role}


def library_example(library_id: str, role: str, index: int) -> str:
    """One move example of ``library_id`` for ``role`` ("" if the library is missing or empty)."""
    from agent.panel.dialogue_library import _moves_for_role, load_dialogue_library

    try:
        moves = _moves_for_role(load_dialogue_library(library_id), role)
    except FileNotFoundError:
        logger.warning("dialogue library %r not found; no style example in the think prompt", library_id)
        return ""
    if not moves:
        return ""
    move = moves[index % len(moves)]
    return f"Style example (do not copy its topic) - [{move.label}]:\n{move.example}"


async def think_prompt_additions(data: Any, role: str, history: Any) -> tuple[str, dict[str, Any]]:
    """(text to append to ``role``'s think prompt, log fields). ("", {}) when nothing applies.

    ``history`` is the transcript the think call sees (the visible prefix of the current line included).
    """
    from agent.panel.concrete import CONCRETE, concrete_instruction_enabled, dialogue_library_override
    from agent.panel.uptake import (
        await_uptake_plan,
        latest_human_unanswered,
        plan_block,
        scaffold_fix_enabled,
        uptake_planner_enabled,
    )

    if not _is_panelist(data, role):
        return "", {}
    scaffold = scaffold_fix_enabled()
    planner = uptake_planner_enabled()
    library = dialogue_library_override()
    concrete = concrete_instruction_enabled()
    if not (scaffold or planner or library or concrete):
        return "", {}

    parts: list[str] = []
    info: dict[str, Any] = {}
    human = None
    if scaffold or planner:
        human = latest_human_unanswered(history.lines, data.scenario.turn_control.listen_role)
    if human is not None and scaffold:
        parts.append(RESPOND_TO_HUMAN)
        info["respond_to_human"] = True
    if human is not None and planner:
        plan, waited = await await_uptake_plan(data, human)
        if plan:
            parts.append(plan_block(plan, lite=True, field="content").strip())
        info.update(plan_used=bool(plan), plan_wait_s=round(waited, 3))
    if library:
        spoken = sum(1 for line in history.lines if line.role_id == role)
        example = library_example(library, role, spoken)
        if example:
            parts.append(example)
            info["dialogue_library"] = library
    if concrete:
        parts.append(CONCRETE)
        info["concrete_instruction"] = True
    if not parts:
        return "", info
    return "\n\n" + HEADER + "\n" + "\n\n".join(parts), info
