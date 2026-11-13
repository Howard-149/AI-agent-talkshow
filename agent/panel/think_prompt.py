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
- TALKSHOW_FACT_PACK: the checked-facts block and its "only from these facts" rule (agent.panel.fact_pack.pack_block,
  the worker's function, on the transcript this think call sees).
- TALKSHOW_PANEL_CONCRETE: the one-fact instruction (same text as the worker; pointing at the facts when a fact
  block is present).

Context variants of the panelist's think prompt (each default OFF; agent/interaction/floor_interaction._think_and_act):
- TALKSHOW_THINK_LEAN: on a free floor, the latest line is not repeated when it is the last transcript line
  (LEAN_SAME_LINE), the occupied-floor and backchannel guidance is left out (backchannels fixed to []), and the
  canned procedural host lines (recorded as such in the show history) are dropped from the transcript messages.
- TALKSHOW_THINK_PERSONA_NEAR: a persona reminder for writing the content (persona_reminder), right before the
  "About content" section (or before the JSON schema when there is none). With TALKSHOW_PERSONA_LIBRARY the library
  persona's block (style, moves, avoid) takes that place instead (near_block).
- TALKSHOW_THINK_FOCUS: on a free floor, a "Respond to now" block (focus_block): who spoke last, whom they
  addressed by name, the uptake planner's points if it ran, and the rule FOCUS_RULE.
"""

from __future__ import annotations

import logging
import os
import re
from typing import Any

logger = logging.getLogger(__name__)

HEADER = 'About "content" (only if you speak or interrupt):'
RESPOND_TO_HUMAN = (
    "The human guest made the latest point. Make your content answer that point directly "
    "(you may also build on a panelist)."
)


LEAN_SAME_LINE = "Latest line: see the transcript (its last line)."
FOCUS_HEADER = "Respond to now:"
FOCUS_RULE = ("If the latest question was addressed to someone else by name, choose silent unless you must correct a "
              "factual error.")


def _flag(name: str) -> bool:
    return os.environ.get(name, "0").strip().lower() in ("1", "true", "yes", "on")


def think_lean_enabled() -> bool:
    return _flag("TALKSHOW_THINK_LEAN")


def persona_near_enabled() -> bool:
    return _flag("TALKSHOW_THINK_PERSONA_NEAR")


def think_focus_enabled() -> bool:
    return _flag("TALKSHOW_THINK_FOCUS")


def style_rules(role: str) -> str:
    """The last 2 sentences of ``role``'s persona card (the card's own style rules; follows the persona swap)."""
    from agent.config import load_persona_personality

    card = " ".join(load_persona_personality(role).split())
    sentences = [x for x in re.split(r"(?<=[.!?])\s+", card) if x]
    return " ".join(sentences[-2:])


def persona_reminder(role: str) -> str:
    """"Write the content as <Name>: <style rules>" (TALKSHOW_THINK_PERSONA_NEAR)."""
    from agent.config import load_persona_name

    return f"Write the content as {load_persona_name(role)}: {style_rules(role)}"


def near_block(role: str) -> str:
    """The persona text right before the line is written: the library persona's block (TALKSHOW_PERSONA_LIBRARY,
    agent/persona_library.render_near; supersedes the reminder), else persona_reminder (TALKSHOW_THINK_PERSONA_NEAR),
    else ""."""
    from agent.persona_library import LIBRARY_SEATS, library_enabled, persona_for_seat, render_near

    if library_enabled() and role in LIBRARY_SEATS:
        from agent.config import content_seat, load_persona_name

        return render_near(persona_for_seat(content_seat(role)), load_persona_name(role))
    return persona_reminder(role) if persona_near_enabled() else ""


def lean_history(history: Any) -> Any:
    """The transcript without the canned procedural host lines (HistoryLine.procedural)."""
    from agent.show.show_history import ShowHistory

    return ShowHistory(lines=[line for line in history.lines if not line.procedural])


def same_as_last_line(history: Any, interval_text: str, visible_text: str) -> bool:
    """The latest interval is the whole latest line and that line is already the transcript's last line."""
    if not history.lines:
        return False
    last = history.lines[-1].text.strip()
    return bool(last) and interval_text.strip() == visible_text.strip() == last


async def focus_block(data: Any, role: str, history: Any, speaker: str, visible_text: str) -> str:
    """"Respond to now" (TALKSHOW_THINK_FOCUS): last speaker, the panelist they named, the planner's points."""
    from agent.config import load_persona_name
    from agent.floor.addressee import addressed_role
    from agent.show.show_history import human_label

    def who(r: str) -> str:
        return f"{human_label()} (human)" if r == "human" else f"{load_persona_name(r)} ({r})"

    panel = [r for r in ("commentator", "guest") if r != data.scenario.turn_control.listen_role]
    target = addressed_role(visible_text, {r: load_persona_name(r) for r in panel if r != speaker})
    if target is None:
        named = "nobody by name"
    elif target == role:
        named = f"you ({load_persona_name(role)})"
    else:
        named = who(target)
    lines = [FOCUS_HEADER, f"- Last speaker: {who(speaker)}.", f"- They addressed: {named}."]
    from agent.panel.uptake import await_uptake_plan, latest_human_unanswered, uptake_planner_enabled

    if speaker == "human" and uptake_planner_enabled():
        human = latest_human_unanswered(history.lines, data.scenario.turn_control.listen_role)
        if human is not None:
            plan, _waited = await await_uptake_plan(data, human)  # the same plan task the additions use
            if plan.strip():
                lines.append("- Their points:\n" + "\n".join("  " + x.strip() for x in plan.strip().splitlines()
                                                             if x.strip()))
    lines.append(FOCUS_RULE)
    return "\n".join(lines)


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
    from agent.panel.fact_pack import CONCRETE_WITH_PACK, fact_pack_enabled, pack_block
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
    facts = fact_pack_enabled()
    if not (scaffold or planner or library or concrete or facts):
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
    block = ""
    if facts:
        # Offline builders know the topic id (data.fact_pack_topic); live, the pack resolves it as for the worker.
        block, fact_info = pack_block(data, getattr(data, "fact_pack_topic", None), history=history)
        if block:
            parts.append(block.strip())
        info.update(fact_info)
    if concrete:
        parts.append(CONCRETE_WITH_PACK if block else CONCRETE)
        info["concrete_instruction"] = True
    if not parts:
        return "", info
    return "\n\n" + HEADER + "\n" + "\n\n".join(parts), info
