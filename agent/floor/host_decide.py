"""LLM host picks who speaks next (show-graph phase 4).

One tool-calling request per moderation: the host chooses ``call_on_panelist``,
``give_floor_to_human`` or ``close_round`` and writes the line it says on the way
(spoken and appraised like any host line). Hand raises are inputs to the decision,
not the decision. A human who raised a hand waits at most one more panelist turn:
after that only ``give_floor_to_human`` is offered. Any failure returns None and the
caller falls back to the FIFO hand-raise queue.

``TALKSHOW_HOST_DECIDE=fifo`` turns the LLM host off.
"""

from __future__ import annotations

import logging
import os
import re
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Sequence

if TYPE_CHECKING:
    from agent.data import TalkShowData

logger = logging.getLogger(__name__)

CALL_ON = "call_on_panelist"
TO_HUMAN = "give_floor_to_human"
CLOSE = "close_round"


def llm_host_enabled() -> bool:
    return os.environ.get("TALKSHOW_HOST_DECIDE", "llm").strip().lower() != "fifo"


@dataclass(frozen=True)
class Raise:
    name: str
    raised: bool
    topic: str = ""


@dataclass(frozen=True)
class HostDecision:
    role: str  # panel role | "human" | "close"
    line: str
    tool: str


def build_tools(panel_names: Sequence[str], *, human_must_go: bool) -> list[dict[str, Any]]:
    line = {
        "type": "string",
        "description": "What you say out loud now: one short sentence.",
    }
    to_human = {
        "type": "function",
        "function": {
            "name": TO_HUMAN,
            "description": "Hand the floor back to the human guest.",
            "parameters": {
                "type": "object",
                "properties": {
                    "line": {
                        "type": "string",
                        "description": "One short sentence spoken to the guest, addressing them as you.",
                    }
                },
                "required": ["line"],
            },
        },
    }
    if human_must_go:
        return [to_human]
    return [
        {
            "type": "function",
            "function": {
                "name": CALL_ON,
                "description": "Give the floor to one panelist. Your line tees them up by name.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "panelist": {"type": "string", "enum": list(panel_names)},
                        "line": line,
                    },
                    "required": ["panelist", "line"],
                },
            },
        },
        to_human,
        {
            "type": "function",
            "function": {
                "name": CLOSE,
                "description": "Wrap up this round when the topic is exhausted.",
                "parameters": {"type": "object", "properties": {"line": line}, "required": ["line"]},
            },
        },
    ]


def render_decide_prompt(
    *,
    host: str,
    raises: Sequence[Raise],
    human: Raise,
    turns: dict[str, int],
    last_speaker: str,
    human_must_go: bool,
    mood: str = "",
) -> str:
    """Pure prompt text (tested without a model)."""
    def raise_line(r: Raise) -> str:
        if not r.raised:
            return f"- {r.name}: not raised"
        return f"- {r.name}: RAISED" + (f' — wants to talk about "{r.topic}"' if r.topic else "")

    status = "\n".join([*(raise_line(r) for r in raises), raise_line(human)])
    turn_bits = ", ".join(f"{n} ×{k}" for n, k in turns.items()) or "nobody yet"
    mood_block = f"\n{mood.strip()}\n" if mood.strip() else ""
    if human_must_go:
        task = (
            f"The human guest raised a hand and has already waited a turn. Give them the floor now "
            f"with {TO_HUMAN}; your line invites them in and may tie back to the last point."
        )
    else:
        task = f"""Decide who speaks next and say your line. Call exactly one tool.
- Follow the live thread: call on whoever can best answer or challenge the last point. A raised
  hand with a fitting topic is a good reason, but you may call on someone who did not raise.
- Keep it balanced: avoid the same panelist twice in a row unless they were asked something directly.
- If the human guest raised a hand, you may let one panelist finish the current point first,
  then use {TO_HUMAN}.
- Use {CLOSE} only when the topic is exhausted."""
    return f"""You are {host}, the moderator. The last line was spoken by {last_speaker or "nobody"}.

Hand raises:
{status}
Panel turns since the human guest last spoke: {turn_bits}.
{mood_block}
{task}

Your line is spoken aloud before they talk: ONE short sentence (under 25 words) that names the
panelist you hand to, or speaks to the human guest as "you", and connects to what was just said.
Do not answer the question yourself, do not mention hand raises, no stage directions."""


_TAG_RE = re.compile(r"\[[a-z_]+(?::[^\]]*)?\]:?", re.I)


def parse_decision(
    calls: Sequence[tuple[str, dict[str, Any]]],
    *,
    name_to_role: dict[str, str],
    human_must_go: bool,
) -> tuple[HostDecision | None, str]:
    """First valid tool call → decision, or (None, why) for the fallback log."""
    if not calls:
        return None, "no_tool_call"
    tool, args = calls[0]
    line = _TAG_RE.sub("", str(args.get("line") or "")).strip().strip('"').strip()
    if len(line) < 4:
        return None, "empty_line"
    if tool == TO_HUMAN:
        return HostDecision("human", line, tool), ""
    if human_must_go:
        return None, f"human_must_go_but_{tool or 'none'}"
    if tool == CLOSE:
        return HostDecision("close", line, tool), ""
    if tool == CALL_ON:
        name = str(args.get("panelist") or "").strip()
        role = name_to_role.get(name.lower())
        if role is None:
            return None, f"unknown_panelist:{name[:20]}"
        if name.lower() not in line.lower():
            line = f"{name}, {line[0].lower()}{line[1:]}"
        return HostDecision(role, line, tool), ""
    return None, f"unknown_tool:{tool[:30]}"


def panel_turns_since_human(data: TalkShowData, panel_roles: Sequence[str]) -> tuple[dict[str, int], str]:
    """Substantive panel lines since the human last spoke, and who spoke last."""
    from agent.config import load_persona_name

    counts: dict[str, int] = {}
    last = ""
    for ln in reversed(data.show_history.lines):
        if ln.procedural:
            continue
        if not last:
            last = "the human guest" if ln.role_id == "human" else load_persona_name(ln.role_id)
        if ln.role_id == "human":
            break
        if ln.role_id in panel_roles:
            name = load_persona_name(ln.role_id)
            counts[name] = counts.get(name, 0) + 1
    return counts, last


async def host_decide(
    data: TalkShowData,
    *,
    panel_roles: Sequence[str],
    human_must_go: bool,
    perspective: str = "host",
) -> HostDecision | None:
    from agent.config import load_persona_instructions, load_persona_name
    from agent.emotion import mood_prompt_block

    host = load_persona_name("host")
    names = {r: load_persona_name(r) for r in panel_roles}
    snap = data.hand_raise_queue.snapshot(list(panel_roles))
    raises = [Raise(names[r], snap[r].raised, snap[r].topic) for r in panel_roles]
    human_entry = data.hand_raise_queue.get("human")
    human = Raise("Human guest", human_entry is not None, human_entry.topic if human_entry else "")
    turns, last = panel_turns_since_human(data, panel_roles)
    prompt = render_decide_prompt(
        host=host, raises=raises, human=human, turns=turns, last_speaker=last,
        human_must_go=human_must_go, mood=mood_prompt_block(data, "host"),
    )
    tools = build_tools(list(names.values()), human_must_go=human_must_go)
    t0 = time.monotonic()
    calls: list[tuple[str, dict[str, Any]]] = []
    content = ""
    error = ""
    try:
        content, calls = await data.runtime.gemma_client.complete_tools(
            prompt,
            tools=tools,
            system_prompt=load_persona_instructions("host", panel_mode=True),
            history_messages=data.show_history.prior_messages(perspective=perspective),
        )
        decision, why = parse_decision(
            calls,
            name_to_role={n.lower(): r for r, n in names.items()},
            human_must_go=human_must_go,
        )
    except Exception as exc:
        logger.exception("host_decide request failed")
        decision, why, error = None, "request_failed", f"{type(exc).__name__}: {exc}"[:200]
    latency = time.monotonic() - t0
    if data.turn_log is not None:
        data.turn_log.log(
            "host_decide",
            role=decision.role if decision else None,
            tool=decision.tool if decision else None,
            line=decision.line[:300] if decision else None,
            fallback=why or None,
            error=error or None,
            human_raised=human.raised,
            human_must_go=human_must_go,
            raised=[r.name for r in raises if r.raised],
            turns=turns,
            calls=[{"name": n, "args": a} for n, a in calls][:2],
            content=content[:200] or None,
            latency_s=round(latency, 3),
            room=data.room_name,
        )
    logger.info(
        "host_decide → %s (%s) %.2fs fallback=%s",
        decision.role if decision else None,
        decision.tool if decision else "-",
        latency,
        why or "-",
    )
    return decision
