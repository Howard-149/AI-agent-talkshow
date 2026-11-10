"""Think & Action schema: listener actions, backchannels, floor outcomes, parser."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Literal

INTENT_TYPES = frozenset(
    {
        "acknowledge",
        "support",
        "challenge",
        "clarify",
        "correct",
        "contribute",
        "answer",
        "facilitate",
    }
)
BACKCHANNEL_INTENT_TYPES = frozenset({"acknowledge", "support", "surprise", "encourage"})
WORD_RE = re.compile(r"\S+")


@dataclass(frozen=True)
class CommunicativeIntent:
    type: str
    goal: str
    target: str | None


@dataclass(frozen=True)
class Backchannel:
    intent: CommunicativeIntent
    content: str
    after_word_index: int
    target_segment_index: int | None = None


@dataclass(frozen=True)
class Action:
    decision: Literal["silent", "interrupt", "speak"]
    intent: CommunicativeIntent | None
    content: str | None
    backchannels: tuple[Backchannel, ...] = ()


@dataclass(frozen=True)
class ThinkAction:
    think: str
    action: Action


@dataclass(frozen=True)
class FloorOutcome:
    decision: Literal["continue", "interrupt", "speak", "host_fallback"]
    selected_role: str | None = None
    reason: str = ""
    reason_code: str = ""
    host_strategy: str | None = None
    host_target: str | None = None
    host_instruction: str | None = None


HOST_STRATEGIES = frozenset({"invite", "summarize", "contribute", "reframe", "close"})


def _json_object(raw: str) -> dict[str, Any] | None:
    start = raw.find("{")
    end = raw.rfind("}")
    if start < 0 or end < start:
        return None
    try:
        value = json.loads(raw[start : end + 1])
    except (json.JSONDecodeError, TypeError):
        return None
    return value if isinstance(value, dict) else None


def _intent_from_json(
    value: Any, allowed_types: frozenset[str] = INTENT_TYPES
) -> CommunicativeIntent | None:
    if not isinstance(value, dict):
        return None
    intent_type = value.get("type")
    goal = value.get("goal")
    target = value.get("target")
    if intent_type not in allowed_types or not isinstance(goal, str) or not goal.strip():
        return None
    if target is not None and not isinstance(target, str):
        return None
    return CommunicativeIntent(intent_type, goal.strip(), target.strip() if target else None)


def parse_think_action(
    raw: str, *, floor_state: Literal["occupied", "free"], interval_text: str,
    allow_backchannels: bool | None = None,
) -> ThinkAction:
    """Parse and validate one Think & Action JSON response; invalid actions fail silent."""
    payload = _json_object(raw)
    if payload is None:
        return ThinkAction("", Action("silent", None, None))

    think = payload.get("think", "")
    if not isinstance(think, str):
        think = ""
    action_data = payload.get("action")
    if not isinstance(action_data, dict):
        return ThinkAction(think.strip(), Action("silent", None, None))

    decision = action_data.get("decision")
    intent = _intent_from_json(action_data.get("intent"))
    content = action_data.get("content")
    if content is not None and not isinstance(content, str):
        content = None
    if isinstance(content, str):
        content = content.strip() or None

    backchannels: list[Backchannel] = []
    raw_backchannels = action_data.get("backchannels", [])
    if not isinstance(raw_backchannels, list):
        raw_backchannels = []
    word_count = len(WORD_RE.findall(interval_text))
    for item in raw_backchannels:
        if not isinstance(item, dict):
            continue
        bc_intent = _intent_from_json(item.get("intent"), BACKCHANNEL_INTENT_TYPES)
        bc_content = item.get("content")
        word_index = item.get("after_word_index")
        if (
            bc_intent is None
            or not isinstance(bc_content, str)
            or not bc_content.strip()
            or len(WORD_RE.findall(bc_content)) > 3
            or bc_intent.type not in BACKCHANNEL_INTENT_TYPES
            or isinstance(word_index, bool)
            or not isinstance(word_index, int)
            or not 1 <= word_index <= word_count
        ):
            continue
        backchannels.append(Backchannel(bc_intent, bc_content.strip(), word_index))
        break  # at most one candidate per listener and interval

    # The final interval is still spoken with an occupied floor, even though
    # the main-floor decision applies to the free floor after it ends.
    if allow_backchannels is None:
        allow_backchannels = floor_state == "occupied"
    if decision == "silent":
        valid_backchannels = backchannels if allow_backchannels else []
        return ThinkAction(think.strip(), Action("silent", None, None, tuple(valid_backchannels)))
    if decision not in {"interrupt", "speak"} or intent is None or content is None:
        return ThinkAction(think.strip(), Action("silent", None, None))
    if (decision == "interrupt") != (floor_state == "occupied"):
        return ThinkAction(think.strip(), Action("silent", None, None))

    valid_backchannels = backchannels if allow_backchannels else []
    return ThinkAction(
        think.strip(),
        Action(decision, intent, content, tuple(valid_backchannels)),
    )


def insert_backchannels(
    text: str,
    events: list[tuple[str, Backchannel]],
) -> str:
    """Insert scheduled events after 1-based word indexes, preserving original spacing."""
    spans = list(WORD_RE.finditer(text))
    by_word: dict[int, list[tuple[str, Backchannel]]] = {}
    for item in events:
        role, event = item
        if 1 <= event.after_word_index <= len(spans):
            by_word.setdefault(event.after_word_index, []).append((role, event))
    output: list[str] = []
    cursor = 0
    for word_index, span in enumerate(spans, start=1):
        output.append(text[cursor : span.end()])
        cursor = span.end()
        for role, event in by_word.get(word_index, []):
            output.append(f" [{role}: {event.content}]")
    output.append(text[cursor:])
    return "".join(output)
