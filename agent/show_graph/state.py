"""ShowState — what the show graph carries between nodes for one show beat."""

from __future__ import annotations

import operator
from dataclasses import dataclass
from typing import Annotated, Any, Literal, TypedDict

Entry = Literal["human", "moderate", "panel"]

LineKind = Literal[
    "host_reply",  # host answers the human (graph entry "human")
    "open_floor",  # canned: "Floor's open — anyone want to jump in?"
    "intro",  # canned: "Amy, the floor is yours."
    "direct_call",  # host names a panelist when nobody raised (LLM line)
    "panelist",  # panelist line (LLM)
    "grant_human",  # canned: hand the mic back to the human
    "close",  # canned: wrap the panel round
]


@dataclass(frozen=True)
class FloorPick:
    """Outcome of a hand-raise round: who speaks next and why."""

    role: str | None
    reason: str
    # Set when nobody raised and the host calls on someone directly: the line the
    # graph must speak (as a host line, appraised) before that panelist.
    host_line: str | None = None


@dataclass(frozen=True)
class PanelistLine:
    """A generated (not yet spoken) panelist line."""

    role: str
    text: str
    next_role: str  # validated [next] tag (host when missing/unknown)
    emotion: str


class LineSpec(TypedDict, total=False):
    """One line the graph is about to speak."""

    role: str
    text: str
    step: str
    kind: LineKind
    # Canned host lines are spoken but not appraised, and are skipped when
    # working out who a later line replies to.
    procedural: bool
    next_tag: str  # panelist [next] tag, applied after the line is spoken
    target: str  # role the line hands the floor to (direct_call / intro)
    reason: str  # floor-grant reason (grant_human)
    texts: dict[str, str]  # pre-localized delivery texts (canned lines)


class HumanEvent(TypedDict, total=False):
    """A finished human turn: transcript + the host's draft reply (Gemma)."""

    heard: str
    parsed: Any  # agent.adapters.response_parser.ParsedTurn
    raw: str | None
    model_latency_s: float
    done_event: str  # gemma_stt_done | ghost_human_done


class AppraiseJob(TypedDict):
    utt: Any  # agent.emotion.appraisal.Utterance
    listener: str


class ShowState(TypedDict, total=False):
    entry: Entry
    trigger: str
    panel_roles: list[str]
    listen_role: str

    human_event: HumanEvent | None
    line: LineSpec | None
    # Utterances the next prepare_speak fans out to listeners besides the line
    # itself (the human's line on a human turn).
    extra_utterances: list[Any]
    after: str  # node to route to once the current line has been spoken

    next_role: str  # panelist the floor goes to next
    grant_reason: str
    skip_open_floor: bool
    skip_intro: bool

    turn_idx: int  # panelist lines spoken this beat
    max_turns: int
    moderations: int  # hand-raise rounds this beat
    max_moderations: int
    spoken_roles: list[str]

    outcome: str  # "human" | "close" | "" — how the beat ended
    stopped: bool

    # Mirrors of LiveKit-side state for inspection / future checkpointing.
    role_emotion: dict[str, str]
    role_pad: dict[str, tuple[float, float, float]]

    # Fan-in channels written by parallel branches (Send).
    spoken: Annotated[list[str], operator.add]
    appraisals: Annotated[list[dict[str, Any]], operator.add]


def base_state(
    *,
    entry: Entry = "panel",
    trigger: str,
    panel_roles: list[str],
    listen_role: str,
    human_event: HumanEvent | None = None,
    skip_open_floor: bool = False,
    max_turns: int = 12,
    max_moderations: int = 8,
    spoken_roles: list[str] | None = None,
    role_emotion: dict[str, str] | None = None,
    role_pad: dict[str, tuple[float, float, float]] | None = None,
    **_legacy: Any,
) -> ShowState:
    return {
        "entry": entry,
        "trigger": trigger,
        "panel_roles": list(panel_roles),
        "listen_role": listen_role,
        "human_event": human_event,
        "line": None,
        "extra_utterances": [],
        "after": "",
        "next_role": "",
        "grant_reason": "",
        "skip_open_floor": skip_open_floor,
        "skip_intro": False,
        "turn_idx": 0,
        "max_turns": max_turns,
        "moderations": 0,
        "max_moderations": max_moderations,
        "spoken_roles": list(spoken_roles or ()),
        "outcome": "",
        "stopped": False,
        "role_emotion": dict(role_emotion or {}),
        "role_pad": dict(role_pad or {}),
        "spoken": [],
        "appraisals": [],
    }
