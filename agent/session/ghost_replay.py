"""Prefix replay for controlled ghost experiments (TALKSHOW_GHOST_REPLAY=1).

Every condition of a minimal-pair experiment must share the exact same show state up
to the edited human turn, so only the lines after it vary. A ghost turn may carry a
``prefix`` (logged earlier as ``history_snapshot``); it replaces the transcript and
the per-role mood state just before the turn is handled. The live opening lines of
the replay session (welcome, floor grant) are discarded with the old transcript.

``persona_swap`` in the prefix gives each panelist the other's personality traits
(names, seats and voices unchanged) for the rest of the session. ``flags`` sets
dialogue-quality flags (REPLAY_FLAGS) for this session only.
"""

from __future__ import annotations

import logging
import os
import time
from typing import TYPE_CHECKING, Any

from agent.show.show_history import HistoryLine

if TYPE_CHECKING:
    from agent.data import TalkShowData

logger = logging.getLogger(__name__)

# Flags a replay prefix may set for its session (the dialogue-quality fixes).
REPLAY_FLAGS = (
    "TALKSHOW_UPTAKE_PLANNER",
    "TALKSHOW_PANEL_SCAFFOLD_FIX",
    "TALKSHOW_ADDRESSEE_FLOOR",
    "TALKSHOW_DIALOGUE_LIBRARY",
    "TALKSHOW_PANEL_CONCRETE",
    "TALKSHOW_PANEL_BEST_OF",
)


def snapshot_state(data: TalkShowData) -> dict[str, Any]:
    """Everything the next prompts read from session state (history + mood)."""
    return {
        "lines": [
            {
                "role_id": line.role_id,
                "speaker": line.speaker,
                "text": line.text,
                "procedural": line.procedural,
            }
            for line in data.show_history.lines
        ],
        "role_pad": {k: list(v) for k, v in data.role_pad.items()},
        "role_emotion": dict(data.role_emotion),
        "role_pad_neighbors": {k: list(v) for k, v in data.role_pad_neighbors.items()},
        "dialogue_example_index": data.dialogue_example_index,
    }


def apply_prefix(data: TalkShowData, prefix: dict[str, Any]) -> None:
    lines = prefix.get("lines")
    if isinstance(lines, list):
        data.show_history.lines = [
            HistoryLine(
                role_id=str(line["role_id"]),
                speaker=str(line["speaker"]),
                text=str(line["text"]),
                procedural=bool(line.get("procedural")),
            )
            for line in lines
        ]
    if isinstance(prefix.get("role_pad"), dict):
        now = time.monotonic()
        data.role_pad = {k: tuple(float(x) for x in v) for k, v in prefix["role_pad"].items()}
        data.role_pad_ts = {k: now for k in data.role_pad}
    if isinstance(prefix.get("role_emotion"), dict):
        data.role_emotion = {k: str(v) for k, v in prefix["role_emotion"].items()}
    if isinstance(prefix.get("role_pad_neighbors"), dict):
        data.role_pad_neighbors = {
            k: [str(x) for x in v] for k, v in prefix["role_pad_neighbors"].items()
        }
    if prefix.get("dialogue_example_index") is not None:
        data.dialogue_example_index = int(prefix["dialogue_example_index"])
    if prefix.get("persona_swap"):
        from agent.config import set_persona_swap

        set_persona_swap(True)
    # Per-session dialogue-quality flags, so two arms can share one job: one room = one job
    # process, so setting them here affects only this session. Only the flags below are accepted.
    for key, val in (prefix.get("flags") or {}).items():
        if key in REPLAY_FLAGS:
            # strings pass through (e.g. TALKSHOW_UPTAKE_PLANNER=lite); other values map to "1" / "0"
            os.environ[key] = val if isinstance(val, str) else ("1" if val else "0")
    logger.info(
        "ghost replay: prefix applied lines=%d persona_swap=%s",
        len(data.show_history.lines),
        bool(prefix.get("persona_swap")),
    )
    if data.turn_log is not None:
        data.turn_log.log(
            "ghost_replay_prefix",
            room=data.room_name,
            lines=len(data.show_history.lines),
            persona_swap=bool(prefix.get("persona_swap")),
            flags=prefix.get("flags") or {},
        )
