"""Shared JSONL log parsing for eval scripts."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

ROLE_DISPLAY = {
    "host": ("Lessac", "Host"),
    "commentator": ("Ryan", "Commentator"),
    "guest": ("Amy", "Guest"),
}


def role_label(role: str) -> str:
    name, kind = ROLE_DISPLAY.get(role, (role.capitalize(), role))
    return f"{name} ({kind})"


def role_name(role: str) -> str:
    return ROLE_DISPLAY.get(role, (role.capitalize(), role))[0]


@dataclass
class PanelReplyRecord:
    role: str
    reply: str
    reply_len: int
    step: str = ""
    next_tag: str = ""
    model_latency_s: float | None = None
    dialogue_pick: str | None = None
    hint_chars: int = 0
    source: str = ""
    turn_num: int = 0
    heard: str = ""
    seq: int = 0
    ts: float = 0.0


@dataclass
class HumanTurn:
    turn_num: int
    source: str
    heard: str = ""
    host_reply: str = ""
    panel_replies: list[PanelReplyRecord] = field(default_factory=list)
    session_meta: dict[str, Any] = field(default_factory=dict)


def load_events(*paths: Path) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    for p in paths:
        with open(p, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                obj["_source"] = p.name
                events.append(obj)
    events.sort(key=lambda e: e["ts"])
    return events


def turn_window_end(
    events: list[dict[str, Any]], start_idx: int, source: str
) -> int:
    """Exclusive end index for one human turn, scoped to a single log file."""
    for j in range(start_idx + 1, len(events)):
        e = events[j]
        if e.get("_source", "") != source:
            return j
        if e.get("event") == "gemma_stt_start":
            return j
    return len(events)


def session_metadata(events: list[dict[str, Any]]) -> dict[str, Any]:
    for e in events:
        if e.get("event") != "session_start":
            continue
        return {
            k: e.get(k)
            for k in (
                "room",
                "log_file",
                "scenario_id",
                "turn_mode",
                "dialogue_library",
                "dialogue_pick",
            )
        } | {"source": e.get("_source", "")}
    return {}


def extract_human_turns(events: list[dict[str, Any]]) -> list[HumanTurn]:
    """One human turn = gemma_stt_start → next gemma_stt_start (or EOF)."""
    meta = session_metadata(events)
    stt_starts = [
        (i, e) for i, e in enumerate(events) if e["event"] == "gemma_stt_start"
    ]
    turns: list[HumanTurn] = []
    seq = 0

    turn_counters: dict[str, int] = {}
    for start_idx, start_event in stt_starts:
        source = start_event.get("_source", "")
        end_idx = turn_window_end(events, start_idx, source)
        window = events[start_idx:end_idx]
        turn_counters[source] = turn_counters.get(source, 0) + 1
        turn_num = turn_counters[source]

        stt_done = next((e for e in window if e["event"] == "gemma_stt_done"), None)
        if stt_done is None:
            continue

        turn = HumanTurn(
            turn_num=turn_num,
            source=source,
            heard=(stt_done.get("heard") or "").strip(),
            host_reply=(stt_done.get("reply") or "").strip(),
            session_meta=dict(meta),
        )

        for e in window:
            if e["event"] != "panel_model_done":
                continue
            reply = (e.get("reply") or "").strip()
            if not reply:
                continue
            reply_len = int(e.get("reply_len") or len(reply))
            turn.panel_replies.append(
                PanelReplyRecord(
                    role=e.get("role", ""),
                    reply=reply,
                    reply_len=reply_len,
                    step=e.get("step", ""),
                    next_tag=e.get("next_tag", ""),
                    model_latency_s=e.get("model_latency_s"),
                    dialogue_pick=e.get("dialogue_pick"),
                    hint_chars=int(e.get("hint_chars") or 0),
                    source=source,
                    turn_num=turn_num,
                    heard=turn.heard,
                    seq=seq,
                    ts=float(e.get("ts") or 0.0),
                )
            )
            seq += 1
        turns.append(turn)

    return turns


def flatten_panel_replies(turns: list[HumanTurn]) -> list[PanelReplyRecord]:
    out: list[PanelReplyRecord] = []
    for turn in turns:
        out.extend(turn.panel_replies)
    return out
