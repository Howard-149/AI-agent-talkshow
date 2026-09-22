"""Print a short JSONL recap after a ghost (or live) session."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from eval.log_parse import extract_human_turns, load_events, role_label, session_metadata


def find_session_log(log_dir: Path, room: str) -> Path | None:
    """Newest ``session-*.jsonl`` whose ``session_start`` matches ``room``."""
    candidates: list[tuple[float, Path]] = []
    for path in log_dir.glob("session-*.jsonl"):
        try:
            mtime = path.stat().st_mtime
        except OSError:
            continue
        if _log_room(path) == room:
            candidates.append((mtime, path))
    if not candidates:
        return None
    candidates.sort(key=lambda item: item[0], reverse=True)
    return candidates[0][1]


def _log_room(path: Path) -> str:
    try:
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                if obj.get("event") == "session_start":
                    return str(obj.get("room") or "")
    except (OSError, json.JSONDecodeError):
        return ""
    return ""


def summarize_session(log_path: Path) -> str:
    events = load_events(log_path)
    meta = session_metadata(events)
    counts = Counter(str(e.get("event") or "") for e in events)
    turns = extract_human_turns(events)
    lines = [
        f"log: {log_path}",
        f"room: {meta.get('room') or '?'}",
        f"scenario: {meta.get('scenario_id') or '?'}  dialogue_pick={meta.get('dialogue_pick')}",
        f"events: {len(events)}  human_turns: {len(turns)}",
        "counts: "
        + ", ".join(
            f"{name}={counts[name]}"
            for name in (
                "session_start",
                "ghost_human_start",
                "ghost_human_done",
                "gemma_stt_start",
                "gemma_stt_done",
                "panel_start",
                "panel_done",
                "avatar_sync_playout",
                "floor_grant_human",
                "pad_map",
            )
            if counts[name]
        ),
    ]
    for turn in turns:
        lines.append(
            f"  H{turn.turn_num} heard={turn.heard[:80]!r} "
            f"host={turn.host_reply[:60]!r} panelists={len(turn.panel_replies)}"
        )
        for rec in turn.panel_replies:
            lat = (
                f" {rec.model_latency_s:.2f}s"
                if rec.model_latency_s is not None
                else ""
            )
            lines.append(
                f"    {role_label(rec.role)}{lat} {rec.reply[:90]!r}"
            )
    model_lines = _format_model_pad_lines(events)
    if model_lines:
        lines.append("model_raw:")
        lines.extend(model_lines)
    pad_events = [e for e in events if e.get("event") == "pad_map"]
    if pad_events:
        lines.append(f"pad_map: {len(pad_events)}")
        for e in pad_events[:16]:
            role = role_label(str(e.get("role") or "?"))
            before = e.get("pad_before")
            after = e.get("pad_after")
            delta = e.get("delta")
            was = e.get("emotion_was") or "?"
            now = e.get("emotion") or "?"
            knn = e.get("neighbors") or []
            delta_s = f" Δ{delta}" if delta else ""
            knn_s = f" knn={knn}" if knn else ""
            lines.append(
                f"  {role} {before}{delta_s} → {after}  {was}→{now}{knn_s}"
            )
    return "\n".join(lines)


_MODEL_DONE_EVENTS = frozenset(
    {
        "ghost_human_done",
        "panel_model_done",
        "gemma_stt_done",
        "host_direct_call",
    }
)


def _format_model_pad_lines(events: list[dict]) -> list[str]:
    """Omit vs explicit [pad] 0 0 0 vs a real delta — from logged model text."""
    out: list[str] = []
    for e in events:
        if e.get("event") not in _MODEL_DONE_EVENTS:
            continue
        raw = str(e.get("raw") or "")
        role = role_label(str(e.get("role") or e.get("active_role") or "?"))
        last = ""
        for line in reversed(raw.splitlines()):
            if line.strip():
                last = line.strip()
                break
        has_tag = "[pad]" in raw.lower()
        delta = e.get("pad_delta")
        if not raw:
            status = "raw missing"
        elif has_tag:
            status = f"pad_delta={delta}  last={last!r}"
        else:
            status = "NO [pad] tag"
        step = e.get("step") or e.get("event")
        out.append(f"  {role} {step} {status}")
    return out
