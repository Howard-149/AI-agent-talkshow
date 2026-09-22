#!/usr/bin/env python3
"""
Dead-air breakdown from talkshow session JSONL logs (avatar on or off).

Usage:
    python eval/gap_breakdown.py logs/session-*.jsonl
    python eval/gap_breakdown.py logs/session-*.jsonl --gaps   # list every gap

Model:
    Agent audio intervals come from ``agent_state`` (speaking → anything else),
    labelled by the enclosing ``line_start`` / ``line_end``. Works with the avatar
    on or off. Older logs without ``agent_state`` fall back to
    ``avatar_chunk_play``: [ts, ts + pcm_duration_sec].
    Every hole between consecutive intervals is silence, except the human's own
    turn (``floor_grant_human`` → end of human speech), which is excluded.
    End of human speech = ``user_state`` speaking → listening when logged,
    else ``gemma_stt_start``.

    Each gap is attributed to what ran inside it, by interval overlap:
        endpoint   end of human speech → gemma_stt_start (VAD + turn detector)
        llm        panel_model_done / gemma_stt_done   [ts - model_latency_s, ts]
        translate  translate_done                      [ts - model_latency_s, ts]
        poll       hand_raise_poll                     [flash_ts - poll_latency_s, flash_ts]
        sleep      hand_raise_poll_flash, hand_raise_grant_pause, hand_raise_wait_*
        tts        tts_synthesize (of the next chunk when steps are logged)
        avatar     avatar_bake of the next chunk       [ts - preroll_ms, ts]
        other      remainder (role switch, UI, event loop, logging)

Sessions with neither ``agent_state`` nor ``avatar_chunk_play`` are skipped.
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

COMPONENTS = ("endpoint", "llm", "translate", "poll", "sleep", "tts", "avatar", "other")


@dataclass
class Speech:
    start: float
    end: float
    role: str
    step: str  # full chunk step, e.g. "queue_guest_turn_0:c1"
    chunk_idx: int

    @property
    def line(self) -> str:
        return re.sub(r":c\d+$", "", self.step)


@dataclass
class Gap:
    session: str
    kind: str
    start: float
    end: float
    prev: Speech | None
    next: Speech
    parts: dict[str, float] = field(default_factory=dict)

    @property
    def dur(self) -> float:
        return self.end - self.start


def load(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    rows.sort(key=lambda r: r.get("ts", 0.0))
    return rows


def line_kind(step: str) -> str:
    """Coarse category of a spoken line from its step name."""
    s = re.sub(r":c\d+$", "", step)
    for prefix, kind in (
        ("session_welcome", "welcome"),
        ("host_reply", "host_reply"),
        ("open_floor", "open_floor"),
        ("intro_", "intro"),
        ("grant_human", "grant_human"),
        ("direct_call", "direct_call"),
        ("close_round", "close"),
        ("queue_", "panelist"),
        ("speech_", "panelist"),
        ("chain_", "panelist"),
    ):
        if s.startswith(prefix):
            return kind
    return "other"


def overlap(a0: float, a1: float, b0: float, b1: float) -> float:
    return max(0.0, min(a1, b1) - max(a0, b0))


def component_intervals(rows: list[dict[str, Any]]) -> list[tuple[str, float, float, str]]:
    """(component, start, end, step-or-empty) for everything that can block speech."""
    out: list[tuple[str, float, float, str]] = []
    flash_ts: list[float] = []
    for r in rows:
        e, ts = r["event"], r["ts"]
        if e in ("panel_model_done", "gemma_stt_done") and r.get("model_latency_s"):
            out.append(("llm", ts - float(r["model_latency_s"]), ts, ""))
        elif e == "translate_done" and r.get("model_latency_s"):
            out.append(("translate", ts - float(r["model_latency_s"]), ts, ""))
        elif e == "hand_raise_poll_flash":
            sec = float(r.get("flash_sec") or 0)
            out.append(("sleep", ts, ts + sec, ""))
            flash_ts.append(ts)
        elif e == "hand_raise_grant_pause":
            out.append(("sleep", ts, ts + float(r.get("pause_sec") or 0), ""))
        elif e == "tts_synthesize":
            lat = r.get("tts_latency_s")
            lat = float(lat) if lat is not None else float(r.get("tts_ms") or 0) / 1000
            out.append(("tts", ts - lat, ts, r.get("step", "")))
        elif e == "avatar_bake":
            ms = r.get("preroll_ms") or r.get("avatar_bake_ms") or 0
            out.append(("avatar", ts - float(ms) / 1000, ts, r.get("step", "")))
    # Poll ends where the flash begins (flash sleeps before hand_raise_poll is logged).
    for r in rows:
        if r["event"] != "hand_raise_poll" or not r.get("poll_latency_s"):
            continue
        ts = r["ts"]
        lat = float(r["poll_latency_s"])
        prior = [f for f in flash_ts if ts - 2.0 <= f <= ts]
        end = prior[-1] if prior else ts
        out.append(("poll", end - lat, end, ""))
    # Empty-queue waits.
    start = None
    for r in rows:
        if r["event"] == "hand_raise_wait_start":
            start = r["ts"]
        elif r["event"] in ("hand_raise_wait_done", "hand_raise_wait_early") and start:
            out.append(("sleep", start, r["ts"], ""))
            start = None
    return out


def speech_intervals(rows: list[dict[str, Any]]) -> list[Speech]:
    if any(r["event"] == "agent_state" for r in rows):
        return _speech_from_agent_state(rows)
    out = []
    for r in rows:
        if r["event"] != "avatar_chunk_play":
            continue
        dur = float(r.get("pcm_duration_sec") or 0)
        out.append(
            Speech(
                start=r["ts"],
                end=r["ts"] + dur,
                role=r.get("role", ""),
                step=r.get("step", ""),
                chunk_idx=int(r.get("chunk_idx") or 0),
            )
        )
    return out


def _speech_from_agent_state(rows: list[dict[str, Any]]) -> list[Speech]:
    """speaking intervals from agent_state, labelled by the open line_start."""
    out: list[Speech] = []
    line_step, line_role = "", ""
    chunk_idx = 0
    start: float | None = None
    for r in rows:
        e = r["event"]
        if e == "line_start":
            line_step, line_role = r.get("step", ""), r.get("role", "")
            chunk_idx = 0
        elif e == "line_end" and r.get("step") == line_step and start is None:
            line_step = ""
        elif e == "agent_state":
            speaking = r.get("state") == "speaking"
            if speaking and start is None:
                start = r["ts"]
            elif not speaking and start is not None:
                step = f"{line_step or 'unlabelled'}:c{chunk_idx}"
                out.append(Speech(start, r["ts"], line_role or r.get("active_role", ""), step, chunk_idx))
                chunk_idx += 1
                start = None
    return out


def human_speech_ends(rows: list[dict[str, Any]]) -> list[float]:
    """VAD end of human speech (user_state speaking → listening/away)."""
    return [
        r["ts"]
        for r in rows
        if r["event"] == "user_state" and r.get("old_state") == "speaking"
    ]


def _human_end_before(ends: list[float], stt_start: float, since: float) -> float | None:
    cands = [t for t in ends if since <= t <= stt_start]
    return cands[-1] if cands else None


def human_windows(rows: list[dict[str, Any]]) -> list[tuple[float, float]]:
    """Human holds the floor: grant → end of human speech (or gemma_stt_start)."""
    ends = human_speech_ends(rows)
    wins = []
    grant = None
    for r in rows:
        if r["event"] == "floor_grant_human":
            grant = r["ts"]
        elif r["event"] == "gemma_stt_start" and grant is not None:
            end = _human_end_before(ends, r["ts"], grant) or r["ts"]
            wins.append((grant, end))
            grant = None
    return wins


def build_gaps(name: str, rows: list[dict[str, Any]]) -> tuple[list[Gap], list[Speech]]:
    speech = speech_intervals(rows)
    if not speech:
        return [], []
    comps = component_intervals(rows)
    humans = human_windows(rows)
    stt_starts = [r["ts"] for r in rows if r["event"] == "gemma_stt_start"]
    for _, human_end in humans:
        stt = next((t for t in stt_starts if t >= human_end), None)
        if stt is not None and stt > human_end:
            comps.append(("endpoint", human_end, stt, ""))

    gaps: list[Gap] = []
    prev: Speech | None = None
    for sp in speech:
        if prev is None:
            prev = sp
            continue
        g0, g1 = prev.end, sp.start
        if g1 - g0 <= 0.05:
            prev = sp if sp.end > prev.end else prev
            continue
        # Human turn inside this hole → only count from the end of human speech.
        human = next((h for h in humans if g0 - 1.0 <= h[0] <= g1 and h[1] <= g1), None)
        if human is not None:
            g0 = human[1]
            kind = "after_human"
        elif any(g0 <= t <= g1 for t in stt_starts):
            g0 = max(t for t in stt_starts if g0 <= t <= g1)
            kind = "after_human"
        elif sp.chunk_idx > 0 and sp.line == prev.line:
            kind = "mid_line"
        else:
            kind = f"{line_kind(prev.step)}->{line_kind(sp.step)}"

        parts = dict.fromkeys(COMPONENTS, 0.0)
        for comp, c0, c1, step in comps:
            if comp in ("tts", "avatar") and step and step != sp.step:
                continue  # only the chunk we are waiting for
            parts[comp] += overlap(g0, g1, c0, c1)
        used = sum(v for k, v in parts.items() if k != "other")
        parts["other"] = max(0.0, (g1 - g0) - used)
        gaps.append(Gap(name, kind, g0, g1, prev, sp, parts))
        prev = sp
    return gaps, speech


def pct(xs: list[float], q: float) -> float:
    if not xs:
        return float("nan")
    xs = sorted(xs)
    k = (len(xs) - 1) * q
    lo, hi = int(k), min(int(k) + 1, len(xs) - 1)
    return xs[lo] + (xs[hi] - xs[lo]) * (k - lo)


def fmt_row(cols: list[str], widths: list[int]) -> str:
    return "  ".join(c.ljust(w) if i == 0 else c.rjust(w) for i, (c, w) in enumerate(zip(cols, widths)))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("logs", nargs="+", type=Path)
    ap.add_argument("--gaps", action="store_true", help="print every gap")
    args = ap.parse_args()

    all_gaps: list[Gap] = []
    print("== Sessions ==")
    widths = [26, 7, 7, 7, 7, 6]
    print(fmt_row(["session", "wall_s", "agent_s", "human_s", "silent_s", "dead%"], widths))
    for p in sorted(args.logs):
        rows = load(p)
        gaps, speech = build_gaps(p.stem, rows)
        if not speech:
            continue
        wall = speech[-1].end - speech[0].start
        talk = sum(s.end - s.start for s in speech)
        human = sum(b - a for a, b in human_windows(rows) if speech[0].start <= a <= speech[-1].end)
        silent = sum(g.dur for g in gaps)
        print(fmt_row(
            [p.stem, f"{wall:.0f}", f"{talk:.0f}", f"{human:.0f}", f"{silent:.0f}",
             f"{100 * silent / max(wall - human, 1e-9):.0f}"], widths))
        all_gaps.extend(gaps)

    if not all_gaps:
        print("no agent_state or avatar_chunk_play events found")
        return

    print("\n== Silence by gap kind (seconds; mean component share) ==")
    by_kind: dict[str, list[Gap]] = defaultdict(list)
    for g in all_gaps:
        by_kind[g.kind].append(g)
    widths = [26, 4, 6, 6, 6, 7] + [6] * len(COMPONENTS)
    print(fmt_row(["kind", "n", "p50", "p90", "max", "total"] + list(COMPONENTS), widths))
    for kind, gs in sorted(by_kind.items(), key=lambda kv: -sum(g.dur for g in kv[1])):
        durs = [g.dur for g in gs]
        means = [statistics.mean(g.parts[c] for g in gs) for c in COMPONENTS]
        print(fmt_row(
            [kind, str(len(gs)), f"{pct(durs, .5):.2f}", f"{pct(durs, .9):.2f}", f"{max(durs):.2f}",
             f"{sum(durs):.0f}"] + [f"{m:.2f}" for m in means], widths))

    total = sum(g.dur for g in all_gaps)
    print(f"\n== Where all {total:.0f} s of silence went ==")
    for c in COMPONENTS:
        s = sum(g.parts[c] for g in all_gaps)
        print(f"  {c:7s} {s:7.1f} s  {100 * s / total:5.1f}%")

    if args.gaps:
        print("\n== Every gap ==")
        for g in all_gaps:
            parts = " ".join(f"{c}={g.parts[c]:.2f}" for c in COMPONENTS if g.parts[c] > 0.005)
            print(f"{g.session} {g.kind:24s} {g.dur:6.2f}s  -> {g.next.step:45s} {parts}")


if __name__ == "__main__":
    main()
