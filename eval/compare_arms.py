#!/usr/bin/env python3
"""
Compare latency A/B arms from deploy/ghost-arms.sh.

Usage:
    python -m eval.compare_arms logs/arms/<job>/manifest.tsv
    python -m eval.compare_arms --arm base=logs/session-1.jsonl,logs/session-2.jsonl --arm fix=...
    python -m eval.compare_arms MANIFEST --csv out.csv

For each arm (in manifest order, the first is the reference):
    silence per moderated handover (panelist line → open floor → intro → next
    panelist), split by cause as eval/gap_breakdown.py attributes it, and the
    change against the first arm; the share of the show that is silent; Gap A
    (human turn → host's first audio); mid-line pauses; and counts that show
    whether each latency flag engaged (canned-line cache hits, early polls,
    drafts used ahead, empty-queue waits).

Line-length correction: a handover's silence grows with the first sentence of each
line rendered inside it (its TTS, then the avatar's buffer), and with an unseeded LLM
that length varies from session to session. For each handover, the first chunks of
the lines that start in it (the host's lines and the next panelist's) are summed, as
seconds of audio. Each arm's mean silence is then moved to the pooled mean of that
length along the arm's own slope, floored at 0, so arms compare at equal lengths.
"""

from __future__ import annotations

import argparse
import csv
import math
import re
import statistics
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

try:
    from eval import gap_breakdown as gb
except ImportError:  # run as a script: eval/ is on sys.path
    import gap_breakdown as gb  # type: ignore[no-redef]

CAUSES = [c for c in gb.COMPONENTS if c not in ("endpoint", "translate")]
# Lines between two panelists in a moderated handover. direct_call = the host's own
# LLM line naming the next panelist (nobody raised, or the LLM host's pick).
MODERATED = ("open_floor", "intro", "direct_call")


@dataclass
class ArmStats:
    name: str
    sessions: list[Path] = field(default_factory=list)
    dead_pct: list[float] = field(default_factory=list)
    handover_total: list[float] = field(default_factory=list)
    handover_silent: list[float] = field(default_factory=list)
    handover_parts: list[dict[str, float]] = field(default_factory=list)
    handover_first_s: list[float] = field(default_factory=list)  # first-sentence audio, s
    session_silent: list[float] = field(default_factory=list)  # mean per handover, each session
    gap_a: list[float] = field(default_factory=list)
    mid_line: list[float] = field(default_factory=list)
    gen_fps: list[float] = field(default_factory=list)
    host_decide_s: list[float] = field(default_factory=list)
    counts: dict[str, int] = field(default_factory=lambda: defaultdict(int))


def first_chunks(rows: list[dict]) -> list[tuple[float, str, float]]:
    """(playout start, line step, audio s) of the first chunk of each line played."""
    out, seen = [], None
    for r in rows:
        if r["event"] != "avatar_chunk_play":
            continue
        line = re.sub(r":c\d+$", "", r.get("step", ""))
        if line != seen:
            out.append((float(r["ts"]), line, float(r.get("pcm_duration_sec") or 0.0)))
        seen = line
    return out


def first_sentence_s(chunks: list[tuple[float, str, float]], a: float, b: float, slack: float = 0.5) -> float:
    """Audio s of the first chunks of the lines that start in (a, b]: the host's lines
    and the next panelist's first sentence. NaN when no chunk events were logged."""
    if not chunks:
        return math.nan
    return sum(dur for ts, _, dur in chunks if a < ts <= b + slack)


def moderated_handovers(
    speech: list[gb.Speech], gaps: list[gb.Gap], chunks: list[tuple[float, str, float]] | None = None
) -> list[tuple[float, float, dict[str, float], float]]:
    """(total s, silent s, silent s by cause, first-sentence audio s) for each
    panelist → panelist handover that goes through the host's open-floor / intro /
    direct-call lines only."""
    seq = [(gb.line_kind(s.step), s) for s in speech]
    out = []
    for i, (kind, sp) in enumerate(seq):
        if kind != "panelist":
            continue
        if i + 1 < len(seq) and seq[i + 1][0] == "panelist" and seq[i + 1][1].line == sp.line:
            continue  # not the last chunk of this line
        j = i + 1
        while j < len(seq) and seq[j][0] != "panelist":
            j += 1
        between = [seq[x][0] for x in range(i + 1, j)]
        if j >= len(seq) or not between or any(k not in MODERATED for k in between):
            continue
        a, b = sp.end, seq[j][1].start
        parts = dict.fromkeys(CAUSES, 0.0)
        silent = 0.0
        for g in gaps:
            if g.start >= a - 1e-6 and g.end <= b + 1e-6:
                silent += g.dur
                for c in CAUSES:
                    parts[c] += g.parts.get(c, 0.0)
        out.append((b - a, silent, parts, first_sentence_s(chunks or [], a, b)))
    return out


def add_session(arm: ArmStats, path: Path) -> None:
    rows = gb.load(path)
    gaps, speech = gb.build_gaps(path.stem, rows)
    if not speech:
        print(f"skip {path}: no speech events", file=sys.stderr)
        return
    arm.sessions.append(path)
    wall = speech[-1].end - speech[0].start
    human = sum(b - a for a, b in gb.human_windows(rows) if speech[0].start <= a <= speech[-1].end)
    arm.dead_pct.append(100 * sum(g.dur for g in gaps) / max(wall - human, 1e-9))
    handovers = moderated_handovers(speech, gaps, first_chunks(rows))
    for total, silent, parts, first_s in handovers:
        arm.handover_total.append(total)
        arm.handover_silent.append(silent)
        arm.handover_parts.append(parts)
        arm.handover_first_s.append(first_s)
    if handovers:
        arm.session_silent.append(statistics.fmean(sum(h[2].values()) for h in handovers))
    arm.gap_a += [g.dur for g in gaps if g.kind == "after_human"]
    arm.mid_line += [g.dur for g in gaps if g.kind == "mid_line"]
    for r in rows:
        e = r["event"]
        if e in ("canned_cache_hit", "canned_cache_miss", "draft_ahead_used", "draft_ahead_dropped",
                 "speak_ahead_dropped", "hand_raise_wait_start"):
            arm.counts[e] += 1
        elif e == "speak_ahead_used":
            arm.counts[e] += 1
            arm.counts["speak_ahead_ready"] += bool(r.get("ready"))
        elif e == "avatar_bake" and r.get("gen_fps"):
            arm.gen_fps.append(float(r["gen_fps"]))
        elif e == "hand_raise_poll":
            arm.counts["poll"] += 1
            arm.counts["poll_prefetched"] += bool(r.get("poll_prefetched"))
        elif e in ("stream_ahead_started", "stream_ahead_dropped"):
            arm.counts[e] += 1
        elif e == "host_decide":
            arm.counts["host_decide"] += 1
            arm.counts["host_decide_fallback"] += bool(r.get("fallback"))
            if r.get("latency_s") is not None:
                arm.host_decide_s.append(float(r["latency_s"]))


def med(xs: list[float]) -> float:
    return statistics.median(xs) if xs else float("nan")


def mean_parts(arm: ArmStats) -> dict[str, float]:
    n = len(arm.handover_parts)
    return {c: (sum(p[c] for p in arm.handover_parts) / n if n else float("nan")) for c in CAUSES}


def length_corrected(arms: list[ArmStats]) -> tuple[float, dict[str, tuple[float, float, float]]]:
    """Mean silence per handover at the pooled mean first-sentence length.

    Returns (reference length s, {arm: (corrected mean s, slope s per s, mean length s)}).
    Each arm uses its own least-squares slope of silence on first-sentence length,
    floored at 0: where a fix hides the render (speak-ahead) the slope is near 0, and
    a longer first sentence can't make the wait shorter.
    """
    pairs = {
        a.name: [(x, sum(p.values())) for x, p in zip(a.handover_first_s, a.handover_parts) if not math.isnan(x)]
        for a in arms
    }
    xs_all = [x for ps in pairs.values() for x, _ in ps]
    if not xs_all:
        return math.nan, {}
    ref = statistics.fmean(xs_all)
    out: dict[str, tuple[float, float, float]] = {}
    for name, ps in pairs.items():
        if len(ps) < 3:
            out[name] = (math.nan, math.nan, math.nan)
            continue
        mx = statistics.fmean(x for x, _ in ps)
        my = statistics.fmean(y for _, y in ps)
        var = sum((x - mx) ** 2 for x, _ in ps)
        slope = max(0.0, sum((x - mx) * (y - my) for x, y in ps) / var) if var > 0 else 0.0
        out[name] = (my + slope * (ref - mx), slope, mx)
    return ref, out


def load_arms(args: argparse.Namespace) -> list[ArmStats]:
    arms: dict[str, ArmStats] = {}
    for manifest in args.manifest:
        with open(manifest, encoding="utf-8") as f:
            for row in csv.DictReader(f, delimiter="\t"):
                arms.setdefault(row["arm"], ArmStats(row["arm"]))
                add_session(arms[row["arm"]], Path(row["session"]))
    for spec in args.arm or []:
        name, _, paths = spec.partition("=")
        arms.setdefault(name, ArmStats(name))
        for p in filter(None, paths.split(",")):
            add_session(arms[name], Path(p))
    return list(arms.values())


def fmt(v: float, d: int = 2) -> str:
    return "—" if v != v else f"{v:.{d}f}"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("manifest", nargs="*", help="manifest.tsv from deploy/ghost-arms.sh")
    ap.add_argument("--arm", action="append", help="NAME=session.jsonl[,session.jsonl...]")
    ap.add_argument("--csv", type=Path, help="also write the per-arm numbers here")
    args = ap.parse_args()
    arms = [a for a in load_arms(args) if a.sessions]
    if not arms:
        ap.error("no sessions found")

    ref = mean_parts(arms[0])
    ref_total = sum(ref.values())
    print("== Silence per moderated handover, by cause (mean s; Δ vs first arm)")
    head = ["arm", "sess", "hand", "silent"] + CAUSES + ["Δ"]
    print("  ".join(f"{h:>9}" if i else f"{h:<10}" for i, h in enumerate(head)))
    table = []
    for a in arms:
        parts = mean_parts(a)
        total = sum(parts.values())
        cells = [a.name, str(len(a.sessions)), str(len(a.handover_parts)), fmt(total)] + [fmt(parts[c]) for c in CAUSES]
        cells.append(fmt(total - ref_total) if a is not arms[0] else "")
        print("  ".join(f"{c:>9}" if i else f"{c:<10}" for i, c in enumerate(cells)))
        table.append((a, parts, total))

    ref_len, corrected = length_corrected(arms)
    if corrected:
        print(f"\n== Corrected to the same first-sentence length ({fmt(ref_len)} s of audio per handover)")
        print(f"{'arm':<10}  {'silent':>7}  {'1st sent':>8}  {'slope':>6}  {'corrected':>9}  {'Δ':>6}  {'sessions':>11}")
        ref_corr = corrected.get(arms[0].name, (math.nan,))[0]
        for a, _, total in table:
            c, slope, mx = corrected.get(a.name, (math.nan, math.nan, math.nan))
            spread = f"{fmt(min(a.session_silent))}–{fmt(max(a.session_silent))}" if a.session_silent else "—"
            delta = fmt(c - ref_corr) if a is not arms[0] else ""
            print(f"{a.name:<10}  {fmt(total):>7}  {fmt(mx):>8}  {fmt(slope):>6}  {fmt(c):>9}  {delta:>6}  {spread:>11}")
        print("  1st sent: mean audio s of the first chunks starting in a handover; slope: s of silence per s of it;")
        print("  sessions: lowest and highest session mean, the run-to-run spread")

    print("\n== Show level (medians)")
    print(f"{'arm':<10}  {'dead%':>6}  {'handover':>9}  {'silent':>7}  {'gapA':>6}  {'mid-line':>8}  {'avatar fps':>10}")
    for a in arms:
        print(
            f"{a.name:<10}  {fmt(med(a.dead_pct), 0):>6}  {fmt(med(a.handover_total), 1):>9}  "
            f"{fmt(med(a.handover_silent), 1):>7}  {fmt(med(a.gap_a), 2):>6}  {fmt(med(a.mid_line), 2):>8}  "
            f"{fmt(med(a.gen_fps), 1):>10}"
        )

    print("\n== Did each flag engage? (event counts)")
    keys = ["canned_cache_hit", "canned_cache_miss", "poll_prefetched", "poll", "draft_ahead_used", "draft_ahead_dropped",
            "speak_ahead_used", "speak_ahead_ready", "speak_ahead_dropped", "hand_raise_wait_start",
            "stream_ahead_started", "stream_ahead_dropped", "host_decide", "host_decide_fallback"]
    labels = ["cache hits", "cache misses", "early polls", "polls", "drafts used", "drafts dropped",
              "spoken ahead", "ahead ready", "ahead dropped", "empty waits",
              "streamed starts", "stream dropped", "LLM host picks", "host fallbacks"]
    print(f"{'arm':<10}  " + "  ".join(f"{h:>14}" for h in labels))
    for a in arms:
        print(f"{a.name:<10}  " + "  ".join(f"{a.counts.get(k, 0):>14}" for k in keys))

    if any(a.host_decide_s for a in arms):
        print("\n== LLM host decision (host_decide latency, s)")
        for a in arms:
            if a.host_decide_s:
                print(f"{a.name:<10}  median {fmt(med(a.host_decide_s), 2)}  n={len(a.host_decide_s)}")

    if args.csv:
        with args.csv.open("w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["arm", "sessions", "handovers", "silent_per_handover_s", *CAUSES,
                        "first_sentence_s_mean", "length_slope", "silent_corrected_s",
                        "session_silent_min_s", "session_silent_max_s",
                        "dead_pct_median", "handover_median_s", "handover_silent_median_s", "gap_a_median_s",
                        "avatar_fps_median", *keys])
            for a, parts, total in table:
                w.writerow([a.name, len(a.sessions), len(a.handover_parts), round(total, 3),
                            *[round(parts[c], 3) for c in CAUSES],
                            *[round(v, 3) for v in (corrected.get(a.name, (math.nan,) * 3)[2],
                                                    corrected.get(a.name, (math.nan,) * 3)[1],
                                                    corrected.get(a.name, (math.nan,) * 3)[0])],
                            round(min(a.session_silent), 3) if a.session_silent else "",
                            round(max(a.session_silent), 3) if a.session_silent else "",
                            round(med(a.dead_pct), 1),
                            round(med(a.handover_total), 2), round(med(a.handover_silent), 2),
                            round(med(a.gap_a), 2), round(med(a.gen_fps), 1), *[a.counts.get(k, 0) for k in keys]])
        print(f"\nwrote {args.csv}")


if __name__ == "__main__":
    main()
