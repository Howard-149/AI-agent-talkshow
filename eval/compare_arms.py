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
"""

from __future__ import annotations

import argparse
import csv
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
MODERATED = ("open_floor", "intro")  # lines between two panelists in a moderated handover


@dataclass
class ArmStats:
    name: str
    sessions: list[Path] = field(default_factory=list)
    dead_pct: list[float] = field(default_factory=list)
    handover_total: list[float] = field(default_factory=list)
    handover_silent: list[float] = field(default_factory=list)
    handover_parts: list[dict[str, float]] = field(default_factory=list)
    gap_a: list[float] = field(default_factory=list)
    mid_line: list[float] = field(default_factory=list)
    counts: dict[str, int] = field(default_factory=lambda: defaultdict(int))


def moderated_handovers(speech: list[gb.Speech], gaps: list[gb.Gap]) -> list[tuple[float, float, dict[str, float]]]:
    """(total s, silent s, silent s by cause) for each panelist → panelist handover
    that goes through the host's open-floor and/or intro lines only."""
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
        out.append((b - a, silent, parts))
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
    for total, silent, parts in moderated_handovers(speech, gaps):
        arm.handover_total.append(total)
        arm.handover_silent.append(silent)
        arm.handover_parts.append(parts)
    arm.gap_a += [g.dur for g in gaps if g.kind == "after_human"]
    arm.mid_line += [g.dur for g in gaps if g.kind == "mid_line"]
    for r in rows:
        e = r["event"]
        if e in ("canned_cache_hit", "canned_cache_miss", "draft_ahead_used", "draft_ahead_dropped", "hand_raise_wait_start"):
            arm.counts[e] += 1
        elif e == "hand_raise_poll":
            arm.counts["poll"] += 1
            arm.counts["poll_prefetched"] += bool(r.get("poll_prefetched"))


def med(xs: list[float]) -> float:
    return statistics.median(xs) if xs else float("nan")


def mean_parts(arm: ArmStats) -> dict[str, float]:
    n = len(arm.handover_parts)
    return {c: (sum(p[c] for p in arm.handover_parts) / n if n else float("nan")) for c in CAUSES}


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

    print("\n== Show level (medians)")
    print(f"{'arm':<10}  {'dead%':>6}  {'handover':>9}  {'silent':>7}  {'gapA':>6}  {'mid-line':>8}")
    for a in arms:
        print(
            f"{a.name:<10}  {fmt(med(a.dead_pct), 0):>6}  {fmt(med(a.handover_total), 1):>9}  "
            f"{fmt(med(a.handover_silent), 1):>7}  {fmt(med(a.gap_a), 2):>6}  {fmt(med(a.mid_line), 2):>8}"
        )

    print("\n== Did each flag engage? (event counts)")
    keys = ["canned_cache_hit", "canned_cache_miss", "poll_prefetched", "poll", "draft_ahead_used", "draft_ahead_dropped", "hand_raise_wait_start"]
    labels = ["cache hits", "cache misses", "early polls", "polls", "drafts used", "drafts dropped", "empty waits"]
    print(f"{'arm':<10}  " + "  ".join(f"{h:>14}" for h in labels))
    for a in arms:
        print(f"{a.name:<10}  " + "  ".join(f"{a.counts.get(k, 0):>14}" for k in keys))

    if args.csv:
        with args.csv.open("w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["arm", "sessions", "handovers", "silent_per_handover_s", *CAUSES,
                        "dead_pct_median", "handover_median_s", "handover_silent_median_s", "gap_a_median_s", *keys])
            for a, parts, total in table:
                w.writerow([a.name, len(a.sessions), len(a.handover_parts), round(total, 3),
                            *[round(parts[c], 3) for c in CAUSES], round(med(a.dead_pct), 1),
                            round(med(a.handover_total), 2), round(med(a.handover_silent), 2),
                            round(med(a.gap_a), 2), *[a.counts.get(k, 0) for k in keys]])
        print(f"\nwrote {args.csv}")


if __name__ == "__main__":
    main()
