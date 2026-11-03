#!/usr/bin/env python3
"""
Who-speaks-next metrics per arm (FIFO queue vs LLM host), complementing
eval/compare_arms.py, which covers latency.

Per arm: panel turns per human turn, speaker balance (largest share of panel turns),
the same panelist twice in a row (with no human turn between), how long a raised
human hand waits for the floor, and for the LLM host the tool-call success rate,
fallback reasons and decision latency.

Usage:
    python -m eval.host_decide_report logs/arms/<job>/manifest.tsv   # FIFO_a, FIFO_b … → FIFO
    python -m eval.host_decide_report --arm FIFO=logs/session-1.jsonl --arm LLM=logs/session-2.jsonl
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


def load(path: Path) -> list[dict[str, Any]]:
    return [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]


def session_stats(events: list[dict[str, Any]]) -> dict[str, Any]:
    panel = []  # panelist roles in order, "|" marks a human turn
    waits: list[float] = []
    raised_at: float | None = None
    decides = [e for e in events if e.get("event") == "host_decide"]
    for e in events:
        ev = e.get("event")
        if ev == "panel_model_done":
            panel.append(e.get("role", "?"))
        elif ev in ("ghost_human_done", "gemma_stt_done"):
            panel.append("|")
        elif ev == "hand_raise_enqueue" and e.get("role") == "human" and raised_at is None:
            raised_at = e["ts"]
        elif ev == "floor_grant_human" and raised_at is not None:
            waits.append(e["ts"] - raised_at)
            raised_at = None
    runs = [r for r in "".join(p if p == "|" else p[0] for p in panel).split("|")]
    repeats = sum(a == b for seg in runs for a, b in zip(seg, seg[1:]))
    roles = Counter(p for p in panel if p != "|")
    return {
        "panel_turns": sum(roles.values()),
        "human_turns": panel.count("|"),
        "roles": roles,
        "repeats": repeats,
        "waits": waits,
        "decides": decides,
        "tracebacks": sum(1 for e in events if "error" in str(e.get("event", "")).lower()),
    }


def arms_from_manifest(path: Path, *, group: bool = True) -> dict[str, list[Path]]:
    """Arms by name; with ``group``, interleaved arms FIFO_a / FIFO_b … count as FIFO."""
    arms: dict[str, list[Path]] = defaultdict(list)
    with path.open(newline="") as f:
        for row in csv.DictReader(f, delimiter="\t"):
            name = row["arm"].split("_", 1)[0] if group else row["arm"]
            arms[name].append(Path(row["session"]))
    return arms


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument("manifest", nargs="?", type=Path)
    ap.add_argument("--arm", action="append", default=[], help="NAME=log1,log2")
    ap.add_argument("--no-group", action="store_true", help="keep FIFO_a / FIFO_b … apart")
    a = ap.parse_args()
    arms = arms_from_manifest(a.manifest, group=not a.no_group) if a.manifest else {}
    for spec in a.arm:
        name, _, logs = spec.partition("=")
        arms[name] = [Path(x) for x in logs.split(",") if x]

    print(f"{'arm':<6}{'sess':>5}{'panel/human':>12}{'max share':>10}{'repeats':>8}"
          f"{'wait p50':>9}{'wait max':>9}{'decides':>8}{'ok':>6}{'dec p50':>8}")
    for name, paths in arms.items():
        stats = [session_stats(load(p)) for p in paths if p.is_file()]
        if not stats:
            continue
        roles = sum((s["roles"] for s in stats), Counter())
        panel = sum(s["panel_turns"] for s in stats)
        human = sum(s["human_turns"] for s in stats) or 1
        waits = [w for s in stats for w in s["waits"]]
        decides = [d for s in stats for d in s["decides"]]
        ok = [d for d in decides if d.get("role")]
        lat = [d["latency_s"] for d in decides if "latency_s" in d]
        print(
            f"{name:<6}{len(stats):>5}{panel / human:>12.1f}"
            f"{(max(roles.values()) / panel if panel else 0):>10.0%}"
            f"{sum(s['repeats'] for s in stats):>8}"
            f"{(statistics.median(waits) if waits else float('nan')):>9.1f}"
            f"{(max(waits) if waits else float('nan')):>9.1f}"
            f"{len(decides):>8}{(len(ok) / len(decides) if decides else float('nan')):>6.0%}"
            f"{(statistics.median(lat) if lat else float('nan')):>8.2f}"
        )
        fallbacks = Counter(d.get("fallback") for d in decides if d.get("fallback"))
        tools = Counter(d.get("tool") for d in ok)
        if decides:
            print(f"       tools={dict(tools)} fallbacks={dict(fallbacks) or '-'}")
        print(f"       panel turns by role={dict(roles)}  error events={sum(s['tracebacks'] for s in stats)}")


if __name__ == "__main__":
    main()
