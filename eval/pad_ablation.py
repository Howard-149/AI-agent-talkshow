#!/usr/bin/env python3
"""
Offline PAD-update ablation: replay logged appraisal deltas under each update rule.

Every live session logs each listener's appraisal delta (``pad_appraisal``) and each
post-speech arousal relax (``pad_expression``). Replaying those events through the
PAD update with a rule switched off isolates what the rule does to the trajectory —
same deltas, same timing, no sampling noise. What it cannot show is the feedback
loop (mood → generated line → later deltas); deploy/arms/pad-ablation.arms covers
that live.

Configs: FULL (soft bound + arousal decay γ, the live default), NOSOFT (paper's
add-and-clip), NODECAY (γ = 0), NEITHER. Fidelity check: FULL replayed on sessions
that ran with FULL must reproduce the logged ``pad_map`` states.

Per config, over all role trajectories: max |P|, share of states with an axis at
|s| ≥ 0.8 ("near the bound") and pinned at |s| = 1, mean arousal, P range, share of
non-Neutral kNN labels and distinct labels.

Usage (Babel, where the session logs are):
    python eval/pad_ablation.py logs/session-*.jsonl
    python eval/pad_ablation.py --manifest logs/arms/<job>/manifest.tsv     # live arms
    python eval/pad_ablation.py --since 1790950000 logs/session-*.jsonl
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

HALF_LIFE_S = 120.0
CONFIGS = {
    "FULL": {"soft": True, "gamma": 0.12},
    "NOSOFT": {"soft": False, "gamma": 0.12},
    "NODECAY": {"soft": True, "gamma": 0.0},
    "NEITHER": {"soft": False, "gamma": 0.0},
}


@dataclass
class Traj:
    session: str
    role: str
    states: list[tuple[float, float, float]] = field(default_factory=list)


def load_events(path: Path) -> list[dict[str, Any]]:
    out = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        try:
            d = json.loads(raw)
        except ValueError:
            continue
        if d.get("event") in ("pad_appraisal", "pad_expression", "pad_map"):
            out.append(d)
    return out


def replay(events: list[dict[str, Any]], *, soft: bool, gamma: float) -> dict[str, list[tuple[float, float, float]]]:
    """Per role, the PAD state after every appraisal (the live update path)."""
    from agent.emotion.pad_state import PADState, decay_then_update

    state: dict[str, PADState] = {}
    last: dict[str, float] = {}
    out: dict[str, list[tuple[float, float, float]]] = {}
    for e in events:
        role = e.get("role")
        if not role or role == "human":
            continue
        s = state.setdefault(role, PADState())
        if e["event"] == "pad_appraisal" and e.get("ok") and e.get("delta"):
            dt = e["ts"] - last[role] if role in last else 0.0
            decay_then_update(s, e["delta"], delta_t=dt, half_life=HALF_LIFE_S, soft_bound=soft)
            last[role] = e["ts"]
            out.setdefault(role, []).append(s.as_tuple())
        elif e["event"] == "pad_expression" and gamma > 0:
            s.arousal *= 1.0 - gamma
    return out


def logged_states(events: list[dict[str, Any]]) -> dict[str, list[tuple[float, float, float]]]:
    out: dict[str, list[tuple[float, float, float]]] = {}
    for e in events:
        if e["event"] == "pad_map" and e.get("reason") == "appraisal" and e.get("pad_after"):
            out.setdefault(e["role"], []).append(tuple(e["pad_after"]))
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument("logs", nargs="*", type=Path)
    ap.add_argument("--manifest", type=Path, help="live arms (deploy/arms/pad-ablation.arms) instead of replay")
    ap.add_argument("--since", type=float, default=0.0, help="skip sessions that started earlier (epoch s)")
    ap.add_argument("--min-appraisals", type=int, default=8)
    a = ap.parse_args()
    if a.manifest:
        live_arms(a.manifest)
        return

    from agent.emotion.pad_pipeline import get_pad_knn, speakable_label

    knn = get_pad_knn()
    sessions = []
    for p in a.logs:
        ev = load_events(p)
        n = sum(1 for e in ev if e["event"] == "pad_appraisal" and e.get("ok"))
        if ev and ev[0]["ts"] >= a.since and n >= a.min_appraisals:
            sessions.append((p, ev))
    print(f"{len(sessions)} sessions with ≥ {a.min_appraisals} appraisals")

    # Fidelity: FULL replay vs the logged live states (only sessions whose live run used FULL:
    # they log pad_expression, i.e. γ > 0, and the replay matches step for step).
    errs = []
    for _, ev in sessions:
        live = logged_states(ev)
        rep = replay(ev, soft=True, gamma=0.12)
        for role, xs in live.items():
            ys = rep.get(role, [])
            if len(xs) == len(ys) and any(e["event"] == "pad_expression" for e in ev):
                errs += [max(abs(x - y) for x, y in zip(p, q)) for p, q in zip(xs, ys)]
    if errs:
        print(f"fidelity (FULL replay vs live pad_map): n={len(errs)} states, "
              f"median |err| {statistics.median(errs):.4f}, p95 {sorted(errs)[int(.95 * len(errs))]:.4f}")

    print_header()
    for name, cfg in CONFIGS.items():
        trajs = [Traj(p.name, role, xs) for p, ev in sessions for role, xs in replay(ev, **cfg).items()]
        print_row(name, trajs, knn, speakable_label)
    print_legend()


HEAD = (f"{'config':<9}{'trajs':>6}{'max|P|':>8}{'p95|s|':>8}{'≥.8':>7}{'pinned':>8}"
        f"{'ever≥.8':>9}{'ever=1':>8}{'mean A':>8}{'P range':>9}{'non-Neu':>9}{'labels':>8}")


def print_header(extra: str = "") -> None:
    print(f"\n{HEAD}{extra}")


def print_row(name: str, trajs: list[Traj], knn: Any, speakable_label: Any, extra: str = "") -> None:
    trajs = [t for t in trajs if t.states]
    states = [s for t in trajs for s in t.states]
    if not states:
        return
    near = sum(any(abs(v) >= 0.8 for v in s) for s in states) / len(states)
    pinned = sum(any(abs(v) >= 0.999 for v in s) for s in states) / len(states)
    labels = [speakable_label(knn.query(s).neighbor_labels) if knn else "Neutral" for s in states]
    non_neu = sum(lab != "Neutral" for lab in labels) / len(labels)
    mags = sorted(max(abs(v) for v in s) for s in states)
    ever_near = sum(any(abs(v) >= 0.8 for s in t.states for v in s) for t in trajs) / len(trajs)
    ever_pin = sum(any(abs(v) >= 0.999 for s in t.states for v in s) for t in trajs) / len(trajs)
    print(
        f"{name:<9}{len(trajs):>6}"
        f"{statistics.median(max(abs(s[0]) for s in t.states) for t in trajs):>8.2f}"
        f"{mags[int(.95 * len(mags))]:>8.2f}"
        f"{near:>7.0%}{pinned:>8.0%}{ever_near:>9.0%}{ever_pin:>8.0%}"
        f"{statistics.mean(s[1] for s in states):>8.2f}"
        f"{statistics.median(max(s[0] for s in t.states) - min(s[0] for s in t.states) for t in trajs):>9.2f}"
        f"{non_neu:>9.0%}{len(set(labels)):>8}{extra}"
    )


def print_legend() -> None:
    print("\nmax|P| and P range: median over role trajectories; p95|s|: 95th pct of the largest axis per state;"
          "\n≥.8 / pinned / non-Neu: share of states; ever≥.8 / ever=1: share of trajectories.")


def live_arms(manifest: Path) -> None:
    """Same metrics on the live trajectories of each arm (FULL_a, FULL_b … → FULL),
    plus the share of TTS lines spoken with a non-Neutral label."""
    import csv
    from collections import defaultdict

    from agent.emotion.pad_pipeline import get_pad_knn, speakable_label

    knn = get_pad_knn()
    arms: dict[str, list[Path]] = defaultdict(list)
    with manifest.open(newline="") as f:
        for row in csv.DictReader(f, delimiter="\t"):
            arms[row["arm"].split("_", 1)[0]].append(Path(row["session"]))
    print_header(f"{'TTS emo':>9}{'sess':>6}")
    for name, paths in arms.items():
        trajs: list[Traj] = []
        tts: list[str] = []
        for p in paths:
            if not p.is_file():
                continue
            trajs += [Traj(p.name, r, xs) for r, xs in logged_states(load_events(p)).items()]
            for raw in p.read_text(encoding="utf-8").splitlines():
                d = json.loads(raw)
                if d.get("event") == "tts_synthesize" and d.get("role") != "human":
                    tts.append((d.get("emotion") or "neutral").lower())
        share = sum(e != "neutral" for e in tts) / len(tts) if tts else float("nan")
        print_row(name, trajs, knn, speakable_label, extra=f"{share:>9.0%}{len(paths):>6}")
    print_legend()


if __name__ == "__main__":
    main()
