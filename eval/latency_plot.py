#!/usr/bin/env python3
"""
Swarm plot of per-turn latency breakdown from talkshow session JSONL logs.

Usage:
    python eval/latency_plot.py logs/session-*.jsonl
    python eval/latency_plot.py logs/session-*.jsonl -o eval/figures/latency.png

Metrics extracted per human turn:
    Total turn time           – gemma_stt_start → panel_done
    Lessac (Host) inference   – Gemma vLLM round-trip for the host reply
    Lessac (Host) TTS         – Piper synthesis of the host reply
    Ryan (Guest) inference    – Gemma vLLM round-trip for commentator
    Ryan (Guest) TTS          – Piper synthesis for commentator
    Amy (Guest) inference     – Gemma vLLM round-trip for guest
    Amy (Guest) TTS           – Piper synthesis for guest
    Hand-raise poll           – LLM calls asking panelists whether to raise hand
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

FIGURES_DIR = Path(__file__).resolve().parent / "figures"

ROLE_DISPLAY = {
    "host": ("Lessac", "Host"),
    "commentator": ("Ryan", "Commentator"),
    "guest": ("Amy", "Guest"),
}


def _label(role: str, suffix: str) -> str:
    name, kind = ROLE_DISPLAY.get(role, (role.capitalize(), role.capitalize()))
    return f"{name} ({kind}) {suffix}"


def load_events(*paths: Path) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    for p in paths:
        with open(p) as f:
            for line in f:
                line = line.strip()
                if line:
                    obj = json.loads(line)
                    obj["_source"] = p.name
                    events.append(obj)
    events.sort(key=lambda e: e["ts"])
    return events


def _read_role_names(events: list[dict[str, Any]]) -> None:
    """Update ROLE_DISPLAY from assistant_reply intro lines if possible."""
    for e in events:
        if e.get("event") != "assistant_reply":
            continue
        text = e.get("text", "")
        role = e.get("active_role", "")
        if role == "host" and "I'm " in text:
            after = text.split("I'm ", 1)[1]
            name = after.split(",")[0].split(".")[0].split(" ")[0].strip()
            if name:
                ROLE_DISPLAY["host"] = (name, "Host")
        if role in ("commentator", "guest") and ", you're up" in text:
            name = text.split(",")[0].strip()
            if name:
                target = "commentator" if name == ROLE_DISPLAY.get("commentator", ("",))[0] else None
                if target is None:
                    target = "guest" if name == ROLE_DISPLAY.get("guest", ("",))[0] else None
    for e in events:
        if e.get("event") == "panel_model_done":
            role = e.get("role", "")
            if role and role not in ROLE_DISPLAY:
                ROLE_DISPLAY[role] = (role.capitalize(), "Guest")


def extract_turn_metrics(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    _read_role_names(events)
    records: list[dict[str, Any]] = []

    stt_starts = [
        (i, e) for i, e in enumerate(events) if e["event"] == "gemma_stt_start"
    ]

    for t, (start_idx, start_event) in enumerate(stt_starts):
        turn_num = t + 1
        start_ts = start_event["ts"]
        source = start_event.get("_source", "")

        end_idx = stt_starts[t + 1][0] if t + 1 < len(stt_starts) else len(events)
        window = events[start_idx:end_idx]

        stt_done = next(
            (e for e in window if e["event"] == "gemma_stt_done"), None
        )
        if stt_done is None:
            continue

        panel_done = next(
            (e for e in window if e["event"] == "panel_done"), None
        )
        last_reply = None
        for e in window:
            if e["event"] == "assistant_reply":
                last_reply = e
        turn_end_ts = (panel_done or last_reply or {}).get("ts")
        if turn_end_ts is None:
            continue

        total = turn_end_ts - start_ts

        def add(metric: str, value: float) -> None:
            records.append(
                dict(metric=metric, value=value, turn=turn_num, source=source)
            )

        add("Total turn time", total)

        # ── Host inference ──────────────────────────────────────────
        if "model_latency_s" in stt_done:
            add(_label("host", "inference"), stt_done["model_latency_s"])

        # ── Host TTS: first tts_synthesize after gemma_stt_done ────
        stt_done_offset = next(
            i for i, e in enumerate(window) if e["event"] == "gemma_stt_done"
        )
        for e in window[stt_done_offset + 1 :]:
            if e["event"] == "tts_synthesize":
                add(_label("host", "TTS"), e["tts_latency_s"])
                break
            if e["event"] == "panel_model_done":
                break

        # ── Panelist inference + TTS ───────────────────────────────
        panel_model_dones = [
            (i, e)
            for i, e in enumerate(window)
            if e["event"] == "panel_model_done"
        ]

        for pm_offset, pm_event in panel_model_dones:
            role = pm_event["role"]
            add(_label(role, "inference"), pm_event["model_latency_s"])

            tts_sum = 0.0
            for e in window[pm_offset + 1 :]:
                if e["event"] == "tts_synthesize":
                    tts_sum += e["tts_latency_s"]
                elif e["event"] == "assistant_reply" and e.get("active_role") == role:
                    break
                elif e["event"] == "panel_model_done":
                    break
            if tts_sum > 0:
                add(_label(role, "TTS"), tts_sum)

        # ── Speaker selection: open-floor → queue pick ─────────────
        # Each cycle starts with "open the floor" speech or open_floor_skip
        # and ends when queue_fifo_pick selects someone (or wait times out).
        for i_w, e in enumerate(window):
            is_open = (
                e["event"] == "open_floor_skip"
                or (
                    e["event"] == "assistant_reply"
                    and "open the floor" in e.get("text", "").lower()
                )
            )
            if not is_open:
                continue
            cycle_start = e["ts"]
            for e2 in window[i_w + 1 :]:
                if e2["event"] == "queue_fifo_pick":
                    add("Speaker selection", round(e2["ts"] - cycle_start, 3))
                    break
                if e2["event"] == "hand_raise_wait_done":
                    add("Speaker selection", round(e2["ts"] - cycle_start, 3))
                    break
                if (
                    e2["event"] == "assistant_reply"
                    and "open the floor" in e2.get("text", "").lower()
                ):
                    break

    return records


def build_metric_order(df: pd.DataFrame) -> list[str]:
    all_metrics = set(df["metric"].unique())
    order = ["Total turn time"]
    order.append(_label("host", "inference"))
    order.append(_label("host", "TTS"))
    roles_seen: list[str] = []
    for role in ("commentator", "guest"):
        inf = _label(role, "inference")
        tts = _label(role, "TTS")
        if inf in all_metrics or tts in all_metrics:
            roles_seen.append(role)
            order.append(inf)
            order.append(tts)
    for m in sorted(all_metrics):
        if m not in order and m != "Speaker selection":
            order.append(m)
    order.append("Speaker selection")
    return [m for m in order if m in all_metrics]


def plot_swarm(df: pd.DataFrame, out: str | None = None) -> None:
    metric_order = build_metric_order(df)
    multi = df["source"].nunique() > 1

    fig, ax = plt.subplots(figsize=(10, max(5, len(metric_order) * 0.9)))

    palette = sns.color_palette("tab10", n_colors=df["source"].nunique())

    sns.swarmplot(
        data=df,
        y="metric",
        x="value",
        order=metric_order,
        hue="source" if multi else None,
        palette=palette if multi else None,
        size=8,
        ax=ax,
    )

    ax.set_xlabel("Time (seconds)")
    ax.set_ylabel("")
    ax.set_title("Talkshow Turn Latency Breakdown")
    ax.grid(axis="x", alpha=0.3)

    if multi:
        ax.legend(title="Session", bbox_to_anchor=(1.02, 1), loc="upper left")

    plt.tight_layout()

    if out:
        out_path = Path(out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(str(out_path), dpi=150, bbox_inches="tight")
        print(f"Saved → {out}")
    else:
        plt.show()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Swarm plot of talkshow turn latencies"
    )
    parser.add_argument("logs", nargs="+", type=Path, help="JSONL log file(s)")
    parser.add_argument(
        "-o",
        "--out",
        type=str,
        default=str(FIGURES_DIR / "latency.png"),
        help="Output image path (default: eval/figures/latency.png)",
    )
    args = parser.parse_args()

    events = load_events(*args.logs)
    records = extract_turn_metrics(events)
    if not records:
        print("No turn metrics found in the log(s).")
        return

    df = pd.DataFrame(records)

    summary = df.groupby("metric")["value"].describe()[
        ["count", "mean", "std", "min", "max"]
    ]
    summary = summary.reindex(build_metric_order(df))
    print(summary.round(3).to_string())
    print()

    plot_swarm(df, out=args.out)


if __name__ == "__main__":
    main()
