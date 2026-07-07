#!/usr/bin/env python3
"""
Swarm/strip plot of per-turn latency breakdown from talkshow session JSONL logs.
Component metrics as horizontal bars (seconds labeled at bar end).

Usage:
    python eval/latency_plot.py logs/session-*.jsonl
    python eval/latency_plot.py logs/session-*.jsonl
    # default: eval/figures/latency/session-<timecode>_latency.png

Metrics extracted per human turn:
    Total turn time           – gemma_stt_start → panel_done (summary only, not plotted)
    Lessac (Host) inference   – Gemma vLLM round-trip for the host reply
    Lessac (Host) TTS         – Piper synthesis of the host reply
    Ryan (Guest) inference    – Gemma vLLM round-trip for commentator
    Ryan (Guest) TTS          – Piper synthesis for commentator
    Amy (Guest) inference     – Gemma vLLM round-trip for guest
    Amy (Guest) TTS           – Piper synthesis for guest
    Hand-raise poll           – LLM hand-raise poll + UI flash (poll_latency_s + flash_sec)
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

_EVAL_DIR = Path(__file__).resolve().parent
if str(_EVAL_DIR) not in sys.path:
    sys.path.insert(0, str(_EVAL_DIR))

import matplotlib.ticker as mticker
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from session_paths import default_figure_path
from log_parse import turn_window_end

ROLE_DISPLAY = {
    "host": ("Lessac", "Host"),
    "commentator": ("Ryan", "Commentator"),
    "guest": ("Amy", "Guest"),
}

# Wall-clock turn total — kept in JSONL metrics for summaries, excluded from charts.
TOTAL_TURN_METRIC = "Total turn time"


def _label(role: str, suffix: str) -> str:
    name, kind = ROLE_DISPLAY.get(role, (role.capitalize(), role.capitalize()))
    return f"{name} ({kind}) {suffix}"


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
    turn_counters: dict[str, int] = {}

    for start_idx, start_event in stt_starts:
        source = start_event.get("_source", "")
        end_idx = turn_window_end(events, start_idx, source)
        window = events[start_idx:end_idx]
        turn_counters[source] = turn_counters.get(source, 0) + 1
        turn_num = turn_counters[source]
        start_ts = start_event["ts"]

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

        add(TOTAL_TURN_METRIC, total)

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

        # ── Speaker selection: hand-raise poll + UI flash ──────────
        for i_w, e in enumerate(window):
            if e.get("_source") != source or e["event"] != "hand_raise_poll":
                continue
            try:
                poll_s = float(e.get("poll_latency_s") or 0)
            except (TypeError, ValueError):
                poll_s = 0.0
            flash_s = 0.0
            for e_prev in reversed(window[:i_w]):
                if e_prev.get("_source") != source:
                    break
                if e_prev["event"] == "hand_raise_poll_flash":
                    try:
                        flash_s = float(e_prev.get("flash_sec") or 0)
                    except (TypeError, ValueError):
                        flash_s = 0.0
                    break
            total = poll_s + flash_s
            if total > 0:
                add("Speaker selection", round(total, 3))

    return records


def build_metric_order(df: pd.DataFrame) -> list[str]:
    all_metrics = set(df["metric"].unique()) - {TOTAL_TURN_METRIC}
    order: list[str] = []
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


def _format_seconds(v: float) -> str:
    if v >= 10:
        return f"{v:.1f}s"
    return f"{v:.2f}s"


def _plot_component_bars(
    ax: plt.Axes,
    comp_df: pd.DataFrame,
    component_order: list[str],
    *,
    multi: bool,
) -> None:
    """Horizontal bars — one row per metric; mean if multiple samples."""
    if comp_df.empty:
        return

    bar_h = 0.52
    y_pos = list(range(len(component_order)))
    ax.set_yticks(y_pos)
    ax.set_yticklabels(component_order)
    ax.invert_yaxis()

    if multi:
        sources = sorted(comp_df["source"].unique())
        palette = sns.color_palette("tab10", n_colors=len(sources))
        n = len(sources)
        row_h = bar_h / n
        for i, source in enumerate(sources):
            sub = comp_df[comp_df["source"] == source]
            means = sub.groupby("metric")["value"].mean().reindex(component_order)
            offsets = [y + (i - (n - 1) / 2) * row_h for y in y_pos]
            bars = ax.barh(
                offsets,
                means.fillna(0).values,
                height=row_h * 0.92,
                color=palette[i],
                label=source,
                alpha=0.9,
            )
            for bar, val in zip(bars, means.values):
                if pd.isna(val):
                    continue
                ax.text(
                    bar.get_width() + 0.04,
                    bar.get_y() + bar.get_height() / 2,
                    _format_seconds(float(val)),
                    va="center",
                    ha="left",
                    fontsize=8,
                )
        ax.legend(title="Session", loc="lower right", fontsize=8)
        vmax = float(comp_df["value"].max())
    else:
        means = comp_df.groupby("metric")["value"].mean().reindex(component_order)
        bars = ax.barh(
            y_pos,
            means.fillna(0).values,
            height=bar_h,
            color="#4f46e5",
            alpha=0.88,
        )
        for bar, val in zip(bars, means.values):
            if pd.isna(val):
                continue
            ax.text(
                bar.get_width() + 0.04,
                bar.get_y() + bar.get_height() / 2,
                _format_seconds(float(val)),
                va="center",
                ha="left",
                fontsize=9,
                color="#1f2937",
            )
        vmax = float(means.max())

    pad = max(vmax * 0.18, 0.4)
    ax.set_xlim(0, vmax + pad)
    ax.set_xlabel("Time (seconds)")
    ax.grid(axis="x", alpha=0.25, linestyle="--")
    ax.xaxis.set_major_locator(mticker.MaxNLocator(nbins=6, prune="lower"))


def plot_swarm(df: pd.DataFrame, out: str | None = None) -> None:
    df = df[df["metric"] != TOTAL_TURN_METRIC]
    if df.empty:
        return

    metric_order = build_metric_order(df)
    multi = df["source"].nunique() > 1
    comp_df = df[df["metric"].isin(metric_order)]

    n_metrics = max(len(metric_order), 1)
    fig_h = max(4.0, n_metrics * 0.72)
    fig, ax = plt.subplots(figsize=(10, fig_h))

    _plot_component_bars(ax, comp_df, metric_order, multi=multi)
    ax.set_title("Component latency", fontsize=12, loc="left", pad=10)

    fig.suptitle("Talkshow Turn Latency", fontsize=13, y=0.98)
    plt.tight_layout(rect=(0, 0.04, 1, 0.96))

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
        default=None,
        help="Output image path (default: eval/figures/latency/session-<timecode>_latency.png)",
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

    out = args.out or str(
        default_figure_path(_EVAL_DIR, "latency", *args.logs, events=events)
    )
    plot_swarm(df, out=out)


if __name__ == "__main__":
    main()
