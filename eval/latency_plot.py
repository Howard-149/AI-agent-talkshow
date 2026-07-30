#!/usr/bin/env python3
"""
Component latency bars from talkshow session JSONL logs.

Usage:
    python eval/latency_plot.py logs/session-*.jsonl
    # default: eval/figures/latency/session-<timecode>_latency.png

Logic (no turn windows):
    Walk the whole session. Each occurrence is one sample:
        host / panel inference – gemma_stt_done / panel_model_done
        TTS                    – tts_synthesize paired with avatar_bake by (role, step)
        1st frame play         – avatar_chunk_play.perceived_wait_ms
                                 (prev playout end → this chunk ready to play first frame;
                                  chunk 0: from line start. Bake-ahead early → ~0)
        gen /frame             – 1/gen_fps (or (motion+render)/N)
        preroll frames         – frames buffered before play (sanity-check wait)
        Speaker selection      – hand_raise_poll (+ flash)
    Chart: top = seconds; bottom = preroll frame counts.
"""

from __future__ import annotations

import argparse
import re
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

from log_parse import ROLE_DISPLAY, load_events
from session_paths import default_figure_path

# "Welcome — I'm Lessac, your host." — never "I'm sure we'll…"
_HOST_NAME_RE = re.compile(
    r"I'm\s+([A-Z][A-Za-z\-']+),\s+your\s+host",
    re.IGNORECASE,
)

# Metric shown as first-frame play latency (perceived wait).
_FIRST_FRAME_PLAY = "1st frame play"
_PREROLL_FRAMES = "preroll frames"


def _label(role: str, suffix: str) -> str:
    name, kind = ROLE_DISPLAY.get(role, (role.capitalize(), role.capitalize()))
    return f"{name} ({kind}) {suffix}"


def _read_role_names(events: list[dict[str, Any]]) -> None:
    """Override display names from greeting lines only (not arbitrary 'I'm …')."""
    for e in events:
        if e.get("event") != "assistant_reply":
            continue
        if e.get("active_role") != "host":
            continue
        text = e.get("text") or ""
        m = _HOST_NAME_RE.search(text)
        if m:
            ROLE_DISPLAY["host"] = (m.group(1), "Host")
            break
    for e in events:
        if e.get("event") == "panel_model_done":
            role = e.get("role", "")
            if role and role not in ROLE_DISPLAY:
                ROLE_DISPLAY[role] = (role.capitalize(), "Guest")


def _ms_to_s(raw: Any) -> float | None:
    if raw is None:
        return None
    try:
        return float(raw) / 1000.0
    except (TypeError, ValueError):
        return None


def _avatar_compose_s(event: dict[str, Any]) -> float | None:
    ms = event.get("avatar_video_compose_ms")
    if ms is None:
        ms = event.get("avatar_bake_ms")
    return _ms_to_s(ms)


def _tts_latency_s(event: dict[str, Any]) -> float | None:
    raw = event.get("tts_latency_s")
    if raw is None and event.get("tts_ms") is not None:
        return _ms_to_s(event["tts_ms"])
    if raw is None:
        return None
    try:
        return float(raw)
    except (TypeError, ValueError):
        return None


def _gen_sec_per_frame(event: dict[str, Any]) -> float | None:
    """Average wall time to produce one frame (seconds)."""
    try:
        gen_fps = float(event.get("gen_fps") or 0)
    except (TypeError, ValueError):
        gen_fps = 0.0
    if gen_fps > 0:
        return 1.0 / gen_fps

    try:
        n = int(event.get("frame_count") or event.get("expected_frames") or 0)
    except (TypeError, ValueError):
        n = 0
    if n <= 0:
        return None
    motion = event.get("motion_ms")
    render = event.get("render_ms")
    if motion is None and render is None:
        return None
    try:
        total_ms = float(motion or 0) + float(render or 0)
    except (TypeError, ValueError):
        return None
    if total_ms <= 0:
        return None
    return (total_ms / 1000.0) / n


def extract_session_metrics(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """One record per occurrence across the whole session (ignore turns)."""
    _read_role_names(events)
    records: list[dict[str, Any]] = []

    def add(metric: str, value: float, source: str, *, unit: str = "s") -> None:
        if value < 0:
            return
        # Allow 0 for 1st frame play (bake-ahead hid the gap).
        if value == 0 and not metric.endswith(_FIRST_FRAME_PLAY):
            return
        records.append(dict(metric=metric, value=value, source=source, unit=unit))

    tts_by_key: dict[tuple[str, str, str], float] = {}
    bake_keys: set[tuple[str, str, str]] = set()
    # Fallback preroll_frames from avatar_bake when chunk_play lacks the field.
    bake_preroll: dict[tuple[str, str, str], int] = {}

    for i, e in enumerate(events):
        source = e.get("_source", "")
        ev = e.get("event")

        if ev == "gemma_stt_done":
            if "model_latency_s" in e:
                try:
                    add(_label("host", "inference"), float(e["model_latency_s"]), source)
                except (TypeError, ValueError):
                    pass
            continue

        if ev == "panel_model_done":
            role = (e.get("role") or "").strip()
            if role:
                try:
                    add(_label(role, "inference"), float(e["model_latency_s"]), source)
                except (KeyError, TypeError, ValueError):
                    pass
            continue

        if ev == "tts_synthesize":
            lat = _tts_latency_s(e)
            if lat is None:
                continue
            role = (e.get("role") or "").strip()
            step = (e.get("step") or "").strip()
            if role and step:
                tts_by_key[(source, role, step)] = lat
            elif role:
                add(_label(role, "TTS"), lat, source)
            continue

        if ev == "avatar_bake":
            role = (e.get("role") or "").strip()
            if not role:
                continue
            step = (e.get("step") or "").strip()
            if step:
                bake_keys.add((source, role, step))
                try:
                    pf = int(e.get("preroll_frames") or 0)
                except (TypeError, ValueError):
                    pf = 0
                if pf > 0:
                    bake_preroll[(source, role, step)] = pf

            gen_s = _gen_sec_per_frame(e)
            if gen_s is not None:
                add(_label(role, "gen /frame"), gen_s, source)

            synth = str(e.get("synth") or "").strip().lower()
            compose_s = _avatar_compose_s(e)
            if (
                compose_s is not None
                and synth in ("frames", "mp4", "bake", "legacy", "")
                and synth != "stream"
            ):
                add(_label(role, "video compose"), compose_s, source)
            continue

        if ev == "avatar_chunk_play":
            role = (e.get("role") or "").strip()
            step = (e.get("step") or "").strip()
            wait_s = _ms_to_s(e.get("perceived_wait_ms"))
            if role and wait_s is not None:
                add(
                    _label(role, _FIRST_FRAME_PLAY),
                    max(wait_s, 0.0),
                    source,
                )

            preroll_n: int | None = None
            try:
                if e.get("preroll_frames") is not None:
                    preroll_n = int(e["preroll_frames"])
            except (TypeError, ValueError):
                preroll_n = None
            if (preroll_n is None or preroll_n <= 0) and role and step:
                preroll_n = bake_preroll.get((source, role, step))
            if role and preroll_n is not None and preroll_n > 0:
                add(
                    _label(role, _PREROLL_FRAMES),
                    float(preroll_n),
                    source,
                    unit="frames",
                )
            continue

        if ev == "hand_raise_poll":
            try:
                poll_s = float(e.get("poll_latency_s") or 0)
            except (TypeError, ValueError):
                poll_s = 0.0
            flash_s = 0.0
            for e_prev in reversed(events[:i]):
                if e_prev.get("_source", "") != source:
                    break
                if e_prev.get("event") == "hand_raise_poll_flash":
                    try:
                        flash_s = float(e_prev.get("flash_sec") or 0)
                    except (TypeError, ValueError):
                        flash_s = 0.0
                    break
            add("Speaker selection", round(poll_s + flash_s, 3), source)

    for key, lat in tts_by_key.items():
        source, role, step = key
        if key not in bake_keys:
            continue
        add(_label(role, "TTS"), lat, source)

    return records


def _role_time_suffixes() -> tuple[str, ...]:
    return (
        "inference",
        "TTS",
        _FIRST_FRAME_PLAY,
        "gen /frame",
        "video compose",
    )


def build_metric_order(df: pd.DataFrame, *, unit: str | None = None) -> list[str]:
    sub = df if unit is None else df[df["unit"] == unit]
    all_metrics = set(sub["metric"].unique())
    order: list[str] = []
    for role in ("host", "commentator", "guest"):
        suffixes = (
            (_PREROLL_FRAMES,)
            if unit == "frames"
            else _role_time_suffixes()
        )
        for suffix in suffixes:
            m = _label(role, suffix)
            if m in all_metrics:
                order.append(m)
    for m in sorted(all_metrics):
        if m not in order and m != "Speaker selection":
            order.append(m)
    if unit != "frames" and "Speaker selection" in all_metrics:
        order.append("Speaker selection")
    return order


def _format_value(v: float, *, unit: str) -> str:
    if unit == "frames":
        return f"{v:.0f}"
    if v >= 10:
        return f"{v:.1f}s"
    if v < 0.01:
        return f"{v * 1000:.1f}ms"
    return f"{v:.2f}s"


def _plot_component_bars(
    ax: plt.Axes,
    comp_df: pd.DataFrame,
    component_order: list[str],
    *,
    multi: bool,
    unit: str,
    xlabel: str,
) -> None:
    if comp_df.empty or not component_order:
        ax.set_visible(False)
        return

    counts = comp_df.groupby("metric").size()
    ytick_labels = [
        f"{m}  (n={int(counts.get(m, 0))})" for m in component_order
    ]

    bar_h = 0.52
    y_pos = list(range(len(component_order)))
    ax.set_yticks(y_pos)
    ax.set_yticklabels(ytick_labels)
    ax.invert_yaxis()

    if multi:
        sources = sorted(comp_df["source"].unique())
        palette = sns.color_palette("tab10", n_colors=len(sources))
        n = len(sources)
        row_h = bar_h / n
        for i, source in enumerate(sources):
            sub = comp_df[comp_df["source"] == source]
            vals = sub.groupby("metric")["value"].mean().reindex(component_order)
            offsets = [y + (i - (n - 1) / 2) * row_h for y in y_pos]
            bars = ax.barh(
                offsets,
                vals.fillna(0).values,
                height=row_h * 0.92,
                color=palette[i],
                label=source,
                alpha=0.9,
            )
            for bar, val in zip(bars, vals.values):
                if pd.isna(val):
                    continue
                ax.text(
                    bar.get_width() + 0.04,
                    bar.get_y() + bar.get_height() / 2,
                    _format_value(float(val), unit=unit),
                    va="center",
                    ha="left",
                    fontsize=8,
                )
        ax.legend(title="Session", loc="lower right", fontsize=8)
        vmax = float(comp_df.groupby(["source", "metric"])["value"].mean().max())
    else:
        vals = comp_df.groupby("metric")["value"].mean().reindex(component_order)
        color = "#0f766e" if unit == "frames" else "#4f46e5"
        bars = ax.barh(
            y_pos,
            vals.fillna(0).values,
            height=bar_h,
            color=color,
            alpha=0.88,
        )
        for bar, val in zip(bars, vals.values):
            if pd.isna(val):
                continue
            ax.text(
                bar.get_width() + 0.04,
                bar.get_y() + bar.get_height() / 2,
                _format_value(float(val), unit=unit),
                va="center",
                ha="left",
                fontsize=9,
                color="#1f2937",
            )
        vmax = float(vals.max()) if len(vals) else 0.0

    pad = max(vmax * 0.18, 0.4 if unit == "s" else 2.0)
    ax.set_xlim(0, vmax + pad)
    ax.set_xlabel(xlabel)
    ax.grid(axis="x", alpha=0.25, linestyle="--")
    ax.xaxis.set_major_locator(mticker.MaxNLocator(nbins=6, prune="lower"))


def plot_swarm(df: pd.DataFrame, out: str | None = None) -> None:
    if df.empty:
        return

    if "unit" not in df.columns:
        df = df.copy()
        df["unit"] = "s"

    multi = df["source"].nunique() > 1
    time_order = build_metric_order(df, unit="s")
    frame_order = build_metric_order(df, unit="frames")
    time_df = df[df["unit"] == "s"]
    frame_df = df[df["unit"] == "frames"]

    n_time = max(len(time_order), 1)
    n_frame = max(len(frame_order), 1)
    has_frames = not frame_df.empty and bool(frame_order)

    if has_frames:
        fig_h = max(5.0, n_time * 0.55 + n_frame * 0.55 + 1.2)
        fig, (ax_t, ax_f) = plt.subplots(
            2,
            1,
            figsize=(10, fig_h),
            gridspec_kw={"height_ratios": [n_time, max(n_frame, 2)]},
        )
    else:
        fig_h = max(4.0, n_time * 0.72)
        fig, ax_t = plt.subplots(figsize=(10, fig_h))
        ax_f = None

    _plot_component_bars(
        ax_t,
        time_df,
        time_order,
        multi=multi,
        unit="s",
        xlabel="Time (seconds) — mean per occurrence",
    )
    ax_t.set_title(
        "Latency — 1st frame play = perceived wait until first frame can play",
        fontsize=11,
        loc="left",
        pad=8,
    )

    if ax_f is not None:
        _plot_component_bars(
            ax_f,
            frame_df,
            frame_order,
            multi=multi,
            unit="frames",
            xlabel="Frames — mean preroll depth (buffered before play)",
        )
        ax_f.set_title(
            "Preroll frames — check against 1st frame play / gen rate",
            fontsize=11,
            loc="left",
            pad=8,
        )

    fig.suptitle("Talkshow Turn Latency", fontsize=13, y=0.99)
    plt.tight_layout(rect=(0, 0.02, 1, 0.97))

    if out:
        out_path = Path(out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(str(out_path), dpi=150, bbox_inches="tight")
        print(f"Saved → {out}")
    else:
        plt.show()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Mean per-occurrence latency bars from talkshow JSONL"
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
    records = extract_session_metrics(events)
    if not records:
        print("No metrics found in the log(s).")
        return

    df = pd.DataFrame(records)
    if "unit" not in df.columns:
        df["unit"] = "s"

    order = build_metric_order(df, unit="s") + build_metric_order(df, unit="frames")
    summary = df.groupby("metric")["value"].agg(
        n="count", mean="mean", std="std", min="min", max="max"
    )
    summary = summary.reindex([m for m in order if m in summary.index])
    print(summary.round(3).to_string())
    print()
    print(
        "Notes: 1st frame play = perceived wait (prev end → ready). "
        "preroll frames = buffered depth before play. "
        "gen /frame = 1/gen_fps."
    )
    print()

    out = args.out or str(
        default_figure_path(_EVAL_DIR, "latency", *args.logs, events=events)
    )
    plot_swarm(df, out=out)


if __name__ == "__main__":
    main()
