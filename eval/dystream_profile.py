#!/usr/bin/env python3
"""Summarize DyStream per-stage synth profile from session JSONL.

Usage:
    python eval/dystream_profile.py logs/session-*.jsonl

Reads ``avatar_synth_profile`` events (sidecar stage split):
  audio2face     clip-level wav2vec (once)
  ar_a2f_inner   wav2vec inside one_clip (usually redundant)
  ar_attn        GPT transformer / attention
  fm_net         flow-matching DiffusionHead
  fm_ode         scheduler.step (Euler)
  ar_other       rest of one_clip
  vis_flow       appearance flow estimator
  face_gen       face generator
  xfer           GPU→CPU RGBA pack
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

_EVAL_DIR = Path(__file__).resolve().parent
if str(_EVAL_DIR) not in sys.path:
    sys.path.insert(0, str(_EVAL_DIR))

import matplotlib.pyplot as plt
import pandas as pd

from log_parse import ROLE_DISPLAY, load_events
from session_paths import default_figure_path

_STAGE_MS = (
    ("audio_load_ms", "audio_load"),
    ("audio2face_ms", "audio2face"),
    ("ar_a2f_inner_ms", "a2f_inner"),
    ("ar_attn_ms", "ar_attn"),
    ("fm_net_ms", "fm_net"),
    ("fm_ode_ms", "fm_ode"),
    ("ar_other_ms", "ar_other"),
    ("vis_flow_ms", "vis_flow"),
    ("face_gen_ms", "face_gen"),
    ("xfer_ms", "xfer"),
)

_STAGE_PF = (
    ("ms_per_frame_ar_a2f_inner", "a2f_inner (dup wav2vec)"),
    ("ms_per_frame_ar_attn", "ar_attn (GPT)"),
    ("ms_per_frame_fm_net", "fm_net (DiffusionHead)"),
    ("ms_per_frame_fm_ode", "fm_ode (Euler)"),
    ("ms_per_frame_ar_other", "ar_other"),
    ("ms_per_frame_vis_flow", "vis_flow"),
    ("ms_per_frame_face_gen", "face_gen"),
    ("ms_per_frame_xfer", "xfer"),
)


def _rows(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for e in events:
        if e.get("event") != "avatar_synth_profile":
            continue
        role = str(e.get("role") or "?")
        name, kind = ROLE_DISPLAY.get(role, (role.capitalize(), role))
        frames = int(e.get("frames") or 0)
        row: dict[str, Any] = {
            "role": f"{name} ({kind})",
            "frames": frames,
            "steps": e.get("steps"),
            "clock": e.get("clock"),
            "step": e.get("step"),
        }
        for key, _label in _STAGE_MS:
            try:
                row[key] = float(e.get(key) or 0)
            except (TypeError, ValueError):
                row[key] = 0.0
        for key, _label in _STAGE_PF:
            try:
                row[key] = float(e.get(key) or 0)
            except (TypeError, ValueError):
                row[key] = 0.0
        out.append(row)
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("logs", nargs="+", type=Path)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()
    events = load_events(*args.logs)
    rows = _rows(events)
    if not rows:
        print("No avatar_synth_profile events. Restart DyStream sidecar + agent after sync.")
        return 1

    df = pd.DataFrame(rows)
    pf_cols = [k for k, _ in _STAGE_PF]
    means = df[pf_cols].mean()
    print("ms / frame (mean across clips)")
    print(means.rename(dict(_STAGE_PF)).to_string())
    print()
    total = float(means.sum()) or 1.0
    print("share of per-frame AR+render")
    for key, label in _STAGE_PF:
        v = float(means[key])
        print(f"  {label:28s} {v:7.2f} ms  {100.0 * v / total:5.1f}%")
    print()
    print(f"n_clips={len(df)}  frames_sum={int(df['frames'].sum())}  clock={df['clock'].mode().iloc[0]}")

    labels = [lab for _, lab in _STAGE_PF]
    vals = [float(means[k]) for k, _ in _STAGE_PF]
    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    colors = [
        "#9e9ac8",
        "#4c78a8",
        "#f58518",
        "#b279a2",
        "#bab0ac",
        "#54a24b",
        "#e45756",
        "#72b7b2",
    ]
    ax.barh(labels[::-1], vals[::-1], color=colors[::-1])
    ax.set_xlabel("ms / frame")
    ax.set_title("DyStream synth breakdown (mean)")
    fig.tight_layout()
    out = args.out or default_figure_path(
        _EVAL_DIR, "dystream", *args.logs, events=events
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=140)
    print(f"\nSaved → {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
