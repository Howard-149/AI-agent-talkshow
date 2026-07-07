#!/usr/bin/env python3
"""
Panel reply length stats from talkshow session JSONL logs.

Compares character (and word) counts per panelist — e.g. Ryan shorter, Amy longer.

Usage:
    python eval/reply_length_plot.py logs/session-*.jsonl
    # default plot: eval/figures/reply_length/session-<timecode>_reply_length.png
    # default JSON: eval/results/reply_length/session-<timecode>_reply_length.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

_EVAL_DIR = Path(__file__).resolve().parent
if str(_EVAL_DIR) not in sys.path:
    sys.path.insert(0, str(_EVAL_DIR))

from log_parse import (
    extract_human_turns,
    flatten_panel_replies,
    load_events,
    role_label,
    role_name,
    session_metadata,
)
from session_paths import default_figure_path, default_result_path

PANEL_ROLES = ("commentator", "guest")


def _word_count(text: str) -> int:
    return len(text.split())


def records_to_rows(replies: list) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for pr in replies:
        if pr.role not in PANEL_ROLES:
            continue
        rows.append(
            {
                "role": pr.role,
                "speaker": role_name(pr.role),
                "label": role_label(pr.role),
                "reply_len": pr.reply_len,
                "word_count": _word_count(pr.reply),
                "turn_num": pr.turn_num,
                "step": pr.step,
                "source": pr.source,
                "seq": pr.seq,
                "reply": pr.reply,
            }
        )
    return rows


def summarize_by_role(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    agg = df.groupby(["role", "speaker"], as_index=False).agg(
        count=("reply_len", "count"),
        chars_mean=("reply_len", "mean"),
        chars_median=("reply_len", "median"),
        chars_std=("reply_len", "std"),
        chars_min=("reply_len", "min"),
        chars_max=("reply_len", "max"),
        words_mean=("word_count", "mean"),
        words_median=("word_count", "median"),
    )
    for col in ("chars_mean", "chars_median", "chars_std", "words_mean", "words_median"):
        if col in agg.columns:
            agg[col] = agg[col].round(1)
    return agg


def print_summary(df: pd.DataFrame, summary: pd.DataFrame) -> None:
    if df.empty:
        print("No panel replies found.")
        return

    print(f"Panel replies: {len(df)}\n")
    print("By role (characters):")
    print(summary.to_string(index=False))
    print()

    roles = [r for r in PANEL_ROLES if r in set(df["role"])]
    if len(roles) == 2:
        a, b = roles
        mean_a = float(df.loc[df["role"] == a, "reply_len"].mean())
        mean_b = float(df.loc[df["role"] == b, "reply_len"].mean())
        ratio = mean_b / mean_a if mean_a else float("nan")
        print(
            f"Mean length ratio {role_name(b)} / {role_name(a)}: "
            f"{ratio:.2f}x ({mean_b:.0f} vs {mean_a:.0f} chars)"
        )


def plot_lengths(df: pd.DataFrame, out: str | None) -> None:
    if df.empty:
        return

    order = [role_label(r) for r in PANEL_ROLES if r in set(df["role"])]
    plot_df = df.copy()
    plot_df["label"] = plot_df["role"].map(
        lambda r: role_label(r)  # type: ignore[arg-type]
    )

    fig, axes = plt.subplots(1, 2, figsize=(11, max(4, len(order) * 1.2)))

    sns.boxplot(
        data=plot_df,
        y="label",
        x="reply_len",
        order=order,
        ax=axes[0],
        color="#93c5fd",
        width=0.5,
        fliersize=3,
    )
    sns.stripplot(
        data=plot_df,
        y="label",
        x="reply_len",
        order=order,
        ax=axes[0],
        color="#1e3a8a",
        alpha=0.75,
        size=6,
        jitter=0.15,
    )
    axes[0].set_xlabel("Characters per reply")
    axes[0].set_ylabel("")
    axes[0].set_title("Reply length (chars)")

    sns.boxplot(
        data=plot_df,
        y="label",
        x="word_count",
        order=order,
        ax=axes[1],
        color="#fcd34d",
        width=0.5,
        fliersize=3,
    )
    sns.stripplot(
        data=plot_df,
        y="label",
        x="word_count",
        order=order,
        ax=axes[1],
        color="#92400e",
        alpha=0.75,
        size=6,
        jitter=0.15,
    )
    axes[1].set_xlabel("Words per reply")
    axes[1].set_ylabel("")
    axes[1].set_title("Reply length (words)")

    fig.suptitle("Panel Reply Length by Speaker", fontsize=13)
    plt.tight_layout()

    if out:
        out_path = Path(out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(str(out_path), dpi=150, bbox_inches="tight")
        print(f"Saved → {out_path}")
    else:
        plt.show()


def build_export(
    rows: list[dict[str, Any]],
    summary: pd.DataFrame,
    meta: dict[str, Any],
) -> dict[str, Any]:
    return {
        "session": meta,
        "panel_reply_count": len(rows),
        "summary_by_role": summary.to_dict(orient="records"),
        "replies": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Panel reply length stats and plot"
    )
    parser.add_argument("logs", nargs="+", type=Path, help="Session JSONL log file(s)")
    parser.add_argument(
        "-o",
        "--out",
        type=str,
        default=None,
        help="Plot output (default: eval/figures/reply_length/session-<timecode>_reply_length.png)",
    )
    parser.add_argument(
        "--json",
        type=Path,
        default=None,
        help="JSON export (default: eval/results/reply_length/session-<timecode>_reply_length.json)",
    )
    parser.add_argument("--no-plot", action="store_true")
    parser.add_argument("--no-json", action="store_true")
    args = parser.parse_args()

    events = load_events(*args.logs)
    meta = session_metadata(events)
    turns = extract_human_turns(events)
    replies = flatten_panel_replies(turns)
    rows = records_to_rows(replies)
    df = pd.DataFrame(rows)
    summary = summarize_by_role(df)

    print_summary(df, summary)

    if not args.no_json:
        json_path = args.json or default_result_path(
            _EVAL_DIR, "reply_length", *args.logs, events=events
        )
        json_path.parent.mkdir(parents=True, exist_ok=True)
        payload = build_export(rows, summary, meta)
        json_path.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        print(f"\nWrote JSON → {json_path}")

    if not args.no_plot and not df.empty:
        out = args.out or str(
            default_figure_path(
                _EVAL_DIR, "reply_length", *args.logs, events=events
            )
        )
        plot_lengths(df, out)


if __name__ == "__main__":
    main()
