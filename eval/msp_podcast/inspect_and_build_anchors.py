"""Build PAD emotion anchors from MSP-PODCAST labels (Sentipolis-style KNN).

Default path (Babel, compute nodes only — /data is not mounted on login)::

    /data/user_data/hsuanhal/MSP-PODCAST-Publish-2.0/Labels/labels_consensus.csv

Use the consensus table (FileName,EmoClass,EmoAct,EmoVal,EmoDom; 264,705 rows as in
Sentipolis A.2). ``labels_detailed.csv`` is per-annotator with integer ratings and
collapses onto 343 grid points — the builder refuses it unless --allow-per-annotator.

"No Agreement" (X) rows are resolved from per-utterance vote counts in
``labels_detailed_aggregated.csv`` (``--votes-csv``): exactly two emotions tied for first
→ 50/50 soft label; single top emotion → that emotion; ≥3 tied → Vague. "Other" (O)
stays Vague. Pass ``--votes-csv ''`` for the paper's plain mapping (all X/O → Vague).

Usage::

    PYTHONPATH=. python eval/msp_podcast/inspect_and_build_anchors.py
    PYTHONPATH=. python eval/msp_podcast/inspect_and_build_anchors.py \\
      --csv /path/to/labels_consensus.csv \\
      --out agent/emotion/data/msp_pad_anchors.npz
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

# Repo root on PYTHONPATH
from agent.emotion.msp_anchors import (
    DEFAULT_MSP_LABELS_CSV,
    DEFAULT_MSP_VOTES_CSV,
    build_anchors_from_dataframe,
    canonicalize_emotion_label,
    discover_columns,
    load_labels_csv,
    normalize_pad_1_to_7,
    soft_labels_from_votes,
    summarize_anchors,
)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--csv",
        type=Path,
        default=Path(DEFAULT_MSP_LABELS_CSV),
        help="MSP labels_consensus.csv (per-utterance consensus)",
    )
    p.add_argument(
        "--votes-csv",
        type=str,
        default=DEFAULT_MSP_VOTES_CSV,
        help="labels_detailed_aggregated.csv for No-Agreement soft labels ('' = all Vague)",
    )
    p.add_argument(
        "--allow-per-annotator",
        action="store_true",
        help="Permit a per-annotator table (labels_detailed.csv); not recommended",
    )
    p.add_argument(
        "--out",
        type=Path,
        default=Path("agent/emotion/data/msp_pad_anchors.npz"),
        help="Output NPZ (pad float32 Nx3 + labels object array)",
    )
    p.add_argument(
        "--summary-json",
        type=Path,
        default=None,
        help="Optional JSON summary path (default: alongside --out)",
    )
    p.add_argument(
        "--max-rows",
        type=int,
        default=0,
        help="If >0, only use first N usable rows (debug)",
    )
    args = p.parse_args(argv)

    csv_path = args.csv
    if not csv_path.is_file():
        print(f"FATAL: CSV not found: {csv_path}", file=sys.stderr)
        print(
            "On Babel this should exist under MSP-PODCAST-Publish-2.0/Labels/.",
            file=sys.stderr,
        )
        return 1

    print(f"Loading {csv_path} …")
    df = load_labels_csv(csv_path)
    print(f"rows={len(df)} cols={list(df.columns)}")

    cols = discover_columns(df)
    print(f"discovered columns: {cols}")
    if cols.get("valence") is None or cols.get("arousal") is None:
        print(
            "FATAL: could not find valence/arousal columns "
            "(expected EmoVal/EmoAct or valence/arousal).",
            file=sys.stderr,
        )
        return 1

    # Quick inspect of label / range stats
    vcol, acol = cols["valence"], cols["arousal"]
    dcol = cols.get("dominance")
    ecol = cols.get("emotion")
    ecol2 = cols.get("emotion_second")
    worker = cols.get("worker")
    if worker:
        msg = (
            f"per-annotator table (worker={worker}); n={len(df)} rows. Integer ratings "
            f"collapse onto ≤343 PAD grid points. Use labels_consensus.csv (paper: 264,705)."
        )
        if not args.allow_per_annotator:
            print(f"FATAL: {msg}", file=sys.stderr)
            return 1
        print(f"WARN: {msg}")
    if ecol2:
        print(f"secondary emotion column present: {ecol2} (anchors use primary only)")
    print(
        f"valence[{vcol}] min={df[vcol].min()} max={df[vcol].max()} "
        f"mean={df[vcol].mean():.3f}"
    )
    print(
        f"arousal[{acol}] min={df[acol].min()} max={df[acol].max()} "
        f"mean={df[acol].mean():.3f}"
    )
    if dcol:
        print(
            f"dominance[{dcol}] min={df[dcol].min()} max={df[dcol].max()} "
            f"mean={df[dcol].mean():.3f}"
        )
    else:
        print("WARN: no dominance column — will use 0.0 for D")
    if ecol:
        raw_counts = Counter(str(x) for x in df[ecol].dropna().tolist())
        print(f"emotion[{ecol}] top: {raw_counts.most_common(15)}")
        mapped = Counter(
            canonicalize_emotion_label(x) for x in df[ecol].dropna().tolist()
        )
        print(f"mapped label counts: {mapped.most_common(15)}")
    else:
        print("WARN: no emotion/class column — anchors will use label='Unknown'")

    # Show one normalized sample
    sample_v = float(df[vcol].iloc[0])
    print(
        f"normalize example: raw_valence={sample_v} → "
        f"{normalize_pad_1_to_7(sample_v):.4f} (1–7 → [-1,1])"
    )

    pad, labels = build_anchors_from_dataframe(df, cols, max_rows=args.max_rows or None)
    print(f"usable anchors: n={len(labels)}")
    if len(labels) == 0:
        print("FATAL: zero usable rows after filtering NaN", file=sys.stderr)
        return 1

    soft = [((lab, 1.0),) for lab in labels]
    if args.votes_csv and not args.max_rows:
        votes_path = Path(args.votes_csv)
        if not votes_path.is_file():
            print(f"FATAL: votes CSV not found: {votes_path}", file=sys.stderr)
            return 1
        print(f"Loading votes {votes_path} …")
        df_votes = load_labels_csv(votes_path)
        usable = df.dropna(subset=[c for c in (vcol, acol) if c]).reset_index(drop=True)
        if len(usable) != len(labels):
            print("FATAL: row alignment mismatch with votes table", file=sys.stderr)
            return 1
        soft = soft_labels_from_votes(usable, df_votes)
        kinds = Counter(
            "mix" if len(s) == 2 else ("vague" if s[0][0] == "Vague" else "single")
            for s, raw in zip(soft, usable[ecol]) if str(raw).strip().upper() == "X"
        )
        print(f"No Agreement resolved: {dict(kinds)}")
        pairs = Counter("/".join(sorted(n for n, _ in s)) for s in soft if len(s) == 2)
        print(f"top mixes: {pairs.most_common(8)}")
    labels = [s[0][0] for s in soft]

    uniq, counts = np.unique(pad.round(4), axis=0, return_counts=True)
    print(
        f"distinct PAD coordinates: {len(uniq)} "
        f"(largest stack {counts.max()} anchors at one point; "
        f"{(counts > 1).sum()} points shared by ≥2 anchors)"
    )

    summary = summarize_anchors(pad, labels)
    weighted: Counter = Counter()
    for s in soft:
        for name, w in s:
            weighted[name] += w
    summary["weighted_label_counts"] = {k: round(v, 1) for k, v in weighted.most_common()}
    summary["mixed_anchors"] = int(sum(1 for s in soft if len(s) == 2))
    print(f"weighted label histogram: {summary['weighted_label_counts']}")
    summary["distinct_coords"] = int(len(uniq))
    summary["max_stack"] = int(counts.max())
    print(f"label histogram: {summary['label_counts']}")
    print(
        f"PAD means P={summary['pad_mean'][0]:.3f} "
        f"A={summary['pad_mean'][1]:.3f} D={summary['pad_mean'][2]:.3f}"
    )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.out,
        pad=pad.astype(np.float32),
        labels=np.asarray(labels, dtype=object),
        labels_b=np.asarray([s[1][0] if len(s) == 2 else "" for s in soft], dtype=object),
        weights_a=np.asarray([s[0][1] for s in soft], dtype=np.float32),
        source_csv=str(csv_path),
        columns_json=json.dumps(cols),
    )
    print(f"wrote {args.out}")

    summary_path = args.summary_json or args.out.with_suffix(".json")
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    print(f"wrote {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
