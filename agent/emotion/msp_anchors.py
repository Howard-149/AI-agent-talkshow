"""Load MSP-PODCAST label CSVs and build PAD→emotion KNN anchors.

Sentipolis (arXiv:2601.18027) uses MSP-Podcast primary emotion + PAD,
normalizes attribute scores from a 1–7 (paper: 0–7) scale into [-1, 1],
then runs KNN (k=3, Euclidean) for semantic enrichment.
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

DEFAULT_MSP_LABELS_CSV = (
    "/data/user_data/hsuanhal/MSP-PODCAST-Publish-2.0/Labels/labels_detailed.csv"
)

# MSP consensus single-letter codes → Sentipolis-style names.
# "Other" / "No Agreement" → Vague (paper A.2).
EMO_CLASS_MAP: dict[str, str] = {
    "A": "Anger",
    "H": "Happiness",
    "S": "Sadness",
    "U": "Surprise",
    "F": "Fear",
    "D": "Disgust",
    "C": "Contempt",
    "N": "Neutral",
    "O": "Vague",
    "X": "Vague",
    "OTHER": "Vague",
    "NO AGREEMENT": "Vague",
    "NO_AGREEMENT": "Vague",
    "ANGER": "Anger",
    "ANGRY": "Anger",
    "HAPPINESS": "Happiness",
    "HAPPY": "Happiness",
    "SADNESS": "Sadness",
    "SAD": "Sadness",
    "SURPRISE": "Surprise",
    "SURPRISED": "Surprise",
    "FEAR": "Fear",
    "DISGUST": "Disgust",
    "CONTEMPT": "Contempt",
    "NEUTRAL": "Neutral",
    "VAGUE": "Vague",
}


def normalize_pad_1_to_7(x: float, *, lo: float = 1.0, hi: float = 7.0) -> float:
    """Map MSP attribute score on [lo, hi] (center 4) to [-1, 1]."""
    if hi <= lo:
        raise ValueError(f"invalid range lo={lo} hi={hi}")
    t = (float(x) - lo) / (hi - lo)
    return float(np.clip(2.0 * t - 1.0, -1.0, 1.0))


def canonicalize_emotion_label(raw: Any) -> str:
    if raw is None or (isinstance(raw, float) and np.isnan(raw)):
        return "Vague"
    s = str(raw).strip()
    if not s:
        return "Vague"
    # "Other-<emotion>" → keep the free-text emotion after the prefix.
    lower = s.lower()
    if lower.startswith("other-") or lower.startswith("other "):
        suffix = s.split("-", 1)[-1].strip() if "-" in s else s[5:].strip()
        if not suffix or suffix.lower() in {"other", "o", "x"}:
            return "Vague"
        # Recurse so "Other-Happy" → Happiness, "Other-Concerned" → Concerned.
        return canonicalize_emotion_label(suffix)
    key = s.upper()
    if key in EMO_CLASS_MAP:
        return EMO_CLASS_MAP[key]
    # Title-case full names already matching values
    titled = s[:1].upper() + s[1:].lower()
    if titled in EMO_CLASS_MAP.values():
        return titled
    return EMO_CLASS_MAP.get(key, "Vague" if key in {"O", "X"} else titled)


def load_labels_csv(path: Path | str):
    import pandas as pd

    path = Path(path)
    return pd.read_csv(path)


def discover_columns(df) -> dict[str, str | None]:
    """Best-effort column map for consensus or detailed MSP label tables."""
    cols = {c: c for c in df.columns}
    lower = {c.lower().strip(): c for c in df.columns}

    def pick(*names: str) -> str | None:
        for n in names:
            if n in cols:
                return n
            if n.lower() in lower:
                return lower[n.lower()]
        return None

    return {
        "valence": pick("EmoVal", "valence", "Valence", "V", "val"),
        "arousal": pick("EmoAct", "arousal", "Arousal", "activation", "Act", "A"),
        "dominance": pick("EmoDom", "dominance", "Dominance", "Dom", "D"),
        # Prefer Major/Primary before bare EmoClass (labels_detailed has no EmoClass).
        "emotion": pick(
            "EmoClass_Major",
            "EmoClass_Primary",
            "EmoClass",
            "primary_emotion",
            "Primary",
            "EmoPrimary",
            "emotion",
            "Emotion",
            "label",
        ),
        "emotion_second": pick(
            "EmoClass_Second",
            "EmoClass_Secondary",
            "secondary_emotion",
            "Secondary",
        ),
        "filename": pick("FileName", "filename", "file", "wav"),
        "worker": pick("WorkerID", "worker", "AnnotatorID"),
    }


def build_anchors_from_dataframe(
    df,
    cols: dict[str, str | None] | None = None,
    *,
    max_rows: int | None = None,
) -> tuple[np.ndarray, list[str]]:
    """
    Returns
    -------
    pad : (N, 3) float64 in [-1, 1] as [P, A, D]  (Pleasure←valence)
    labels : list[str] length N
    """
    cols = cols or discover_columns(df)
    vcol = cols.get("valence")
    acol = cols.get("arousal")
    dcol = cols.get("dominance")
    ecol = cols.get("emotion")
    if not vcol or not acol:
        raise ValueError(f"need valence+arousal columns; got {cols}")

    pad_rows: list[list[float]] = []
    labels: list[str] = []
    for i, row in df.iterrows():
        if max_rows is not None and len(labels) >= max_rows:
            break
        try:
            v = float(row[vcol])
            a = float(row[acol])
        except (TypeError, ValueError):
            continue
        if np.isnan(v) or np.isnan(a):
            continue
        if dcol is not None:
            try:
                d = float(row[dcol])
            except (TypeError, ValueError):
                d = 4.0
            if np.isnan(d):
                d = 4.0
        else:
            d = 4.0
        p = normalize_pad_1_to_7(v)
        ar = normalize_pad_1_to_7(a)
        dom = normalize_pad_1_to_7(d)
        lab = canonicalize_emotion_label(row[ecol]) if ecol else "Unknown"
        pad_rows.append([p, ar, dom])
        labels.append(lab)

    if not pad_rows:
        return np.zeros((0, 3), dtype=np.float64), []
    return np.asarray(pad_rows, dtype=np.float64), labels


def summarize_anchors(pad: np.ndarray, labels: list[str]) -> dict[str, Any]:
    counts = Counter(labels)
    return {
        "n": int(len(labels)),
        "label_counts": dict(counts.most_common()),
        "pad_mean": pad.mean(axis=0).tolist() if len(pad) else [0.0, 0.0, 0.0],
        "pad_std": pad.std(axis=0).tolist() if len(pad) else [0.0, 0.0, 0.0],
    }


def load_anchor_npz(path: Path | str) -> tuple[np.ndarray, np.ndarray]:
    path = Path(path)
    data = np.load(path, allow_pickle=True)
    pad = np.asarray(data["pad"], dtype=np.float64)
    labels = np.asarray(data["labels"], dtype=object)
    return pad, labels
