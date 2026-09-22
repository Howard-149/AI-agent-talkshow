"""KNN map from continuous PAD → MSP / Plutchik-style emotion labels.

Sentipolis A.2: k=3, Euclidean (minkowski p=2); return all neighbor labels
(not majority vote) so callers can surface multi-label complexity.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np

from agent.emotion.msp_anchors import load_anchor_npz


@dataclass(frozen=True)
class KNNEmotionResult:
    query: tuple[float, float, float]
    neighbor_labels: tuple[str, ...]
    neighbor_distances: tuple[float, ...]
    # Majority for convenience (CosyVoice / closed-set bridging); paper prefers all.
    primary_label: str

    @property
    def label_counts(self) -> dict[str, int]:
        return dict(Counter(self.neighbor_labels))


class PADEmotionKNN:
    def __init__(
        self,
        pad: np.ndarray,
        labels: Sequence[str] | np.ndarray,
        *,
        k: int = 3,
    ) -> None:
        pad = np.asarray(pad, dtype=np.float64)
        if pad.ndim != 2 or pad.shape[1] != 3:
            raise ValueError(f"pad must be (N,3), got {pad.shape}")
        if len(labels) != pad.shape[0]:
            raise ValueError("labels length must match pad rows")
        if k < 1:
            raise ValueError("k must be >= 1")
        self.pad = pad
        self.labels = [str(x) for x in labels]
        self.k = min(k, max(1, pad.shape[0]))

    @classmethod
    def from_npz(cls, path: Path | str, *, k: int = 3) -> PADEmotionKNN:
        pad, labels = load_anchor_npz(path)
        return cls(pad, labels, k=k)

    def query(self, pad: Sequence[float]) -> KNNEmotionResult:
        if len(pad) != 3:
            raise ValueError("pad query must be length 3 [P,A,D]")
        q = np.asarray(pad, dtype=np.float64)
        # Euclidean distance to all anchors
        d = np.linalg.norm(self.pad - q, axis=1)
        k = self.k
        if d.size <= k:
            idx = np.argsort(d)
        else:
            # partial sort for speed on large MSP tables
            part = np.argpartition(d, kth=k - 1)[:k]
            idx = part[np.argsort(d[part])]
        labs = tuple(self.labels[i] for i in idx)
        dists = tuple(float(d[i]) for i in idx)
        primary = Counter(labs).most_common(1)[0][0]
        return KNNEmotionResult(
            query=(float(q[0]), float(q[1]), float(q[2])),
            neighbor_labels=labs,
            neighbor_distances=dists,
            primary_label=primary,
        )
