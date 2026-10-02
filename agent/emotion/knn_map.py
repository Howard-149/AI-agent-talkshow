"""KNN map from continuous PAD → MSP / Plutchik-style emotion labels.

Sentipolis A.2: k=3, Euclidean (minkowski p=2); return all neighbor labels
(not majority vote) so callers can surface multi-label complexity.
Ours: anchors may carry soft labels (MSP "No Agreement" → two-label mix), and
distance ties at the k-th neighbor are resolved by the tied group's vote shares.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np

from agent.emotion.msp_anchors import load_anchor_npz_soft


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

    @property
    def top_labels(self) -> tuple[str, ...]:
        """All labels tied for the most votes, nearest first (a k=3 three-way tie → 3)."""
        return modal_labels(self.neighbor_labels)


def modal_labels(labels: Sequence[str]) -> tuple[str, ...]:
    """Labels tied for the highest count, in first-seen (nearest-first) order."""
    counts = Counter(labels)
    if not counts:
        return ()
    best = max(counts.values())
    return tuple(dict.fromkeys(x for x in labels if counts[x] == best))


SoftLabel = tuple[tuple[str, float], ...]


class PADEmotionKNN:
    def __init__(
        self,
        pad: np.ndarray,
        labels: Sequence[str] | np.ndarray,
        *,
        k: int = 3,
        soft_labels: Sequence[SoftLabel] | None = None,
    ) -> None:
        pad = np.asarray(pad, dtype=np.float64)
        if pad.ndim != 2 or pad.shape[1] != 3:
            raise ValueError(f"pad must be (N,3), got {pad.shape}")
        if len(labels) != pad.shape[0]:
            raise ValueError("labels length must match pad rows")
        if soft_labels is not None and len(soft_labels) != pad.shape[0]:
            raise ValueError("soft_labels length must match pad rows")
        if k < 1:
            raise ValueError("k must be >= 1")
        self.pad = pad
        self.labels = [str(x) for x in labels]
        # Each anchor votes with total weight 1, split across its labels (e.g. an MSP
        # "No Agreement" 2-2-1 utterance → 0.5 Fear + 0.5 Contempt).
        self.soft_labels: list[SoftLabel] = (
            [tuple((str(n), float(w)) for n, w in sl) for sl in soft_labels]
            if soft_labels is not None
            else [((lab, 1.0),) for lab in self.labels]
        )
        self.k = min(k, max(1, pad.shape[0]))

    @classmethod
    def from_npz(cls, path: Path | str, *, k: int = 3) -> PADEmotionKNN:
        pad, soft = load_anchor_npz_soft(path)
        return cls(pad, [sl[0][0] for sl in soft], k=k, soft_labels=soft)

    def query(self, pad: Sequence[float]) -> KNNEmotionResult:
        """k nearest anchors → k label slots apportioned by (soft) vote share.

        Ties: MSP anchors still stack on shared coordinates (e.g. 1,090 at the
        origin), so when more than k anchors fall within the k-th distance, the
        whole tied group votes. Slots are split by largest remainder; ordering is
        by vote, then nearest distance, then name — deterministic.
        """
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
        kth = float(d[idx[-1]])
        tied = np.flatnonzero(d <= kth + _TIE_EPS)
        voters = tied if tied.size > k else idx
        weights: dict[str, float] = {}
        nearest: dict[str, float] = {}
        for i in voters:
            for name, w in self.soft_labels[i]:
                if w <= 0:
                    continue
                weights[name] = weights.get(name, 0.0) + w
                nearest[name] = min(nearest.get(name, np.inf), float(d[i]))
        labs = _apportion(weights, k, nearest)
        dists = tuple(float(d[i]) for i in idx)
        return KNNEmotionResult(
            query=(float(q[0]), float(q[1]), float(q[2])),
            neighbor_labels=labs,
            neighbor_distances=dists,
            primary_label=labs[0],
        )


_TIE_EPS = 1e-9


def _apportion(
    weights: dict[str, float], k: int, nearest: dict[str, float]
) -> tuple[str, ...]:
    """k label slots split by vote share (largest remainder); most votes first."""
    total = sum(weights.values())
    order = sorted(weights, key=lambda x: (-weights[x], nearest[x], x))
    quotas = {x: weights[x] * k / total for x in order}
    slots = {x: int(quotas[x] + 1e-9) for x in order}
    left = k - sum(slots.values())
    by_remainder = sorted(
        order, key=lambda x: (-(quotas[x] - slots[x]), -weights[x], nearest[x], x)
    )
    for x in by_remainder[:left]:
        slots[x] += 1
    return tuple(x for x in order for _ in range(slots[x]))
