"""Statistics for the ruler: Krippendorff's alpha, Bradley–Terry with bootstrap, length-controlled logistic fit."""

from __future__ import annotations

import math
import random
from collections import defaultdict

import numpy as np


def krippendorff_alpha(units: dict, level: str = "nominal") -> float:
    """units: unit_id → list of values (one per coder; None = missing). nominal or interval."""
    vals = [[v for v in vs if v is not None] for vs in units.values()]
    vals = [v for v in vals if len(v) >= 2]
    if not vals:
        return float("nan")
    allv = [x for v in vals for x in v]
    n = len(allv)

    def d(a, b):
        return float(a != b) if level == "nominal" else (a - b) ** 2

    do = sum(sum(d(a, b) for i, a in enumerate(v) for j, b in enumerate(v) if i != j) / (len(v) - 1) for v in vals) / n
    if level == "nominal":
        cnt = defaultdict(int)
        for x in allv:
            cnt[x] += 1
        de = sum(cnt[a] * cnt[b] * d(a, b) for a in cnt for b in cnt) / (n * (n - 1))
    else:
        arr = np.array(allv, dtype=float)
        de = float(((arr[:, None] - arr[None, :]) ** 2).sum() / (n * (n - 1)))
    return 1 - do / de if de > 0 else float("nan")


def bradley_terry(wins: list[tuple[str, str, float]], iters: int = 200) -> dict[str, float]:
    """wins: (a, b, s) with s = share of a's wins over b in [0, 1]. Returns log-strengths (mean 0), MM algorithm."""
    items = sorted({x for a, b, _ in wins for x in (a, b)})
    p = {i: 1.0 for i in items}
    w = defaultdict(float)
    n = defaultdict(float)
    for a, b, s in wins:
        w[a] += s
        w[b] += 1 - s
        n[(a, b)] += 1
        n[(b, a)] += 1
    for _ in range(iters):
        new = {}
        for i in items:
            den = sum(n[(i, j)] / (p[i] + p[j]) for j in items if n[(i, j)])
            new[i] = (w[i] + 0.1) / den if den else p[i]  # +0.1 pseudo-count keeps all-win items finite
        g = math.exp(np.mean([math.log(v) for v in new.values()]))
        p = {i: v / g for i, v in new.items()}
    return {i: math.log(v) for i, v in p.items()}


def bt_bootstrap(rows: list[dict], key_a: str, key_b: str, cluster: str, boot: int = 1000, seed: int = 0):
    """rows have source labels under key_a/key_b and a pairwise share 's' (a beats b). Bootstrap over clusters."""
    clusters = sorted({r[cluster] for r in rows})
    by = defaultdict(list)
    for r in rows:
        by[r[cluster]].append(r)
    point = bradley_terry([(r[key_a], r[key_b], r["s"]) for r in rows])
    rng = random.Random(seed)
    samples = defaultdict(list)
    for _ in range(boot):
        rs = [r for c in (rng.choice(clusters) for _ in clusters) for r in by[c]]
        for k, v in bradley_terry([(r[key_a], r[key_b], r["s"]) for r in rs]).items():
            samples[k].append(v)
    ci = {k: (float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))) for k, v in samples.items()}
    return point, ci


def logistic_length(y: list[float], dlen: list[float]) -> tuple[float, float]:
    """Fit P(a wins) = σ(β0 + β1·Δlength) by IRLS on soft labels; returns (β0 = length-controlled preference, β1)."""
    X = np.column_stack([np.ones(len(y)), np.array(dlen, dtype=float)])
    t = np.array(y, dtype=float)
    beta = np.zeros(2)
    for _ in range(50):
        p = 1 / (1 + np.exp(-X @ beta))
        W = np.clip(p * (1 - p), 1e-6, None)
        H = X.T @ (X * W[:, None]) + 0.1 * np.eye(2)  # small ridge: stays finite on separable data
        beta = beta + np.linalg.solve(H, X.T @ (t - p) - 0.1 * beta)
    return float(beta[0]), float(beta[1])


def pearson(a: list[float], b: list[float]) -> float:
    a, b = np.array(a, dtype=float), np.array(b, dtype=float)
    ok = ~(np.isnan(a) | np.isnan(b))
    if ok.sum() < 3 or a[ok].std() == 0 or b[ok].std() == 0:
        return float("nan")
    return float(np.corrcoef(a[ok], b[ok])[0, 1])
