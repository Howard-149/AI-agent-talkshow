"""Continuous PAD emotion state with decay and delta updates (Sentipolis-style).

Pleasure–Arousal–Dominance each in [-1, 1]. Decay uses half-life:

    s(t+Δt) = s(t) * 2^(-Δt / T_half)

Paper default half-life is 120 minutes; talk-show turns are shorter — callers
pass Δt in seconds (or any unit consistent with half_life).
"""

from __future__ import annotations

from dataclasses import dataclass, field
import time
from typing import Iterable, Sequence


def _clip_pad(x: float) -> float:
    return max(-1.0, min(1.0, float(x)))


@dataclass
class PADState:
    pleasure: float = 0.0
    arousal: float = 0.0
    dominance: float = 0.0
    updated_at: float = field(default_factory=time.monotonic)

    def as_tuple(self) -> tuple[float, float, float]:
        return (self.pleasure, self.arousal, self.dominance)

    def as_list(self) -> list[float]:
        return [self.pleasure, self.arousal, self.dominance]

    def copy(self) -> PADState:
        return PADState(
            pleasure=self.pleasure,
            arousal=self.arousal,
            dominance=self.dominance,
            updated_at=self.updated_at,
        )


def clamp_state(state: PADState) -> PADState:
    state.pleasure = _clip_pad(state.pleasure)
    state.arousal = _clip_pad(state.arousal)
    state.dominance = _clip_pad(state.dominance)
    return state


def decay_pad(
    state: PADState,
    *,
    delta_t: float,
    half_life: float = 120.0 * 60.0,
    now: float | None = None,
) -> PADState:
    """
    Exponential decay toward 0 (neutrality).

    Parameters
    ----------
    delta_t : float
        Elapsed time in the same units as ``half_life`` (default: seconds;
        paper uses minutes with half_life=120).
    half_life : float
        Sentipolis default 120 minutes → pass ``120*60`` if using seconds.
    """
    if half_life <= 0:
        raise ValueError("half_life must be > 0")
    if delta_t < 0:
        delta_t = 0.0
    factor = 2.0 ** (-float(delta_t) / float(half_life))
    state.pleasure = _clip_pad(state.pleasure * factor)
    state.arousal = _clip_pad(state.arousal * factor)
    state.dominance = _clip_pad(state.dominance * factor)
    state.updated_at = time.monotonic() if now is None else now
    return state


def apply_pad_delta(
    state: PADState,
    delta: Sequence[float],
    *,
    scale: float = 1.0,
    now: float | None = None,
) -> PADState:
    """Add appraisal delta [dP, dA, dD] (optionally scaled) and clamp."""
    if len(delta) != 3:
        raise ValueError(f"delta must have length 3, got {len(delta)}")
    state.pleasure = _clip_pad(state.pleasure + scale * float(delta[0]))
    state.arousal = _clip_pad(state.arousal + scale * float(delta[1]))
    state.dominance = _clip_pad(state.dominance + scale * float(delta[2]))
    state.updated_at = time.monotonic() if now is None else now
    return state


def decay_then_update(
    state: PADState,
    delta: Sequence[float],
    *,
    delta_t: float,
    half_life: float = 120.0 * 60.0,
    scale: float = 1.0,
) -> PADState:
    """Apply time decay, then appraisal delta (fast-update path)."""
    decay_pad(state, delta_t=delta_t, half_life=half_life)
    apply_pad_delta(state, delta, scale=scale)
    return state


def blend_toward(
    state: PADState,
    target: Sequence[float],
    *,
    alpha: float,
) -> PADState:
    """Move state toward target: s ← (1-α)s + α target (α in [0,1])."""
    a = max(0.0, min(1.0, float(alpha)))
    if len(target) != 3:
        raise ValueError("target must have length 3")
    state.pleasure = _clip_pad((1 - a) * state.pleasure + a * float(target[0]))
    state.arousal = _clip_pad((1 - a) * state.arousal + a * float(target[1]))
    state.dominance = _clip_pad((1 - a) * state.dominance + a * float(target[2]))
    state.updated_at = time.monotonic()
    return state


def mean_pad(states: Iterable[PADState]) -> tuple[float, float, float]:
    xs = list(states)
    if not xs:
        return (0.0, 0.0, 0.0)
    n = float(len(xs))
    return (
        sum(s.pleasure for s in xs) / n,
        sum(s.arousal for s in xs) / n,
        sum(s.dominance for s in xs) / n,
    )
