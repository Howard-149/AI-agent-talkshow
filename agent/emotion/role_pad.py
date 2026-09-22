"""Per-role PAD helpers on TalkShowData (parallel to categorical role_emotion)."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Sequence

from agent.emotion.pad_state import (
    PADState,
    decay_pad,
    decay_then_update,
)

if TYPE_CHECKING:
    from agent.data import TalkShowData

logger = logging.getLogger(__name__)

# Default half-life: Sentipolis 120 minutes, expressed in seconds.
DEFAULT_HALF_LIFE_S = 120.0 * 60.0


def _state_from_data(data: TalkShowData, role: str) -> PADState:
    role = (role or "").strip().lower()
    raw = getattr(data, "role_pad", {}).get(role)
    if raw is None:
        return PADState()
    if isinstance(raw, PADState):
        return raw.copy()
    if len(raw) >= 3:
        return PADState(
            pleasure=float(raw[0]),
            arousal=float(raw[1]),
            dominance=float(raw[2]),
        )
    return PADState()


def _store(data: TalkShowData, role: str, state: PADState) -> None:
    role = (role or "").strip().lower()
    if not hasattr(data, "role_pad") or data.role_pad is None:
        data.role_pad = {}
    data.role_pad[role] = state.as_tuple()


def get_role_pad(data: TalkShowData, role: str) -> tuple[float, float, float]:
    return _state_from_data(data, role).as_tuple()


def set_role_pad(
    data: TalkShowData, role: str, pad: Sequence[float]
) -> tuple[float, float, float]:
    if len(pad) != 3:
        raise ValueError("pad must be length 3")
    state = PADState(
        pleasure=float(pad[0]),
        arousal=float(pad[1]),
        dominance=float(pad[2]),
    )
    from agent.emotion.pad_state import clamp_state

    clamp_state(state)
    _store(data, role, state)
    return state.as_tuple()


def update_role_pad(
    data: TalkShowData,
    role: str,
    delta: Sequence[float],
    *,
    delta_t_s: float = 0.0,
    half_life_s: float = DEFAULT_HALF_LIFE_S,
    scale: float = 1.0,
) -> tuple[float, float, float]:
    """Decay by elapsed seconds (optional), then apply appraisal delta."""
    role = (role or "").strip().lower()
    if not role or role == "human":
        return (0.0, 0.0, 0.0)
    state = _state_from_data(data, role)
    decay_then_update(
        state,
        delta,
        delta_t=delta_t_s,
        half_life=half_life_s,
        scale=scale,
    )
    _store(data, role, state)
    logger.info(
        "role_pad role=%s → P=%.3f A=%.3f D=%.3f (Δt=%.1fs)",
        role,
        state.pleasure,
        state.arousal,
        state.dominance,
        delta_t_s,
    )
    return state.as_tuple()


def decay_role_pad(
    data: TalkShowData,
    role: str,
    *,
    delta_t_s: float,
    half_life_s: float = DEFAULT_HALF_LIFE_S,
) -> tuple[float, float, float]:
    state = _state_from_data(data, role)
    decay_pad(state, delta_t=delta_t_s, half_life=half_life_s)
    _store(data, role, state)
    return state.as_tuple()
