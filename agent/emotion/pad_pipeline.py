"""PAD → kNN MSP label → role_emotion materialization pipeline."""

from __future__ import annotations

import logging
import os
import time
from pathlib import Path
from typing import TYPE_CHECKING, Sequence

from agent.emotion.knn_map import PADEmotionKNN, modal_labels
from agent.emotion.role_pad import (
    decay_role_pad,
    get_role_pad,
    update_role_pad,
)
from agent.emotion.state import (
    DEFAULT_EMOTION,
    emotion_source,
    get_role_emotion,
    set_role_emotion,
)

if TYPE_CHECKING:
    from agent.data import TalkShowData

logger = logging.getLogger(__name__)

# Talk-show turn scale (seconds). Paper uses 120 minutes — too slow for audible shifts.
DEFAULT_TALKSHOW_HALF_LIFE_S = 120.0
# Optional global gain on appraisal deltas (1.0 = use the appraiser's values as-is).
DEFAULT_TALKSHOW_DELTA_SCALE = 1.0

_knn: PADEmotionKNN | None = None
_knn_failed = False


def talkshow_half_life_s() -> float:
    raw = os.environ.get("TALKSHOW_PAD_HALF_LIFE_S", "").strip()
    if raw:
        try:
            v = float(raw)
            if v > 0:
                return v
        except ValueError:
            pass
    return DEFAULT_TALKSHOW_HALF_LIFE_S


def talkshow_delta_scale() -> float:
    raw = os.environ.get("TALKSHOW_PAD_DELTA_SCALE", "").strip()
    if raw:
        try:
            v = float(raw)
            if v > 0:
                return v
        except ValueError:
            pass
    return DEFAULT_TALKSHOW_DELTA_SCALE


def talkshow_soft_bound() -> bool:
    """Diminishing-returns PAD update (default on; ``TALKSHOW_PAD_SOFT_BOUND=0`` = paper's plain clip)."""
    raw = os.environ.get("TALKSHOW_PAD_SOFT_BOUND", "").strip().lower()
    return raw not in ("0", "false", "no", "off")


def default_anchors_path() -> Path:
    env = os.environ.get("TALKSHOW_MSP_ANCHORS", "").strip()
    if env:
        return Path(env)
    return Path(__file__).resolve().parent / "data" / "msp_pad_anchors.npz"


def get_pad_knn(*, force_reload: bool = False) -> PADEmotionKNN | None:
    """Lazy-load MSP anchors; None if missing (materialize falls back to Neutral)."""
    global _knn, _knn_failed
    if force_reload:
        _knn = None
        _knn_failed = False
    if _knn is not None:
        return _knn
    if _knn_failed:
        return None
    path = default_anchors_path()
    if not path.is_file():
        logger.warning("MSP anchors missing at %s — PAD materialize → Neutral", path)
        _knn_failed = True
        return None
    try:
        _knn = PADEmotionKNN.from_npz(path, k=3)
        logger.info("loaded PADEmotionKNN anchors=%s n=%d", path, _knn.pad.shape[0])
        return _knn
    except Exception:
        logger.exception("failed to load MSP anchors from %s", path)
        _knn_failed = True
        return None


def _elapsed_s(data: TalkShowData, role: str) -> float:
    role = (role or "").strip().lower()
    ts_map = getattr(data, "role_pad_ts", None)
    if not isinstance(ts_map, dict):
        return 0.0
    prev = ts_map.get(role)
    if prev is None:
        return 0.0
    return max(0.0, time.monotonic() - float(prev))


def _touch_ts(data: TalkShowData, role: str) -> None:
    role = (role or "").strip().lower()
    if not hasattr(data, "role_pad_ts") or data.role_pad_ts is None:  # type: ignore[attr-defined]
        data.role_pad_ts = {}  # type: ignore[attr-defined]
    data.role_pad_ts[role] = time.monotonic()  # type: ignore[attr-defined]


VAGUE_LABEL = "Vague"


def speakable_label(neighbors: Sequence[str]) -> str:
    """Primary label for TTS / UI: first non-Vague neighbor, else Neutral.

    MSP "Vague" (annotators disagreed / Other) is kept in kNN and logs, but it is
    not a voice or a mood a speaker can act on.
    """
    for lab in neighbors:
        if lab and lab != VAGUE_LABEL:
            return lab
    return DEFAULT_EMOTION


def _store_neighbors(data: TalkShowData, role: str, labels: Sequence[str]) -> None:
    """kNN neighbor labels feed the prompt mood descriptor (Sentipolis: keep all k)."""
    if not hasattr(data, "role_pad_neighbors") or data.role_pad_neighbors is None:  # type: ignore[attr-defined]
        data.role_pad_neighbors = {}  # type: ignore[attr-defined]
    data.role_pad_neighbors[role] = [str(x) for x in labels]  # type: ignore[attr-defined]


def _round3(pad: Sequence[float]) -> list[float]:
    return [round(float(pad[0]), 4), round(float(pad[1]), 4), round(float(pad[2]), 4)]


def _log_pad_map(
    data: TalkShowData,
    role: str,
    *,
    pad_before: Sequence[float],
    pad_after: Sequence[float],
    delta: Sequence[float] | None,
    dt_s: float,
    half_life_s: float,
    emotion: str,
    emotion_was: str,
    neighbors: Sequence[str],
    neighbor_distances: Sequence[float],
    reason: str,
) -> None:
    turn_log = getattr(data, "turn_log", None)
    if turn_log is None or not hasattr(turn_log, "log"):
        return
    turn_log.log(
        "pad_map",
        role=role,
        pad_before=_round3(pad_before),
        pad_after=_round3(pad_after),
        delta=_round3(delta) if delta is not None and len(delta) == 3 else None,
        dt_s=round(float(dt_s), 3),
        half_life_s=round(float(half_life_s), 1),
        emotion=emotion,
        emotion_was=emotion_was,
        neighbors=list(neighbors),
        labels=list(modal_labels(neighbors)) if neighbors else [emotion],
        neighbor_distances=[round(float(x), 4) for x in neighbor_distances],
        source="pad",
        reason=reason,
        room=getattr(data, "room_name", "") or "",
    )


def materialize_emotion_from_pad(
    data: TalkShowData,
    role: str,
    *,
    decay: bool = True,
    pad_before: Sequence[float] | None = None,
    delta: Sequence[float] | None = None,
    dt_s: float | None = None,
    reason: str | None = None,
) -> str:
    """
    Decay PAD (optional), kNN → MSP primary label, write ``role_emotion``.

    Safe no-op path when anchors are missing: sets Neutral.
    Logs ``pad_map`` on ``data.turn_log`` when present.
    """
    role = (role or "").strip().lower()
    if not role or role == "human":
        return DEFAULT_EMOTION

    half = talkshow_half_life_s()
    before = tuple(pad_before) if pad_before is not None else get_role_pad(data, role)
    emotion_was = get_role_emotion(data, role)
    dt = float(dt_s) if dt_s is not None else (_elapsed_s(data, role) if decay else 0.0)
    if decay and dt > 0:
        decay_role_pad(data, role, delta_t_s=dt, half_life_s=half)
    _touch_ts(data, role)

    pad = get_role_pad(data, role)
    knn = get_pad_knn()
    neighbors: tuple[str, ...] = ()
    neighbor_distances: tuple[float, ...] = ()
    if knn is None:
        label = set_role_emotion(data, role, DEFAULT_EMOTION)
        _store_neighbors(data, role, (label,))
    else:
        result = knn.query(pad)
        label = speakable_label(result.neighbor_labels)
        neighbors = result.neighbor_labels
        neighbor_distances = result.neighbor_distances
        logger.info(
            "pad_materialize role=%s pad=(%.3f,%.3f,%.3f) → %s neighbors=%s",
            role,
            pad[0],
            pad[1],
            pad[2],
            label,
            neighbors,
        )
        label = set_role_emotion(data, role, label)
        _store_neighbors(data, role, neighbors or (label,))

    why = reason or ("decay" if decay else "materialize")
    _log_pad_map(
        data,
        role,
        pad_before=before,
        pad_after=pad,
        delta=delta,
        dt_s=dt,
        half_life_s=half,
        emotion=label,
        emotion_was=emotion_was,
        neighbors=neighbors,
        neighbor_distances=neighbor_distances,
        reason=why,
    )
    return label


def apply_pad_delta_from_parsed(
    data: TalkShowData,
    role: str,
    delta: Sequence[float] | None,
    *,
    reason: str = "delta",
) -> str:
    """Apply optional PAD delta (with decay), then materialize MSP emotion tag."""
    role = (role or "").strip().lower()
    if not role or role == "human":
        return DEFAULT_EMOTION

    half = talkshow_half_life_s()
    dt = _elapsed_s(data, role)
    if delta is not None and len(delta) == 3:
        pad_before = get_role_pad(data, role)
        update_role_pad(
            data,
            role,
            delta,
            delta_t_s=dt,
            half_life_s=half,
            scale=talkshow_delta_scale(),
            soft_bound=talkshow_soft_bound(),
        )
        _touch_ts(data, role)
        # Skip second decay inside materialize — already decayed+updated.
        return materialize_emotion_from_pad(
            data,
            role,
            decay=False,
            pad_before=pad_before,
            delta=delta,
            dt_s=dt,
            reason=reason,
        )

    return materialize_emotion_from_pad(data, role, decay=True, reason="decay")


def apply_mood_from_parsed(
    data: TalkShowData,
    role: str,
    *,
    emotion: str | None = None,
) -> str:
    """Dispatch pad vs llm mood update after a model turn."""
    if emotion_source() == "llm":
        from agent.emotion.state import apply_emotion_from_parsed

        return apply_emotion_from_parsed(data, role, emotion)
    # PAD itself is updated by listeners' separate appraisal (agent/emotion/appraisal.py).
    return materialize_emotion_from_pad(data, role, decay=True, reason="decay")
