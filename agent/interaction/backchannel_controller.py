"""Rule-based coordination for optional listener backchannels."""

from __future__ import annotations

import os
import re
import time
from collections import deque
from dataclasses import dataclass

from agent.interaction.think_action import Backchannel

DEFAULT_BACKCHANNEL_COOLDOWN_S = 12.0
DEFAULT_BACKCHANNEL_REPEAT_WINDOW_S = 30.0
_ACKNOWLEDGEMENTS = frozenset(
    {"right", "exactly", "yeah", "yes", "yep", "absolutely", "agreed", "mm hmm", "uh huh"}
)


def backchannel_cooldown_s() -> float:
    try:
        return max(
            0.0,
            float(os.environ.get("TALKSHOW_BACKCHANNEL_COOLDOWN_S", DEFAULT_BACKCHANNEL_COOLDOWN_S)),
        )
    except ValueError:
        return DEFAULT_BACKCHANNEL_COOLDOWN_S


def backchannel_repeat_window_s() -> float:
    try:
        return max(
            0.0,
            float(
                os.environ.get(
                    "TALKSHOW_BACKCHANNEL_REPEAT_WINDOW_S", DEFAULT_BACKCHANNEL_REPEAT_WINDOW_S
                )
            ),
        )
    except ValueError:
        return DEFAULT_BACKCHANNEL_REPEAT_WINDOW_S


def _content_key(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()


def _repeat_key(text: str) -> str:
    key = _content_key(text)
    return "acknowledgement" if key in _ACKNOWLEDGEMENTS else key


@dataclass(frozen=True)
class BackchannelCandidate:
    role: str
    event: Backchannel
    offset: int
    target_segment_index: int | None


class BackchannelController:
    """Select at most one candidate per speaker cue and rate-limit actual playback."""

    def __init__(self) -> None:
        self._reserved_cues: deque[int] = deque(maxlen=256)
        self._started: deque[tuple[float, int, str]] = deque(maxlen=16)

    def arbitrate(
        self,
        cue_id: int,
        candidates: list[BackchannelCandidate],
        *,
        now: float | None = None,
    ) -> tuple[BackchannelCandidate | None, list[tuple[BackchannelCandidate, str]]]:
        """Choose the earliest candidate; retain reasons for every rejection."""
        current = time.monotonic() if now is None else now
        if cue_id in self._reserved_cues:
            return None, [(candidate, "speaker_turn_limit") for candidate in candidates]

        self._expire(current)
        candidates.sort(key=lambda candidate: candidate.offset)
        recent_keys = {_repeat_key(text) for _, _, text in self._started}
        decisions: list[tuple[BackchannelCandidate, str]] = []
        selected: BackchannelCandidate | None = None
        for candidate in candidates:
            if _repeat_key(candidate.event.content) in recent_keys:
                reason = "repeated_feedback"
            elif selected is not None:
                reason = "one_candidate_per_speaker_turn"
            else:
                reason = "approved_pending_playback"
                selected = candidate
            decisions.append((candidate, reason))
        if selected is not None:
            self._reserved_cues.append(cue_id)
        return selected, decisions

    def try_start(
        self,
        cue_id: int,
        role: str,
        content: str,
        *,
        now: float | None = None,
    ) -> tuple[bool, str]:
        """Enforce cooldown and repeat suppression at actual playback start."""
        current = time.monotonic() if now is None else now
        self._expire(current)
        key = _repeat_key(content)
        if self._started:
            last_started = self._started[-1][0]
            if current - last_started < backchannel_cooldown_s():
                return False, "global_cooldown"
            if any(_repeat_key(previous) == key for _, _, previous in self._started):
                return False, "repeated_feedback"
        self._started.append((current, cue_id, content))
        return True, "played"

    def _expire(self, now: float) -> None:
        window = max(backchannel_cooldown_s(), backchannel_repeat_window_s())
        while self._started and now - self._started[0][0] > window:
            self._started.popleft()


__all__ = [
    "BackchannelCandidate",
    "BackchannelController",
    "backchannel_cooldown_s",
    "backchannel_repeat_window_s",
]