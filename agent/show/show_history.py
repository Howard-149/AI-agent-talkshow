"""Shared show transcript (Human guest ≠ Amy) for all panelist prompts."""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from typing import Any

from agent.config import load_persona_name

logger = logging.getLogger(__name__)

HUMAN_LABEL = "Human guest"


@dataclass
class HistoryLine:
    role_id: str  # human | host | guest | commentator
    speaker: str
    text: str
    # Canned procedural host line ("Floor's open.", "Amy, you're up.") — kept in the
    # transcript but skipped when working out who a line is replying to.
    procedural: bool = False


@dataclass
class ShowHistory:
    lines: list[HistoryLine] = field(default_factory=list)

    def append(
        self,
        role_id: str,
        text: str,
        *,
        speaker: str | None = None,
        procedural: bool = False,
    ) -> None:
        text = text.strip()
        if not text:
            return
        label = speaker or _speaker_label(role_id)
        self.lines.append(
            HistoryLine(role_id=role_id, speaker=label, text=text, procedural=procedural)
        )
        self._trim()
        logger.debug("history +%s (%d lines)", label, len(self.lines))

    def _trim(self) -> None:
        max_lines = int(os.environ.get("TALKSHOW_HISTORY_MAX_LINES", "48"))
        if len(self.lines) > max_lines:
            self.lines = self.lines[-max_lines:]

    def latest_human_text(self) -> str:
        for line in reversed(self.lines):
            if line.role_id == "human":
                return line.text
        return ""

    def role_has_spoken(self, role_id: str) -> bool:
        return any(line.role_id == role_id for line in self.lines)

    def prior_messages(self, perspective: str | None = None) -> list[dict[str, Any]]:
        """All lines so far as chat messages, fed to Gemma before the current turn.

        ``perspective`` = the role about to speak. Its own earlier lines become
        unlabelled ``assistant`` turns; everyone else's lines become labelled
        ``user`` turns ("Ryan (Commentator): …"). Consecutive same-role messages
        are merged so turns alternate. Without a perspective (or with
        ``TALKSHOW_HISTORY_FORMAT=legacy``) every AI line is a labelled
        ``assistant`` turn — which led models to answer *as* another panelist
        ("Amy (Guest): …") and to copy the label prefix.
        """
        legacy = os.environ.get("TALKSHOW_HISTORY_FORMAT", "").strip().lower() == "legacy"
        if perspective is None or legacy:
            return self._legacy_messages()

        msgs: list[dict[str, Any]] = []
        for line in self.lines:
            if line.role_id == perspective:
                role, content = "assistant", line.text
            else:
                role, content = "user", f"{_line_label(line)}: {line.text}"
            if msgs and msgs[-1]["role"] == role:
                msgs[-1]["content"] += "\n" + content
            else:
                msgs.append({"role": role, "content": content})
        return msgs

    def _legacy_messages(self) -> list[dict[str, Any]]:
        msgs: list[dict[str, Any]] = []
        for line in self.lines:
            if line.role_id == "human":
                msgs.append({"role": "user", "content": f"{HUMAN_LABEL}: {line.text}"})
            else:
                msgs.append({"role": "assistant", "content": f"{_line_label(line)}: {line.text}"})
        return msgs


def _line_label(line: HistoryLine) -> str:
    """Transcript label: "Human guest", "Lessac (host)", "Ryan (Commentator)"."""
    if line.role_id == "human":
        return HUMAN_LABEL
    if line.role_id == "host":
        return f"{line.speaker} (host)"
    from agent.panel.panel_context import role_label

    return f"{line.speaker} ({role_label(line.role_id)})"


def _speaker_label(role_id: str) -> str:
    if role_id == "human":
        return HUMAN_LABEL
    return load_persona_name(role_id)


def _schedule_appraisal(data: object, speaker: str, text: str) -> None:
    """Every spoken line passes here — speaker relaxes, listeners appraise (PAD)."""
    from agent.emotion.appraisal import on_utterance

    try:
        on_utterance(data, speaker, text)  # type: ignore[arg-type]
    except Exception:
        logger.exception("pad appraisal scheduling failed speaker=%s", speaker)


def append_human(data: object, text: str, *, appraise: bool = True) -> None:
    from agent.data import TalkShowData

    if isinstance(data, TalkShowData):
        data.show_history.append("human", text)
        if appraise:
            _schedule_appraisal(data, "human", text)


def append_role(
    data: object,
    role_id: str,
    text: str,
    *,
    appraise: bool = True,
    procedural: bool | None = None,
) -> None:
    """Record a spoken line.

    ``appraise`` runs the legacy background appraisal hook (non-graph turn modes);
    the show graph appraises lines itself and passes ``appraise=False``.
    ``procedural`` marks canned host lines ("Floor's open.") — defaults to
    ``not appraise`` for legacy callers.
    """
    from agent.data import TalkShowData

    if isinstance(data, TalkShowData):
        is_procedural = (not appraise) if procedural is None else procedural
        data.show_history.append(role_id, text, procedural=is_procedural)
        if appraise:
            _schedule_appraisal(data, role_id, text)
