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


@dataclass
class ShowHistory:
    lines: list[HistoryLine] = field(default_factory=list)

    def append(self, role_id: str, text: str, *, speaker: str | None = None) -> None:
        text = text.strip()
        if not text:
            return
        label = speaker or _speaker_label(role_id)
        self.lines.append(HistoryLine(role_id=role_id, speaker=label, text=text))
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

    def prior_messages(self) -> list[dict[str, Any]]:
        """All lines so far — fed to Gemma before the current user turn."""
        from agent.panel_context import role_label

        msgs: list[dict[str, Any]] = []
        for line in self.lines:
            if line.role_id == "human":
                msgs.append(
                    {
                        "role": "user",
                        "content": f"{HUMAN_LABEL}: {line.text}",
                    }
                )
            else:
                if line.role_id == "host":
                    tag = f"{line.speaker} (host)"
                else:
                    tag = f"{line.speaker} ({role_label(line.role_id)})"
                msgs.append(
                    {
                        "role": "assistant",
                        "content": f"{tag}: {line.text}",
                    }
                )
        return msgs


def _speaker_label(role_id: str) -> str:
    if role_id == "human":
        return HUMAN_LABEL
    return load_persona_name(role_id)


def append_human(data: object, text: str) -> None:
    from agent.data import TalkShowData

    if isinstance(data, TalkShowData):
        data.show_history.append("human", text)


def append_role(data: object, role_id: str, text: str) -> None:
    from agent.data import TalkShowData

    if isinstance(data, TalkShowData):
        data.show_history.append(role_id, text)
