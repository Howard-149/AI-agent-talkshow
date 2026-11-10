"""Plain-text log for model responses, separate from session telemetry."""

from __future__ import annotations

import time
import json
from datetime import datetime
from pathlib import Path

from agent.config import load_persona_name


class ModelOutputLogger:
    def __init__(self, log_dir: Path) -> None:
        log_dir.mkdir(parents=True, exist_ok=True)
        self._path = log_dir / f"session-{time.time_ns()}.log"
        self._names: dict[str, str | None] = {}

    @property
    def path(self) -> Path:
        return self._path

    def _speaker_name(self, role: str) -> str | None:
        """Display name of a speaker; None for an event category (floor, rollback, …)."""
        if role not in self._names:
            if role in {"floor", "topic", "interval"}:
                name = None
            elif role == "human":
                name = "Human guest"
            else:
                try:
                    name = load_persona_name(role)
                except FileNotFoundError:  # rollback, selection, speech: not a persona
                    name = None
            self._names[role] = name
        return self._names[role]

    def event(self, role: str, kind: str, **fields: object) -> None:
        if kind == "topic":
            topic = fields.get("topic", "")
            lines = [f"topic: {json.dumps(topic, ensure_ascii=False)}"]
        else:
            lines = [f"[{kind}]"]
        # Wall-clock time of the entry; a field, so header lines stay parseable.
        stamp = datetime.now().isoformat(sep=" ", timespec="milliseconds")
        lines.append(f"time: {json.dumps(stamp)}")
        name = self._speaker_name(role) if kind != "topic" else None
        if name is not None:
            lines.append(f"speaker: {json.dumps(name, ensure_ascii=False)}")
            lines.append(f"role: {json.dumps(role, ensure_ascii=False)}")
        if kind != "topic":
            lines.extend(
                f"{key}: {json.dumps(value, ensure_ascii=False)}"
                for key, value in fields.items()
            )
        with self._path.open("a", encoding="utf-8") as output:
            separator = "\n\n\n" if kind == "selection" else "\n\n"
            output.write("\n".join(lines) + separator)