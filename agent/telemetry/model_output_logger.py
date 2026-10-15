"""Plain-text log for model responses, separate from session telemetry."""

from __future__ import annotations

import time
from pathlib import Path

from agent.config import load_persona_name


class ModelOutputLogger:
    def __init__(self, log_dir: Path) -> None:
        log_dir.mkdir(parents=True, exist_ok=True)
        self._path = log_dir / f"session-{time.time_ns()}.log"
        self._names: dict[str, str] = {}

    @property
    def path(self) -> Path:
        return self._path

    def log(self, role: str, content: str) -> None:
        text = content.strip()
        if not text:
            return
        if role not in self._names:
            self._names[role] = load_persona_name(role)
        name = self._names[role]
        with self._path.open("a", encoding="utf-8") as output:
            output.write(f"{name}: {text}\n\n")