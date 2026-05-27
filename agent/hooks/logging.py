from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any


class TurnJsonlLogger:
    def __init__(self, log_dir: Path) -> None:
        self._path = log_dir / f"session-{int(time.time())}.jsonl"
        log_dir.mkdir(parents=True, exist_ok=True)

    def log(self, event: str, **fields: Any) -> None:
        row = {"ts": time.time(), "event": event, **fields}
        with self._path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
