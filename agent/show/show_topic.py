"""The configured discussion topic."""

from __future__ import annotations

import os


def show_topic(default: str = "") -> str:
    """The discussion topic: ``TALKSHOW_TOPIC`` (``TALKSHOW_TEXT_TOPIC`` still honored)."""
    for name in ("TALKSHOW_TOPIC", "TALKSHOW_TEXT_TOPIC"):
        value = os.environ.get(name, "").strip()
        if value:
            return value
    return default.strip()
