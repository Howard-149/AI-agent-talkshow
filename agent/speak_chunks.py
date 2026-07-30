"""Sentence chunks for pseudo-streaming avatar bake — match LiveKit TTS StreamAdapter splits."""
from __future__ import annotations

import logging
import os
import re

logger = logging.getLogger(__name__)


def avatar_chunk_stream_enabled() -> bool:
    """Split panel lines into sentence chunks and pipeline bake/play (default on)."""
    raw = os.environ.get("TALKSHOW_AVATAR_CHUNK_STREAM", "1").strip().lower()
    return raw not in ("0", "false", "no", "off")


def split_speak_sentences(text: str) -> list[str]:
    """
    Same sentence boundaries as LiveKit `tokenize.basic.SentenceTokenizer`
    (used by TTS StreamAdapter), so avatar chunks align with TTS playout chunks.
    """
    text = text.strip()
    if not text:
        return []

    min_len = int(os.environ.get("TALKSHOW_AVATAR_CHUNK_MIN_SENTENCE_LEN", "20"))
    try:
        from livekit.agents.tokenize import basic

        parts = [
            p.strip()
            for p in basic.SentenceTokenizer(min_sentence_len=min_len).tokenize(text)
            if p.strip()
        ]
        if parts:
            return parts
    except Exception as exc:
        logger.warning("LiveKit SentenceTokenizer unavailable (%s); using regex split", exc)

    parts = [p.strip() for p in re.split(r"(?<=[.!?])\s+", text) if p.strip()]
    return parts or [text]
