"""Closed-set role emotion tags and CosyVoice instruct mapping."""

from agent.emotion.state import (
    DEFAULT_EMOTION,
    EMOTION_CHOICES,
    EMOTIONS,
    apply_emotion_from_parsed,
    emotion_output_lines,
    emotion_prompt_block,
    get_role_emotion,
    normalize_emotion,
    parse_emotion_tag,
    set_role_emotion,
    strip_emotion_tag,
)

__all__ = [
    "DEFAULT_EMOTION",
    "EMOTION_CHOICES",
    "EMOTIONS",
    "apply_emotion_from_parsed",
    "emotion_output_lines",
    "emotion_prompt_block",
    "get_role_emotion",
    "normalize_emotion",
    "parse_emotion_tag",
    "set_role_emotion",
    "strip_emotion_tag",
]
