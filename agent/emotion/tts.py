"""Map talk-show [emotion] codes → CosyVoice instruct2 prompt strings."""

from __future__ import annotations

from agent.emotion.state import DEFAULT_EMOTION, normalize_emotion

# CosyVoice3 expects instruct text ending with <|endofprompt|>.
_END = "<|endofprompt|>"

# Closed-set → natural-language instruct (aligned with upstream examples:
# 「请非常开心地说一句话」 / soft / loud / speed).
_INSTRUCT_EN: dict[str, str] = {
    "neutral": "Speak in a calm, natural conversational tone.",
    "amused": "Speak with light amusement and a warm wry smile in your voice.",
    "curious": "Speak with curious, engaged interest.",
    "serious": "Speak in a serious, measured, thoughtful tone.",
    "skeptical": "Speak with mild skepticism and a questioning edge.",
    "warm": "Speak in a warm, friendly, welcoming tone.",
    "surprised": "Speak with mild surprise and lifted energy.",
}

_INSTRUCT_ZH: dict[str, str] = {
    "neutral": "请用平静自然的语气说这句话。",
    "amused": "请用略带俏皮、开心的语气说这句话。",
    "curious": "请用好奇、感兴趣的语气说这句话。",
    "serious": "请用认真、沉稳的语气说这句话。",
    "skeptical": "请用略带怀疑、反问感的语气说这句话。",
    "warm": "请用温暖、亲切的语气说这句话。",
    "surprised": "请用略带惊讶的语气说这句话。",
}


def emotion_instruct(*, emotion: str | None, locale: str = "en") -> str:
    """Build CosyVoice ``inference_instruct2`` instruct string for a mood code."""
    code = normalize_emotion(emotion) or DEFAULT_EMOTION
    loc = (locale or "en").strip().lower()
    if loc.startswith("zh"):
        body = _INSTRUCT_ZH.get(code, _INSTRUCT_ZH[DEFAULT_EMOTION])
        return f"You are a helpful assistant. {body}{_END}"
    body = _INSTRUCT_EN.get(code, _INSTRUCT_EN[DEFAULT_EMOTION])
    return f"You are a helpful assistant. {body}{_END}"
