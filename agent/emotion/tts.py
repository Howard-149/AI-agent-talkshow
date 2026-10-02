"""Map emotion labels → CosyVoice instruct2 prompt strings.

Open vocabulary: MSP kNN labels (Anger, Happiness, Concerned, …) are the live
tags. Legacy closed-set codes still have dedicated phrasing for llm rollback.
"""

from __future__ import annotations

from agent.emotion.state import DEFAULT_EMOTION, LEGACY_DEFAULT_EMOTION, normalize_emotion

# CosyVoice3 expects instruct text ending with <|endofprompt|>.
_END = "<|endofprompt|>"

# Legacy closed-set (llm rollback).
_LEGACY_INSTRUCT_EN: dict[str, str] = {
    "neutral": "Speak in a calm, natural conversational tone.",
    "amused": "Speak with light amusement and a warm wry smile in your voice.",
    "curious": "Speak with curious, engaged interest.",
    "serious": "Speak in a serious, measured, thoughtful tone.",
    "skeptical": "Speak with mild skepticism and a questioning edge.",
    "warm": "Speak in a warm, friendly, welcoming tone.",
    "surprised": "Speak with mild surprise and lifted energy.",
}

_LEGACY_INSTRUCT_ZH: dict[str, str] = {
    "neutral": "请用平静自然的语气说这句话。",
    "amused": "请用略带俏皮、开心的语气说这句话。",
    "curious": "请用好奇、感兴趣的语气说这句话。",
    "serious": "请用认真、沉稳的语气说这句话。",
    "skeptical": "请用略带怀疑、反问感的语气说这句话。",
    "warm": "请用温暖、亲切的语气说这句话。",
    "surprised": "请用略带惊讶的语气说这句话。",
}

# Primary MSP classes — richer phrasing than the generic template.
_MSP_INSTRUCT_EN: dict[str, str] = {
    "Neutral": "Speak in a calm, natural conversational tone.",
    "Happiness": "Speak with warmth and light happiness in your voice.",
    "Anger": "Speak with controlled anger and firm intensity.",
    "Sadness": "Speak with a subdued, sad, reflective tone.",
    "Surprise": "Speak with mild surprise and lifted energy.",
    "Fear": "Speak with uneasy tension and cautious fear.",
    "Disgust": "Speak with distaste and mild disgust.",
    "Contempt": "Speak with cool contempt and a dismissive edge.",
    "Vague": "Speak in a calm, natural conversational tone.",
}

_MSP_INSTRUCT_ZH: dict[str, str] = {
    "Neutral": "请用平静自然的语气说这句话。",
    "Happiness": "请用温暖、带有喜悦的语气说这句话。",
    "Anger": "请用克制但坚定、带怒意的语气说这句话。",
    "Sadness": "请用低沉、略带悲伤的语气说这句话。",
    "Surprise": "请用略带惊讶的语气说这句话。",
    "Fear": "请用紧张、不安的语气说这句话。",
    "Disgust": "请用略带厌恶的语气说这句话。",
    "Contempt": "请用冷淡、轻蔑的语气说这句话。",
    "Vague": "请用平静自然的语气说这句话。",
}


def emotion_instruct(*, emotion: str | None, locale: str = "en") -> str:
    """Build CosyVoice ``inference_instruct2`` instruct string for a mood label."""
    code = normalize_emotion(emotion) or DEFAULT_EMOTION
    loc = (locale or "en").strip().lower()
    zh = loc.startswith("zh")

    if code in _LEGACY_INSTRUCT_EN:
        body = (_LEGACY_INSTRUCT_ZH if zh else _LEGACY_INSTRUCT_EN).get(
            code,
            (_LEGACY_INSTRUCT_ZH if zh else _LEGACY_INSTRUCT_EN)[LEGACY_DEFAULT_EMOTION],
        )
    elif code in _MSP_INSTRUCT_EN:
        body = (_MSP_INSTRUCT_ZH if zh else _MSP_INSTRUCT_EN)[code]
    elif zh:
        body = f"请用带有「{code}」情绪的语气自然地说这句话。"
    else:
        body = f"Speak with a {code} emotional tone, natural and conversational."

    return f"You are a helpful assistant. {body}{_END}"
