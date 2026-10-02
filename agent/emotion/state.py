"""Per-role emotion state + [emotion]/[pad] tag helpers for talk-show turns.

Live path (``TALKSHOW_EMOTION_SOURCE=pad``, default): listeners' separate appraisal
calls (agent/emotion/appraisal.py) emit ``[pad]: <word> ΔP ΔA ΔD``; ``role_emotion``
holds the MSP kNN label (e.g. Anger, Concerned). Legacy closed-set tags remain
only for ``TALKSHOW_EMOTION_SOURCE=llm`` rollback.
"""

from __future__ import annotations

import logging
import os
import re
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from agent.data import TalkShowData

logger = logging.getLogger(__name__)

# Legacy CosyVoice closed set (llm rollback only).
LEGACY_EMOTIONS = frozenset(
    {
        "neutral",
        "amused",
        "curious",
        "serious",
        "skeptical",
        "warm",
        "surprised",
    }
)
# Back-compat alias
EMOTIONS = LEGACY_EMOTIONS
DEFAULT_EMOTION = "Neutral"
LEGACY_DEFAULT_EMOTION = "neutral"
EMOTION_CHOICES = " | ".join(sorted(LEGACY_EMOTIONS))

_EMOTION_RE = re.compile(
    r"\[emotion\]\s*:\s*([A-Za-z][\w-]*)\b"
    r"|\[emotion\s*:\s*([A-Za-z][\w-]*)\s*\]",
    re.IGNORECASE,
)

def _pad_num(name: str) -> str:
    # Models sometimes echo the placeholder brackets: <0.10> / < -0.15>.
    return rf"<?\s*(?P<{name}>[+\-\u2212]?(?:\d+(?:\.\d*)?|\.\d+))\s*>?"


# Optional per-axis label before each number: P=, ΔP:, dA=, D …
_PAD_AXIS = r"(?:(?:\u0394|\u03b4|d)?[pad]\s*[=:]?\s*)?"
_PAD_SEP = r"(?:\s*[,\uff0c;/]\s*|\s+)"
# Optional appraisal feeling phrase (1–4 words, e.g. "mildly challenged") before
# the numbers; an axis label followed by a number (P 0.2, ΔA=…) is never a word.
_PAD_WORD_TOKEN = (
    r"(?!(?:\u0394|\u03b4|d)?[pad]\s*[=:]?\s*[+\-\u2212]?[\d.])[a-z][a-z'-]*"
)
# Words may be space- or comma-separated ("curious, engaged").
_PAD_WORD = (
    rf"(?:(?P<word>{_PAD_WORD_TOKEN}(?:(?:\s*,\s*|\s+){_PAD_WORD_TOKEN}){{0,3}})"
    r"\s*[,:]?\s*)?"
)
# Optional valence class between the feeling and the numbers:
#   [pad]: saddened | unpleasant | -0.25 0.10 -0.05
_VALENCE_WORDS = ("unpleasant", "pleasant", "neutral", "negative", "positive", "mixed")
_PAD_VAL = (
    r"(?:\|\s*)?(?:(?P<val>" + "|".join(_VALENCE_WORDS) + r")\b\s*)?(?:\|\s*)?"
)
# Accepts [pad]: a b c | [pad] a b c | [pad: a, b, c] | [PAD]: P=a A=b D=c
#         [pad]: stung -0.3 0.2 -0.1  (appraisal word)
_PAD_RE = re.compile(
    r"\[\s*pad\s*(?:\]\s*:?|:)\s*"
    + _PAD_WORD
    + _PAD_VAL
    + _PAD_AXIS + _pad_num("p") + _PAD_SEP
    + _PAD_AXIS + _pad_num("a") + _PAD_SEP
    + _PAD_AXIS + _pad_num("d")
    + r"(?:\s*\])?",
    re.IGNORECASE,
)
_PAD_TAG_RE = re.compile(r"\[\s*pad\b", re.IGNORECASE)


def emotion_source() -> str:
    """``pad`` (default) = PAD→kNN MSP tags; ``llm`` = legacy [emotion] closed set."""
    return (os.environ.get("TALKSHOW_EMOTION_SOURCE") or "pad").strip().lower()


def normalize_emotion(raw: str | None) -> str | None:
    """Return a usable emotion label, or None if missing/invalid."""
    if not raw:
        return None
    s = str(raw).strip()
    if not s:
        return None
    key = s.lower()
    if key in LEGACY_EMOTIONS:
        # Preserve lowercase closed-set codes for llm / instruct overrides.
        return key
    from agent.emotion.msp_anchors import canonicalize_emotion_label

    label = canonicalize_emotion_label(s)
    if not label or label.lower() in {"none", "nan"}:
        return None
    return label


def parse_emotion_tag(text: str) -> str | None:
    m = _EMOTION_RE.search(text or "")
    if not m:
        return None
    raw = m.group(1) or m.group(2)
    return normalize_emotion(raw)


def strip_emotion_tag(text: str) -> str:
    cleaned = _EMOTION_RE.sub("", text or "")
    cleaned = re.sub(r"\[emotion\s*:?\s*\]", "", cleaned, flags=re.IGNORECASE)
    return cleaned.strip()


def has_pad_tag(text: str) -> bool:
    """True when the text contains any ``[pad`` tag opener (parsable or not)."""
    return bool(_PAD_TAG_RE.search(text or ""))


@dataclass(frozen=True)
class AppraisalParse:
    word: str | None
    valence: str | None  # pleasant | unpleasant | neutral (normalized) or None
    delta: tuple[float, float, float] | None  # after sign correction
    raw_p: float | None  # ΔP exactly as the model wrote it
    flipped: bool  # ΔP sign was corrected to match the valence class


_VALENCE_NORMAL = {
    "pleasant": "pleasant",
    "positive": "pleasant",
    "unpleasant": "unpleasant",
    "negative": "unpleasant",
    "neutral": "neutral",
    "mixed": "neutral",
}


def parse_pad_appraisal_full(text: str) -> AppraisalParse:
    """Parse ``[pad]: <feeling> | <pleasant|unpleasant|neutral> | ΔP ΔA ΔD``.

    Small models often pick the right feeling and class but the wrong sign on ΔP
    ("deeply saddened +0.35"). The categorical call is the more reliable one, so ΔP's
    sign is made to agree with it (ours): unpleasant → ΔP ≤ 0, pleasant → ΔP ≥ 0.
    """
    none = AppraisalParse(None, None, None, None, False)
    m = _PAD_RE.search(text or "")
    if not m:
        if has_pad_tag(text):
            idx = _PAD_TAG_RE.search(text).start()  # type: ignore[union-attr]
            logger.warning("unparsable [pad] tag: %r", text[idx : idx + 80])
        return none
    try:
        p, a, d = (float(m.group(k).replace("\u2212", "-")) for k in ("p", "a", "d"))
    except ValueError:
        return none

    def _clip(x: float) -> float:
        return max(-1.0, min(1.0, x))

    word = m.group("word")
    word = " ".join(word.lower().replace(",", " ").split()) if word else None
    val = m.group("val")
    if val is None and word:
        # "saddened unpleasant -0.2 …" without pipes: valence swallowed into the phrase.
        toks = word.split()
        if toks[-1] in _VALENCE_NORMAL:
            val = toks[-1]
            word = " ".join(toks[:-1]) or None
    valence = _VALENCE_NORMAL.get(val.lower()) if val else None
    raw_p = _clip(p)
    fixed_p = raw_p
    if valence == "unpleasant" and raw_p > 0:
        fixed_p = -raw_p
    elif valence == "pleasant" and raw_p < 0:
        fixed_p = -raw_p
    flipped = fixed_p != raw_p
    if flipped:
        logger.info("pad valence sign fix: %r %s ΔP %+.2f → %+.2f", word, valence, raw_p, fixed_p)
    return AppraisalParse(word, valence, (fixed_p, _clip(a), _clip(d)), raw_p, flipped)


def parse_pad_appraisal(
    text: str,
) -> tuple[str | None, tuple[float, float, float] | None]:
    """``(feeling word, sign-corrected delta)``; see :func:`parse_pad_appraisal_full`."""
    r = parse_pad_appraisal_full(text)
    return r.word, r.delta


def parse_pad_delta(text: str) -> tuple[float, float, float] | None:
    return parse_pad_appraisal(text)[1]


def strip_pad_tag(text: str) -> str:
    cleaned = _PAD_RE.sub("", text or "")
    cleaned = re.sub(r"\[\s*pad\s*:?\s*\]\s*:?", "", cleaned, flags=re.IGNORECASE)
    return cleaned.strip()


def get_role_emotion(data: TalkShowData, role: str) -> str:
    role = (role or "").strip().lower()
    default = LEGACY_DEFAULT_EMOTION if emotion_source() == "llm" else DEFAULT_EMOTION
    if not role or role == "human":
        return default
    current = data.role_emotion.get(role)
    return normalize_emotion(current) or default


def set_role_emotion(data: TalkShowData, role: str, emotion: str | None) -> str:
    """Update role emotion when tag is valid; otherwise keep previous. Returns current."""
    role = (role or "").strip().lower()
    default = LEGACY_DEFAULT_EMOTION if emotion_source() == "llm" else DEFAULT_EMOTION
    if not role or role == "human":
        return default
    normalized = normalize_emotion(emotion)
    if normalized is None:
        return get_role_emotion(data, role)
    prev = data.role_emotion.get(role)
    data.role_emotion[role] = normalized
    if normalized != prev:
        logger.info("role_emotion role=%s → %s (was %s)", role, normalized, prev)
    return normalized


def apply_emotion_from_parsed(
    data: TalkShowData, role: str, emotion: str | None
) -> str:
    """Apply parsed emotion tag (or keep prior). Returns the role's current emotion."""
    return set_role_emotion(data, role, emotion)


def emotion_prompt_block(data: TalkShowData, role: str) -> str:
    """Legacy llm-mode prompt: current mood + closed-set output contract."""
    current = get_role_emotion(data, role)
    others: list[str] = []
    for other_role, emo in sorted(data.role_emotion.items()):
        if other_role == role:
            continue
        code = normalize_emotion(emo) or LEGACY_DEFAULT_EMOTION
        if code == LEGACY_DEFAULT_EMOTION or code == DEFAULT_EMOTION:
            continue
        from agent.config import load_persona_name

        others.append(f"{load_persona_name(other_role)}={code}")
    others_line = ""
    if others:
        others_line = f"\nOther panel moods (for context): {', '.join(others)}."

    return (
        f"Mood state: you are currently **{current}**.{others_line}\n"
        f"- Keep [reply] wording and tone consistent with that mood.\n"
        f"- Update [emotion] only when the conversation warrants a gradual shift; "
        f"do not jump randomly.\n"
        f"- Output exactly one tag from: {EMOTION_CHOICES}."
    )


def emotion_output_lines() -> str:
    return f"[emotion]: {EMOTION_CHOICES}"


def _other_moods_line(data: TalkShowData, role: str) -> str:
    others: list[str] = []
    for other_role, emo in sorted(getattr(data, "role_emotion", {}).items()):
        if other_role == role:
            continue
        code = normalize_emotion(emo)
        if not code or code in {DEFAULT_EMOTION, LEGACY_DEFAULT_EMOTION}:
            continue
        from agent.config import load_persona_name

        others.append(f"{load_persona_name(other_role)}={code}")
    if not others:
        return ""
    return f"\nOther panel moods (for context): {', '.join(others)}."


def _join_and(words: list[str]) -> str:
    return words[0] if len(words) == 1 else ", ".join(words[:-1]) + " and " + words[-1]


def affect_descriptor(
    pad: tuple[float, float, float] | list[float],
    labels: list[str] | tuple[str, ...] | None = None,
) -> str:
    """Plain-language mood from PAD + kNN neighbor labels (no raw numbers)."""
    p, a, d = (float(x) for x in pad)
    norm = (p * p + a * a + d * d) ** 0.5 / 3**0.5
    if norm < 0.08:
        return "calm and neutral"
    from agent.emotion.knn_map import modal_labels

    # Vague (MSP no-consensus) is not an actionable mood; describe the rest.
    names = [str(x) for x in (labels or ()) if x and str(x) != "Vague"] or ["Neutral"]
    top = modal_labels(names)
    rest = [n for n in dict.fromkeys(names) if n not in top]
    if len(top) > 1:
        # Tie among neighbors: keep every label (Sentipolis: no forced majority).
        blend = "a mix of " + _join_and([t.lower() for t in top])
    else:
        blend = top[0].lower()
    if rest:
        blend += " with a touch of " + _join_and([r.lower() for r in rest])
    if norm < 0.25:
        intensity = "mild"
    elif norm < 0.5:
        intensity = "clear"
    else:
        intensity = "strong"
    traits: list[str] = []
    if a > 0.25:
        traits.append("energized")
    elif a < -0.25:
        traits.append("subdued")
    if d > 0.25:
        traits.append("confident and in control")
    elif d < -0.25:
        traits.append("on the back foot")
    if p < -0.25:
        traits.append("displeased")
    elif p > 0.25:
        traits.append("in good spirits")
    tail = f"; {', '.join(traits)}" if traits else ""
    return f"{blend} ({intensity}){tail}"


def pad_prompt_block(data: TalkShowData, role: str) -> str:
    """PAD mood block: plain-language descriptor only (appraisal owns PAD updates)."""
    from agent.emotion.role_pad import get_role_pad

    pad = get_role_pad(data, role)
    tag = get_role_emotion(data, role)
    neighbors = (getattr(data, "role_pad_neighbors", None) or {}).get(role) or [tag]
    return (
        f"Your current mood: {affect_descriptor(pad, neighbors)}."
        f"{_other_moods_line(data, role)}\n"
        f"- Let this mood color your wording and stance naturally; do not name it "
        f"or describe your feelings explicitly."
    )


def mood_persona_addon() -> str:
    """Mood output contract appended to persona ``instructions`` (llm rollback only)."""
    if emotion_source() == "llm":
        return (
            f"Also end your output with:\n{emotion_output_lines()}\n"
            "Match [reply] tone to [emotion]; shift emotion only gradually."
        )
    return ""


def mood_prompt_block(data: TalkShowData, role: str) -> str:
    if emotion_source() == "llm":
        return emotion_prompt_block(data, role)
    return pad_prompt_block(data, role)


def mood_output_lines() -> str:
    """Extra output line for dialogue prompts — only the legacy llm [emotion] mode."""
    if emotion_source() == "llm":
        return emotion_output_lines()
    return ""


def mood_output_block() -> str:
    """``mood_output_lines()`` + newline, or empty — for "Output exactly" blocks."""
    line = mood_output_lines()
    return f"{line}\n" if line else ""
