"""Post-generation localization: canonical English → viewer locales."""

from __future__ import annotations

import hashlib
import logging
from typing import TYPE_CHECKING

from agent.locale.viewer_locales import DEFAULT_LOCALE, SUPPORTED_LOCALES, normalize_locale

if TYPE_CHECKING:
    from agent.data import TalkShowData

logger = logging.getLogger(__name__)

_TRANSLATE_SYSTEM = (
    "You are a precise translator for a live talk-show. "
    "Translate the user message into the target language. "
    "Keep the same meaning, tone, and brevity. "
    "Output only the translation — no quotes, labels, or [heard]/[reply]/[next] tags."
)

# (source_hash, target_locale) → translated text
_cache: dict[tuple[str, str], str] = {}


def _cache_key(text: str, target: str) -> tuple[str, str]:
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()[:24]
    return digest, target


async def translate_text(
    data: TalkShowData,
    text: str,
    *,
    target_locale: str,
    source_locale: str = DEFAULT_LOCALE,
) -> str:
    """Translate text into target_locale. Identity when source == target."""
    text = text.strip()
    if not text:
        return ""
    target = normalize_locale(target_locale)
    source = normalize_locale(source_locale)
    if target == source:
        return text
    if target not in SUPPORTED_LOCALES:
        return text

    key = _cache_key(text, target)
    cached = _cache.get(key)
    if cached is not None:
        return cached

    lang_name = "Simplified Chinese" if target == "zh" else "English"
    src_name = "Simplified Chinese" if source == "zh" else "English"
    user = (
        f"Source language: {src_name}\n"
        f"Target language: {lang_name}\n\n"
        f"{text}"
    )
    try:
        out = await data.runtime.gemma_client.complete_text(
            user,
            system_prompt=_TRANSLATE_SYSTEM,
            history_messages=None,
        )
        out = (out or "").strip()
        # Strip accidental wrapping quotes
        if len(out) >= 2 and out[0] == out[-1] and out[0] in "\"'「」":
            out = out[1:-1].strip()
    except Exception:
        logger.exception("translate_text failed target=%s; using source text", target)
        out = text

    if not out:
        out = text
    _cache[key] = out
    return out


async def texts_for_needed_locales(
    data: TalkShowData,
    canonical_en: str,
    *,
    source_locale: str = DEFAULT_LOCALE,
) -> dict[str, str]:
    """Build texts{} for each locale in data.needed_locales.

    ``canonical_en`` is the English show line (or English [heard]). When the
    source is already Chinese, pass source_locale='zh' and we still fill ``en``.
    """
    canonical_en = canonical_en.strip()
    needed = getattr(data, "needed_locales", None) or frozenset({DEFAULT_LOCALE})
    source = normalize_locale(source_locale)
    texts: dict[str, str] = {}

    if source == DEFAULT_LOCALE:
        base_en = canonical_en
    else:
        base_en = await translate_text(
            data, canonical_en, target_locale=DEFAULT_LOCALE, source_locale=source
        )

    for loc in sorted(needed):
        if loc == DEFAULT_LOCALE:
            texts[loc] = base_en
        elif source == loc:
            texts[loc] = canonical_en
        else:
            texts[loc] = await translate_text(
                data, base_en, target_locale=loc, source_locale=DEFAULT_LOCALE
            )
    return texts
