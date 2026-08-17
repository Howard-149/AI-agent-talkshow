"""Track which UI locales live humans need (demand-driven TTS / avatar / texts)."""

from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING, Iterable

from agent.data import TalkShowData

if TYPE_CHECKING:
    from livekit import rtc

logger = logging.getLogger(__name__)

SUPPORTED_LOCALES = frozenset({"en", "zh"})
DEFAULT_LOCALE = "en"


def normalize_locale(raw: str | None) -> str:
    code = (raw or "").strip().lower().replace("_", "-")
    if not code:
        return DEFAULT_LOCALE
    if code in SUPPORTED_LOCALES:
        return code
    if code.startswith("zh"):
        return "zh"
    if code.startswith("en"):
        return "en"
    return DEFAULT_LOCALE


def locale_from_metadata(raw: str | None) -> str | None:
    """Return locale from participant metadata, or None if absent/invalid JSON."""
    if not raw or not raw.strip():
        return None
    try:
        meta = json.loads(raw)
    except json.JSONDecodeError:
        return None
    if not isinstance(meta, dict):
        return None
    if meta.get("talkshowAgent"):
        return None
    if "locale" not in meta:
        return None
    return normalize_locale(str(meta.get("locale") or ""))


def is_talkshow_agent(participant: object) -> bool:
    raw = getattr(participant, "metadata", None) or ""
    if not raw:
        return False
    try:
        meta = json.loads(raw)
    except json.JSONDecodeError:
        return False
    return bool(isinstance(meta, dict) and meta.get("talkshowAgent"))


def recompute_needed_locales(data: TalkShowData, room: object) -> frozenset[str]:
    """Union of locales from remote humans + explicit overrides; default {en} if empty."""
    locales: set[str] = set()
    remote_participants = getattr(room, "remote_participants", {}) or {}
    for participant in remote_participants.values():
        if is_talkshow_agent(participant):
            continue
        identity = getattr(participant, "identity", None) or ""
        override = data.viewer_locale_by_identity.get(identity)
        if override:
            locales.add(normalize_locale(override))
            logger.debug(
                "locale identity=%s source=override locale=%s", identity, override
            )
            continue
        raw_meta = getattr(participant, "metadata", None)
        from_meta = locale_from_metadata(raw_meta)
        if from_meta:
            locales.add(from_meta)
            logger.info(
                "locale identity=%s source=token_metadata locale=%s meta=%r",
                identity,
                from_meta,
                (raw_meta or "")[:80],
            )
        else:
            locales.add(DEFAULT_LOCALE)
            logger.warning(
                "locale identity=%s missing token metadata — defaulting to %s meta=%r",
                identity,
                DEFAULT_LOCALE,
                (raw_meta or "")[:80],
            )

    if not locales:
        locales.add(DEFAULT_LOCALE)

    needed = frozenset(locales & SUPPORTED_LOCALES) or frozenset({DEFAULT_LOCALE})
    prev = getattr(data, "needed_locales", None)
    data.needed_locales = needed
    if needed != prev:
        logger.info("needed_locales → %s", sorted(needed))
    return needed


def remote_humans(room: object) -> list[object]:
    remote = getattr(room, "remote_participants", {}) or {}
    return [p for p in remote.values() if not is_talkshow_agent(p)]


async def wait_for_remote_humans(
    room: object,
    *,
    timeout_sec: float = 30.0,
) -> bool:
    """Return as soon as ≥1 non-agent remote is in the room (already present ⇒ immediate)."""
    import asyncio

    deadline = asyncio.get_running_loop().time() + max(0.0, timeout_sec)
    while True:
        if remote_humans(room):
            return True
        if asyncio.get_running_loop().time() >= deadline:
            logger.warning(
                "wait_for_remote_humans timed out after %.0fs — opening may default to en",
                timeout_sec,
            )
            return False
        await asyncio.sleep(0.05)


def set_viewer_locale(
    data: TalkShowData,
    room: object,
    *,
    identity: str,
    locale: str,
) -> frozenset[str]:
    data.viewer_locale_by_identity[identity] = normalize_locale(locale)
    return recompute_needed_locales(data, room)


def primary_delivery_locale(needed: Iterable[str]) -> str:
    """Locale that drives session.say + default avatar track (prefer en when mixed)."""
    s = frozenset(needed)
    if DEFAULT_LOCALE in s:
        return DEFAULT_LOCALE
    for code in sorted(s):
        return code
    return DEFAULT_LOCALE


def avatar_track_name(locale: str | None = None) -> str:
    """Primary/compat track stays talkshow-avatar; secondary locales get a suffix."""
    code = normalize_locale(locale) if locale else DEFAULT_LOCALE
    if code == DEFAULT_LOCALE:
        return "talkshow-avatar"
    return f"talkshow-avatar-{code}"


def audio_track_name(locale: str) -> str:
    """Named secondary audio; primary locale uses the agent session mic track."""
    return f"talkshow-audio-{normalize_locale(locale)}"
