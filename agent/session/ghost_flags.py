"""Ghost-session guards (no LiveKit / adapter imports)."""

from __future__ import annotations

import json
import os


def ghost_turns_allowed() -> bool:
    """Kill switch. Default on — sender must still be a ghost participant."""
    raw = os.environ.get("TALKSHOW_GHOST_TURNS", "1").strip().lower()
    return raw not in ("0", "false", "no", "off")


def ghost_replay_enabled() -> bool:
    """Prefix replay for controlled experiments (TALKSHOW_GHOST_REPLAY). Default off."""
    raw = os.environ.get("TALKSHOW_GHOST_REPLAY", "0").strip().lower()
    return raw in ("1", "true", "yes", "on")


def is_ghost_participant(participant: object | None) -> bool:
    """True for identity ``ghost-*`` or token metadata ``{"ghost": true}``."""
    if participant is None:
        return False
    identity = (getattr(participant, "identity", None) or "").strip()
    if identity.startswith("ghost-"):
        return True
    raw = getattr(participant, "metadata", None) or ""
    if not raw:
        return False
    try:
        meta = json.loads(raw)
    except json.JSONDecodeError:
        return False
    return bool(isinstance(meta, dict) and meta.get("ghost"))


def _meta(participant: object) -> dict:
    raw = getattr(participant, "metadata", None) or ""
    try:
        meta = json.loads(raw) if raw else {}
    except json.JSONDecodeError:
        return {}
    return meta if isinstance(meta, dict) else {}


def is_ghost_viewer(participant: object) -> bool:
    """The headless page a recorded ghost session films (identity ``ghost-viewer*``)."""
    identity = (getattr(participant, "identity", None) or "").strip()
    return identity.startswith("ghost-viewer") or bool(_meta(participant).get("ghostViewer"))


def awaits_viewer(room: object) -> bool:
    """A ghost asked the show to wait for its recording page (token ``awaitViewer``)."""
    remote = getattr(room, "remote_participants", {}) or {}
    return any(
        is_ghost_participant(p) and _meta(p).get("awaitViewer") for p in remote.values()
    )


async def wait_for_ghost_viewer(room: object, *, timeout_sec: float = 60.0) -> bool:
    """Hold the opening until the recording page has joined and had a moment to show
    the panel, so the video doesn't start seconds into the host's first words.
    Returns at once unless a ghost asked for it (``awaitViewer``)."""
    import asyncio

    if not awaits_viewer(room):
        return False
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout_sec
    while not any(is_ghost_viewer(p) for p in (getattr(room, "remote_participants", {}) or {}).values()):
        if loop.time() >= deadline:
            return False
        await asyncio.sleep(0.1)
    # Joined is not yet showing: the page still subscribes to the avatar tracks.
    await asyncio.sleep(float(os.environ.get("TALKSHOW_GHOST_VIEWER_GRACE_S", "2.0")))
    return True
