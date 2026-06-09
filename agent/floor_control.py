from __future__ import annotations

import logging

from agent.data import TalkShowData

logger = logging.getLogger(__name__)

DIRECT_SPEAKERS = frozenset({"commentator", "guest"})
TERMINAL_NEXT = frozenset({"human", "close"})
VALID_FLOOR_NEXT = DIRECT_SPEAKERS | TERMINAL_NEXT | frozenset({"host"})


def apply_floor_next(data: TalkShowData, role: str | None) -> None:
    """
    Store who speaks next from a Gemma [next]: tag.
    Missing/unknown/host → hand-raise pending (host moderates).
    """
    if not role:
        data.floor_next_speaker = "host"
        return
    key = role.strip().lower()
    if key in VALID_FLOOR_NEXT:
        data.floor_next_speaker = key
    else:
        data.floor_next_speaker = "host"


def resolve_floor_after_host_speech(
    data: TalkShowData,
    *,
    spoken: str,
    tagged_next: str | None,
    tee_fallback: bool = False,
    after_human_turn: bool = False,
) -> str:
    """
    Apply floor_next from a Gemma [next] tag; reconcile when tag says host but tee-up names someone.
    Returns the resolved role id.

    When ``after_human_turn`` is True, always hand off to host moderation (open floor + poll + FIFO)
    except explicit [next:human] or [next:close].
    """
    tag = (tagged_next or "").strip().lower()
    if tag == "human":
        apply_floor_next(data, "human")
        return "human"
    if tag == "close":
        apply_floor_next(data, "close")
        return "close"
    if after_human_turn:
        apply_floor_next(data, "host")
        logger.info(
            "floor next=host (after human turn; tag=%r tee_fallback=%s)",
            tag or None,
            tee_fallback,
        )
        return "host"
    if tag in ("commentator", "guest"):
        apply_floor_next(data, tag)
        return tag
    if tee_fallback:
        apply_floor_next(data, "commentator")
        logger.info("floor next=commentator (programmatic tee-up fallback)")
        return "commentator"

    data.last_host_panel_tee = spoken.strip()
    from agent.floor_parser import infer_host_called_role

    called = infer_host_called_role(data)
    if called in DIRECT_SPEAKERS:
        apply_floor_next(data, called)
        logger.info(
            "floor reconciled next=%s (host tee named them; tag was %r)",
            called,
            tag or None,
        )
        return called

    apply_floor_next(data, "host")
    return "host"


def consume_floor_next(data: TalkShowData) -> str:
    role = (data.floor_next_speaker or "host").strip().lower()
    data.floor_next_speaker = ""
    return role


def peek_floor_next(data: TalkShowData) -> str:
    return (data.floor_next_speaker or "host").strip().lower()


def is_direct_next(role: str) -> bool:
    return role in DIRECT_SPEAKERS


def is_hand_raise_pending(role: str) -> bool:
    return role in ("host", "")
