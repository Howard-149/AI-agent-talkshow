"""Host-moderated floor: session welcome, return to host, graph entry points."""

from __future__ import annotations

import logging

from livekit.agents import AgentSession

from agent.data import TalkShowData
from agent.floor.floor_control import (
    apply_floor_next,
    peek_floor_next,
)
from agent.show.show_history import append_role
from agent.show.show_topic import show_topic
from agent.session.session_handoff import speak_panel_line
from agent.floor import TurnController

logger = logging.getLogger(__name__)


def session_welcome_line(data: TalkShowData) -> tuple[str, dict[str, str]]:
    """Canonical (English) welcome text + per-locale delivery texts for the opening."""
    from agent.config import text_only_enabled
    from agent.locale.viewer_locales import recompute_needed_locales
    from agent.panel.panel_context import session_welcome_texts

    room = None
    if not text_only_enabled():
        try:
            from livekit.agents.job import get_job_context

            room = get_job_context().room
        except Exception:  # no LiveKit job / room
            room = None
    if room is not None:
        try:
            recompute_needed_locales(data, room)
        except Exception:
            logger.exception("session_welcome: could not recompute locales from room")

    needed = frozenset(getattr(data, "needed_locales", None) or {"en"})
    canned = session_welcome_texts(data.scenario)
    texts = {loc: canned[loc] for loc in needed if loc in canned}
    if not texts:
        texts = {"en": canned["en"]}
    return canned["en"], texts


TOPIC_PREFIX = "Today's discussion topic:"


def topic_announcement(default: str = "") -> str:
    """The host's topic line spoken right after the welcome ("" when no topic is set)."""
    topic = show_topic(default)
    return f"{TOPIC_PREFIX} {topic}" if topic else ""


async def host_announce_topic(
    session: AgentSession | None,
    data: TalkShowData,
    controller: TurnController,
    *,
    default: str = "",
) -> None:
    """After the welcome the host announces the topic (an on-record show line)."""
    text = topic_announcement(default)
    if not text:
        return
    listen = controller.listen_role()
    append_role(data, listen, text, appraise=False, procedural=False)
    log_show_topic(data, text)
    await speak_panel_line(session, data, speak_role=listen, text=text, step="topic")


async def host_opening_line(data: TalkShowData, *, default: str = "") -> str:
    """The host's one opening turn: self-introduction, today's topic, the question to
    weigh, and an invitation to the first panelist (who then gets the first pick).

    Falls back to the canned welcome + topic announcement when no topic is set or
    the model returns nothing usable."""
    from agent.config import load_persona_instructions, load_persona_name
    from agent.floor.floor_parser import spoken_reply_text

    canonical, _ = session_welcome_line(data)
    topic = show_topic(default)
    canned = f"{canonical} {topic_announcement(default)}".strip()
    if not topic:
        return canned
    invite = _opening_invite_role(data)
    invite_line = (
        f"End by inviting {load_persona_name(invite)} by name to share their view first. "
        if invite
        else "End by opening the floor to the panel. "
    )
    prompt = (
        f"The show is starting now. You are {load_persona_name('host')}, the host.\n"
        f"Today's topic: {topic}\n"
        "In one natural spoken turn of three or four sentences, in your own words: "
        "introduce yourself, bring in today's topic, and frame the question the panel "
        "will weigh. Let the audience know they can jump in at any time. "
        f"{invite_line}"
        "Output only the spoken words, with no labels or control tags."
    )
    try:
        raw = await data.runtime.gemma_client.complete_text(
            prompt,
            system_prompt=load_persona_instructions("host", panel_mode=True),
            history_messages=[],
            raw=True,
            max_tokens=256,
            temperature=0.6,
            reasoning_effort="none",
        )
    except Exception:
        logger.exception("host opening generation failed — using canned welcome + topic")
        return canned
    text = spoken_reply_text(raw)
    if not text:
        return canned
    interaction = getattr(data, "floor_interaction", None)
    if invite and interaction is not None:
        # The opening handed the floor to ``invite``: the first pick honours that.
        interaction.pending_invitation_target = invite
        interaction.opening_invite_role = invite
    return text


def _opening_invite_role(data: TalkShowData) -> str | None:
    """First panelist in the scenario order (the host's opening invites them)."""
    turn_control = getattr(getattr(data, "scenario", None), "turn_control", None)
    if turn_control is None:
        return None
    skip = {"human", turn_control.listen_role}
    return next((role for role in turn_control.order if role not in skip), None)


HOST_CLOSE_FALLBACK = (
    "That brings our discussion to a close. Thanks to our panel and to everyone watching."
)


async def host_closing_line(data: TalkShowData) -> str:
    """The host's closing turn: a brief wrap-up of the discussion, with no new question."""
    from agent.config import load_persona_instructions, load_persona_name
    from agent.floor.floor_parser import spoken_reply_text

    topic = show_topic()
    topic_line = f"Topic: {topic}\n" if topic else ""
    prompt = (
        f"The discussion is over and the show is ending now. You are "
        f"{load_persona_name('host')}, the host.\n"
        f"{topic_line}"
        "In one natural spoken turn of three or four sentences, in your own words: "
        "sum up the main points the panel made and where they landed, then wrap up the "
        "show and thank the panel and the audience. Do not ask a question or invite "
        "anyone to continue. Output only the spoken words, with no labels or control tags."
    )
    try:
        raw = await data.runtime.gemma_client.complete_text(
            prompt,
            system_prompt=load_persona_instructions("host", panel_mode=True),
            history_messages=data.show_history.prior_messages(perspective="host"),
            raw=True,
            max_tokens=256,
            temperature=0.5,
            reasoning_effort="none",
        )
    except Exception:
        logger.exception("host closing generation failed — using canned close")
        return HOST_CLOSE_FALLBACK
    return spoken_reply_text(raw) or HOST_CLOSE_FALLBACK


async def host_speak_opening(
    session: AgentSession | None,
    data: TalkShowData,
    controller: TurnController,
    *,
    default: str = "",
) -> None:
    """Opening beat outside the show graph: the host's merged welcome + topic turn."""
    text = await host_opening_line(data, default=default)
    listen = controller.listen_role()
    append_role(data, listen, text, appraise=False, procedural=False)
    topic = topic_announcement(default)
    if topic:
        log_show_topic(data, topic)
    await speak_panel_line(session, data, speak_role=listen, text=text, step="opening")
    apply_floor_next(data, "host")


def log_show_topic(data: TalkShowData, text: str) -> None:
    interaction = getattr(data, "floor_interaction", None)
    if interaction is not None:
        interaction.log_topic(data, text)


async def host_speak_session_welcome(
    session: AgentSession,
    data: TalkShowData,
    controller: TurnController,
) -> None:
    """Opening beat — welcome only; no topic, no direct call-outs."""
    from agent.locale.viewer_locales import primary_delivery_locale

    canonical, texts = session_welcome_line(data)
    needed = frozenset(getattr(data, "needed_locales", None) or {"en"})
    primary = primary_delivery_locale(needed)
    listen = controller.listen_role()
    # History stays English for shared transcript / prompts.
    append_role(data, listen, canonical, appraise=False)
    await speak_panel_line(
        session,
        data,
        speak_role=listen,
        text=canonical,
        step="session_welcome",
        texts=texts,
    )
    apply_floor_next(data, "host")
    logger.info(
        "session_welcome spoken primary=%s needed=%s line=%.60r",
        primary,
        sorted(needed),
        texts.get(primary) or canonical,
    )


async def run_host_moderation_from_queue(
    session: AgentSession,
    data: TalkShowData,
    controller: TurnController,
    *,
    trigger: str = "host_moderate",
    skip_open_floor: bool = False,
    depth: int = 0,
    spoken_roles: set[str] | None = None,
) -> str | None:
    """
    Whenever the host must pick the next speaker: poll into queue, resolve, grant.
    Orchestrated by LangGraph (agent.show_graph); actuators stay in this module.
    """
    from agent.show_graph.runner import (
        run_host_moderation_from_queue as _langgraph_moderation,
    )

    return await _langgraph_moderation(
        session,
        data,
        controller,
        trigger=trigger,
        skip_open_floor=skip_open_floor,
        depth=depth,
        spoken_roles=spoken_roles,
    )


async def return_floor_to_host(
    session: AgentSession,
    data: TalkShowData,
    controller: TurnController,
    *,
    trigger: str,
) -> None:
    """Switch active role back to host — host moderates who speaks next, not auto-human."""
    from agent.session.session_handoff import switch_to_role

    listen = controller.listen_role()
    if data.active_role != listen:
        await switch_to_role(session, data, listen, reason=f"return_host:{trigger}")
    data.touch_activity()
    logger.info(
        "returned floor to host trigger=%s pending_next=%s",
        trigger,
        peek_floor_next(data),
    )
    if data.turn_log:
        data.turn_log.log(
            "floor_return_host",
            trigger=trigger,
            room=data.room_name,
            active_role=listen,
            floor_next=peek_floor_next(data),
        )


async def run_host_moderated_panel(
    session: AgentSession,
    data: TalkShowData,
    controller: TurnController,
) -> None:
    """
    Floor routing via [next] tags:
    - panelist role → direct grant (chain if panelists tag each other)
    - host / unset → hand-raise pending, then host moderates

    Orchestrated by LangGraph (agent.show_graph).
    """
    from agent.show_graph.runner import (
        run_host_moderated_panel as _langgraph_panel,
    )

    await _langgraph_panel(session, data, controller)
