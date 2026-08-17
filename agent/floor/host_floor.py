"""Host-moderated floor: poll batch, priority tie-break, queue call-ons, welcome."""

from __future__ import annotations

import asyncio
import logging
import os
import time

from livekit.agents import AgentSession

from agent.config import load_persona_name
from agent.data import TalkShowData
from agent.floor.floor_control import (
    apply_floor_next,
    consume_floor_next,
    is_direct_next,
    is_hand_raise_pending,
    peek_floor_next,
)
from agent.floor.floor_parser import (
    HandRaiseResult,
    fallback_floor_decision,
    parse_floor_decision,
    parse_hand_raise,
)
from agent.floor.hand_raise_ui import (
    dequeue_hand_raise,
    flash_poll_raises,
    sync_hand_raise_ui,
    wait_for_hand_raises,
)
from agent.floor.host_lines import (
    host_human_floor_line,
    host_intro_speaker_line,
    host_open_floor_line,
)
from agent.panel.panel_prompts import panelist_hand_raise_system
from agent.panel.panel_speech import PANEL_HOST_CLOSE, speak_one_panelist
from agent.session.session_lifecycle import should_stop_session_work
from agent.show.show_history import append_role
from agent.session.session_handoff import speak_panel_line
from agent.floor import TurnController
from agent.floor.host_direct_call import host_direct_call_when_no_raises
from agent.ui.ui_events import (
    emit_floor_grant,
    emit_floor_pending,
)

logger = logging.getLogger(__name__)


def _eligible_panel_raises(
    data: TalkShowData, panel_roles: list[str], spoken_roles: set[str]
) -> list[str]:
    return [
        r
        for r in data.hand_raise_queue.raised_roles(panel_roles)
        if r not in spoken_roles
    ]


async def _run_hand_raise_round(
    session: AgentSession,
    data: TalkShowData,
    controller: TurnController,
    *,
    panel_roles: list[str],
    spoken_roles: set[str],
    turn_idx: int,
) -> tuple[str | None, str]:
    """
    Pick next speaker: FIFO queue (human UI + AI poll).
    Poll only when queue is empty; wait 10s only when still empty after poll.
    """
    if should_stop_session_work(data, session):
        return None, ""
    await emit_floor_pending(active=True)
    try:
        if not data.hand_raise_queue.roles():
            t0 = time.monotonic()
            poll = await poll_panel_hand_raises(data, panel_roles)
            poll_latency_s = time.monotonic() - t0
            await flash_poll_raises(data, panel_roles=panel_roles, poll=poll)
            winner, yes_roles, had_tie = data.hand_raise_queue.apply_poll_batch(
                poll,
                data.panel_priority,
            )
            if had_tie and winner:
                data.rotate_panel_priority(winner)
            await sync_hand_raise_ui(data, panel_roles=panel_roles, phase="poll")
            if data.turn_log:
                data.turn_log.log(
                    "hand_raise_poll",
                    results={
                        role: {"raised": hr.raised, "topic": hr.topic[:40]}
                        for role, hr in poll.items()
                    },
                    winner=winner,
                    yes_roles=yes_roles,
                    had_tie=had_tie,
                    poll_latency_s=round(poll_latency_s, 3),
                    panel_priority=list(data.panel_priority),
                    room=data.room_name,
                    queue=data.hand_raise_queue.roles(),
                )
        else:
            logger.info(
                "hand_raise poll skip queue=%s",
                data.hand_raise_queue.roles(),
            )
            if data.turn_log:
                data.turn_log.log(
                    "hand_raise_poll_skip",
                    reason="queue_nonempty",
                    room=data.room_name,
                    queue=data.hand_raise_queue.roles(),
                )

        if not data.hand_raise_queue.roles():
            await wait_for_hand_raises(data, panel_roles=panel_roles)
            await sync_hand_raise_ui(data, panel_roles=panel_roles, phase="wait")
        else:
            logger.info(
                "hand_raise wait skip queue=%s",
                data.hand_raise_queue.roles(),
            )
            if data.turn_log:
                data.turn_log.log(
                    "hand_raise_wait_skip",
                    reason="queue_nonempty",
                    room=data.room_name,
                    queue=data.hand_raise_queue.roles(),
                )

        if data.hand_raise_queue.roles():
            pause = float(os.environ.get("TALKSHOW_HAND_RAISE_GRANT_PAUSE_SEC", "1"))
            if pause > 0:
                if should_stop_session_work(data, session):
                    return None, ""
                logger.info(
                    "hand raise grant pause sec=%.2f queue=%s",
                    pause,
                    data.hand_raise_queue.roles(),
                )
                if data.turn_log:
                    data.turn_log.log(
                        "hand_raise_grant_pause",
                        pause_sec=pause,
                        room=data.room_name,
                        queue=data.hand_raise_queue.roles(),
                    )
                await asyncio.sleep(pause)
                if should_stop_session_work(data, session):
                    return None, ""
            return await resolve_next_speaker(
                data,
                panel_roles=panel_roles,
                spoken_roles=spoken_roles,
            )

        return await host_direct_call_when_no_raises(
            session,
            data,
            controller,
            panel_roles=panel_roles,
            spoken_roles=spoken_roles,
            trigger=f"no_raises_{turn_idx}",
        )
    finally:
        await emit_floor_pending(active=False)


async def _poll_one_hand_raise(data: TalkShowData, role: str) -> HandRaiseResult:
    name = load_persona_name(role)
    hist = data.show_history.prior_messages()
    prompt = f"""Panel floor check — you are {name}.

Read the transcript. Do you want to raise your hand to speak next?
You may speak multiple times this round. If you want to OPEN a new topic angle, say so.

Reply ONLY:
[raise]: yes
or
[raise]: no
[topic]: optional topic you want to bring up (if yes)
[reason]: optional one short phrase (if yes)
"""
    text = await data.runtime.gemma_client.complete_text(
        prompt,
        system_prompt=panelist_hand_raise_system(role),
        history_messages=hist,
    )
    result = parse_hand_raise(role, text)
    logger.info(
        "hand_raise poll role=%s raised=%s topic=%.40r reason=%.40r",
        role,
        result.raised,
        result.topic,
        result.reason,
    )
    return result


async def poll_panel_hand_raises(
    data: TalkShowData, panel_roles: list[str]
) -> dict[str, HandRaiseResult]:
    """Parallel Gemma polls — same worker, asyncio.gather (not separate processes)."""
    results = await asyncio.gather(
        *[_poll_one_hand_raise(data, role) for role in panel_roles]
    )
    return {r.role: r for r in results}


def _human_hand_raise_result(data: TalkShowData) -> HandRaiseResult | None:
    entry = data.hand_raise_queue.get("human")
    if not entry:
        return None
    return HandRaiseResult(
        role="human",
        raised=True,
        reason=entry.reason,
        topic=entry.topic,
    )


def _queue_raises(data: TalkShowData, panel_roles: list[str]) -> dict[str, HandRaiseResult]:
    return data.hand_raise_queue.snapshot(panel_roles)


async def _host_decide_floor(
    data: TalkShowData,
    *,
    panel_roles: list[str],
    spoken_roles: set[str],
) -> str:
    host = load_persona_name("host")
    from agent.panel.panel_context import next_tag_options, panel_role_name_map

    role_map = panel_role_name_map(panel_roles)
    next_opts = next_tag_options(panel_roles, include_host=False) + " | human | close"
    raises = _queue_raises(data, panel_roles)
    lines = []
    for role in panel_roles:
        hr = raises[role]
        label = load_persona_name(role)
        if hr.raised:
            topic_bit = f' topic="{hr.topic}"' if hr.topic else ""
            lines.append(
                f"- {label}: RAISED — {hr.reason or '(no reason)'}{topic_bit}"
            )
        else:
            lines.append(f"- {label}: not raised")
    human_hr = _human_hand_raise_result(data)
    if human_hr:
        topic_bit = f' topic="{human_hr.topic}"' if human_hr.topic else ""
        lines.append(
            f"- Human guest: RAISED — {human_hr.reason or '(no reason)'}{topic_bit}"
        )
    else:
        lines.append("- Human guest: not raised")
    raise_block = "\n".join(lines)
    queue_order = data.hand_raise_queue.roles()
    order_hint = ""
    if len(queue_order) > 1:
        order_hint = f"\nQueue order (FIFO): {', '.join(queue_order)}"

    prompt = f"""You are {host}, moderator. Hand-raise status:
{raise_block}{order_hint}

Rules:
- [next:human] returns floor to the human guest (they speak via mic, not TTS)
- If multiple panelists raised, pick the best next speaker
- If none raised, pick who should speak OR [next:close] OR [next:human] if the guest raised
- Panelist names: {role_map}
- Panelists may speak more than once; honor [topic] angles when choosing

Output ONLY:
[next]: {next_opts}
[reason]: one short sentence (optional)
"""
    hist = data.show_history.prior_messages()
    from agent.config import load_persona_instructions

    system = load_persona_instructions("host", panel_mode=True)
    text = await data.runtime.gemma_client.complete_text(
        prompt,
        system_prompt=system,
        history_messages=hist,
    )
    parsed = parse_floor_decision(text)
    if parsed and parsed.next_role in set(panel_roles) | {"close", "human"}:
        logger.info(
            "host floor decision next=%s reason=%.60r",
            parsed.next_role,
            parsed.reason,
        )
        return parsed.next_role

    fb = fallback_floor_decision(
        panel_roles=panel_roles,
        raises=raises,
        spoken_roles=spoken_roles,
    )
    logger.warning("host floor parse failed; fallback next=%s raw=%.80r", fb, text)
    return fb


async def resolve_next_speaker(
    data: TalkShowData,
    *,
    panel_roles: list[str],
    spoken_roles: set[str],
) -> tuple[str, str]:
    """Pick next speaker from the hand-raise queue (strict FIFO)."""
    first = data.hand_raise_queue.peek_first_eligible(panel_roles)
    if first is not None:
        logger.info(
            "queue fifo pick role=%s queue=%s",
            first.role,
            data.hand_raise_queue.roles(),
        )
        if data.turn_log:
            data.turn_log.log(
                "queue_fifo_pick",
                role=first.role,
                queue=data.hand_raise_queue.roles(),
                room=data.room_name,
            )
        return first.role, "queue_fifo"
    raised = _eligible_panel_raises(data, panel_roles, spoken_roles)
    if not raised:
        fb = fallback_floor_decision(
            panel_roles=panel_roles,
            raises=_queue_raises(data, panel_roles),
            spoken_roles=spoken_roles,
        )
        return fb, "queue_empty_fallback"
    return (
        await _host_decide_floor(
            data,
            panel_roles=panel_roles,
            spoken_roles=spoken_roles,
        ),
        "host_moderation",
    )


async def _grant_panelist_turn(
    session: AgentSession,
    data: TalkShowData,
    controller: TurnController,
    role: str,
    *,
    grant_reason: str,
    trigger: str,
    step: str,
    skip_intro: bool = False,
) -> None:
    await emit_floor_grant(role, reason=grant_reason)
    await dequeue_hand_raise(data, role)
    if not skip_intro:
        await host_introduce_speaker(
            session, data, controller, role, trigger=trigger
        )
    await speak_one_panelist(
        session,
        data,
        speak_role=role,
        step=step,
    )


async def host_speak_session_welcome(
    session: AgentSession,
    data: TalkShowData,
    controller: TurnController,
) -> None:
    """Opening beat — welcome only; no topic, no direct call-outs."""
    from agent.locale.viewer_locales import (
        primary_delivery_locale,
        recompute_needed_locales,
    )
    from agent.panel.panel_context import session_welcome_texts

    try:
        from livekit.agents.job import get_job_context

        room = get_job_context().room
        recompute_needed_locales(data, room)
    except Exception:
        logger.exception("session_welcome: could not recompute locales from room")

    needed = frozenset(getattr(data, "needed_locales", None) or {"en"})
    canned = session_welcome_texts(data.scenario)
    texts = {loc: canned[loc] for loc in needed if loc in canned}
    if not texts:
        texts = {"en": canned["en"]}
    primary = primary_delivery_locale(needed)
    line = texts.get(primary) or canned["en"]
    listen = controller.listen_role()
    # History stays English for shared transcript / prompts.
    append_role(data, listen, canned["en"])
    await speak_panel_line(
        session,
        data,
        speak_role=listen,
        text=canned["en"],
        step="session_welcome",
        texts=texts,
    )
    apply_floor_next(data, "host")
    logger.info(
        "session_welcome spoken primary=%s needed=%s line=%.60r",
        primary,
        sorted(needed),
        line,
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
    Returns granted role id, or None when the beat ends with host still pending.
    """
    if should_stop_session_work(data, session):
        return None
    max_depth = int(os.environ.get("TALKSHOW_HOST_MODERATE_MAX_DEPTH", "8"))
    if depth >= max_depth:
        logger.warning("host moderation max depth=%d trigger=%s", max_depth, trigger)
        return None
    panel_roles = controller.panel_speaker_roles()
    spoken = set(spoken_roles or ())
    queue_before = data.hand_raise_queue.roles()
    if not skip_open_floor and not queue_before:
        await host_speak_open_floor(session, data, controller, trigger=trigger)
    elif queue_before:
        logger.info(
            "open floor skip queue=%s trigger=%s",
            queue_before,
            trigger,
        )
        if data.turn_log:
            data.turn_log.log(
                "open_floor_skip",
                reason="queue_nonempty",
                trigger=trigger,
                room=data.room_name,
                queue=queue_before,
            )
    next_role, grant_reason = await _run_hand_raise_round(
        session,
        data,
        controller,
        panel_roles=panel_roles,
        spoken_roles=spoken,
        turn_idx=0,
    )
    if next_role == "close":
        return "close"
    if is_direct_next(next_role, data.scenario):
        await _grant_panelist_turn(
            session,
            data,
            controller,
            next_role,
            grant_reason=grant_reason,
            trigger=trigger,
            step=f"queue_{next_role}_{trigger}",
            skip_intro=grant_reason == "host_direct_no_raises",
        )
        spoken.add(next_role)
        while is_direct_next(peek_floor_next(data)):
            chained = consume_floor_next(data)
            await emit_floor_grant(chained, reason="next_tag")
            await dequeue_hand_raise(data, chained)
            await speak_one_panelist(
                session,
                data,
                speak_role=chained,
                step=f"chain_{chained}_{trigger}",
            )
            spoken.add(chained)
        await return_floor_to_host(
            session, data, controller, trigger=f"queue_{trigger}"
        )
        if is_hand_raise_pending(peek_floor_next(data)):
            nested = await run_host_moderation_from_queue(
                session,
                data,
                controller,
                trigger=f"after_{trigger}",
                depth=depth + 1,
                spoken_roles=spoken,
            )
            if nested in ("human", "close"):
                return nested
        return next_role
    if next_role == "human":
        await grant_human_floor(session, data, controller, reason=grant_reason)
        return "human"
    return None


async def host_speak_open_floor(
    session: AgentSession,
    data: TalkShowData,
    controller: TurnController,
    *,
    trigger: str,
) -> None:
    """Host opens hand-raise moderation before panelists are polled."""
    listen = controller.listen_role()
    line = host_open_floor_line()
    append_role(data, listen, line)
    await speak_panel_line(
        session,
        data,
        speak_role=listen,
        text=line,
        step=f"open_floor_{trigger}",
    )


async def host_introduce_speaker(
    session: AgentSession,
    data: TalkShowData,
    controller: TurnController,
    role: str,
    *,
    trigger: str,
) -> None:
    """Host verbally grants the floor before a panelist speaks."""
    listen = controller.listen_role()
    name = load_persona_name(role)
    line = host_intro_speaker_line(name)
    append_role(data, listen, line)
    await speak_panel_line(
        session,
        data,
        speak_role=listen,
        text=line,
        step=f"intro_{role}_{trigger}",
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


async def grant_human_floor(
    session: AgentSession,
    data: TalkShowData,
    controller: TurnController,
    *,
    reason: str,
) -> None:
    """End panel beat — human speaks via mic."""
    listen = controller.listen_role()
    line = host_human_floor_line(topic=data.human_hand_topic or None)
    append_role(data, listen, line)
    await speak_panel_line(
        session,
        data,
        speak_role=listen,
        text=line,
        step="grant_human",
    )
    apply_floor_next(data, "human")
    await emit_floor_grant("human", reason=reason)
    await dequeue_hand_raise(data, "human")
    logger.info(
        "granted floor to human reason=%s queue=%s",
        reason,
        data.hand_raise_queue.roles(),
    )
    if data.turn_log:
        data.turn_log.log(
            "floor_grant_human",
            reason=reason,
            room=data.room_name,
            queue_remaining=data.hand_raise_queue.roles(),
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
    """
    listen = controller.listen_role()
    panel_roles = controller.panel_speaker_roles()
    max_turns = int(os.environ.get("TALKSHOW_PANEL_MAX_TURNS", "12"))
    spoken_roles: set[str] = set()
    turn_idx = 0
    close_round = False

    while turn_idx < max_turns and not close_round:
        while True:
            next_role = consume_floor_next(data)
            if is_direct_next(next_role):
                logger.info("floor direct grant role=%s", next_role)
                await dequeue_hand_raise(data, next_role)
                await emit_floor_grant(next_role, reason="next_tag")
                await speak_one_panelist(
                    session,
                    data,
                    speak_role=next_role,
                    step=f"speech_{next_role}_{turn_idx}",
                )
                spoken_roles.add(next_role)
                turn_idx += 1
                continue
            if next_role == "human":
                await grant_human_floor(
                    session, data, controller, reason="next_tag"
                )
                return
            if next_role == "close":
                close_round = True
                break
            apply_floor_next(data, next_role or "host")
            break

        if close_round:
            break

        outcome = await run_host_moderation_from_queue(
            session,
            data,
            controller,
            trigger=f"turn_{turn_idx}",
            spoken_roles=spoken_roles,
        )
        if outcome == "human":
            return
        if outcome == "close":
            logger.info("host_moderated: close round after %d panel turns", turn_idx)
            break
        if is_direct_next(outcome, data.scenario):
            spoken_roles.add(outcome)
        turn_idx += 1

    append_role(data, listen, PANEL_HOST_CLOSE)
    await speak_panel_line(
        session,
        data,
        speak_role=listen,
        text=PANEL_HOST_CLOSE,
        step="close_round",
    )
