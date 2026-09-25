"""Run a panel round: switch roles, generate lines, apply floor tags, speak.

Thinking (prompt → Gemma → parsed reply) lives in agent.multi_agent.PersonaWorker;
this module only owns what happens once a draft actually gets the floor.
"""

from __future__ import annotations

import logging

from livekit.agents import AgentSession

from agent.data import TalkShowData
from agent.emotion import set_role_emotion
from agent.floor.floor_control import apply_floor_next
from agent.multi_agent.types import AgentDraft
from agent.session.session_handoff import speak_panel_line, switch_to_role
from agent.show.show_history import append_role
from agent.floor import TurnController

logger = logging.getLogger(__name__)

PANEL_HOST_CLOSE = (
    "That wraps our panel on this round. Back to you — "
    "what would you like to add or ask?"
)


async def draft_panelist(
    data: TalkShowData,
    *,
    speak_role: str,
    step: str,
) -> AgentDraft:
    return await data.runtime.multi_agent.draft_role(
        speak_role, 
        data, 
        step=step
    )


async def speak_draft(
    session: AgentSession,
    data: TalkShowData,
    draft: AgentDraft,
    *,
    step: str,
) -> str:
    """Put an already-generated draft on the floor."""
    speak_role = draft.role
    if speak_role != data.active_role:
        await switch_to_role(session, data, speak_role, reason=f"panel:{step}")

    emotion = set_role_emotion(data, speak_role, draft.emotion)
    if draft.dialogue_example_index is not None:
        data.dialogue_example_index = draft.dialogue_example_index
    append_role(data, speak_role, draft.speech)
    await speak_panel_line(
        session,
        data,
        speak_role=speak_role,
        text=draft.speech,
        step=step,
    )
    apply_floor_next(data, draft.next_role)
    logger.info(
        "panelist spoke role=%s tagged next=%s emotion=%s",
        speak_role,
        draft.next_role,
        emotion,
    )
    return draft.next_role


async def speak_one_panelist(
    session: AgentSession,
    data: TalkShowData,
    *,
    speak_role: str,
    step: str,
) -> str:
    """Generate and speak one panelist line. Returns consumed [next] role for floor routing."""
    draft = await draft_panelist(data, speak_role=speak_role, step=step)
    return await speak_draft(session, data, draft, step=step)


async def run_panel_round(
    session: AgentSession,
    data: TalkShowData,
    controller: TurnController,
) -> None:
    """Dispatch fixed-order panel or host-moderated floor control."""
    if controller.is_host_moderated_mode():
        from agent.show_graph import run_host_moderated_panel

        await run_host_moderated_panel(session, data, controller)
        return

    listen = controller.listen_role()

    for role in controller.panel_followup_roles():
        await speak_one_panelist(
            session,
            data,
            speak_role=role,
            step=f"speech_{role}",
        )

    append_role(data, listen, PANEL_HOST_CLOSE)
    await speak_panel_line(
        session,
        data,
        speak_role=listen,
        text=PANEL_HOST_CLOSE,
        step="close_round",
    )
