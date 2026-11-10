"""Panelist drafts for the show graph, and the entry to a panel round.

Thinking (prompt → Gemma → parsed reply) lives in agent.multi_agent.PersonaWorker;
the show graph records, appraises and speaks the line.
"""

from __future__ import annotations

import logging
import os

from livekit.agents import AgentSession

from agent.data import TalkShowData
from agent.emotion import set_role_emotion
from agent.multi_agent.types import AgentDraft
from agent.show_graph.state import PanelistLine
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


def draft_ahead_enabled() -> bool:
    """Draft the next panelist's line while the host's intro plays (opt-in)."""
    raw = os.environ.get("TALKSHOW_DRAFT_AHEAD", "").strip().lower()
    return raw in ("1", "true", "yes", "on")


def panelist_line_from_draft(data: TalkShowData, draft: AgentDraft) -> PanelistLine:
    """Bookkeeping for a draft that will be spoken: dialogue-example index, emotion tag."""
    if draft.dialogue_example_index is not None:
        data.dialogue_example_index = draft.dialogue_example_index
    emotion = set_role_emotion(data, draft.role, draft.emotion)
    return PanelistLine(
        role=draft.role, text=draft.speech, next_role=draft.next_role, emotion=emotion
    )


async def generate_panelist_line(
    session: AgentSession,
    data: TalkShowData,
    *,
    speak_role: str,
    step: str,
) -> PanelistLine:
    """Show graph: draft a line that *will* be spoken.

    The graph records, appraises and speaks it itself (prepare_speak), so only the
    draft-side bookkeeping is applied here.
    """
    draft = await draft_panelist(data, speak_role=speak_role, step=step)
    return panelist_line_from_draft(data, draft)


async def run_panel_round(
    session: AgentSession | None,
    data: TalkShowData,
    controller: TurnController,
) -> None:
    """Run the panel round (Think & Action floor control)."""
    from agent.show_graph import run_host_moderated_panel

    await run_host_moderated_panel(session, data, controller)
