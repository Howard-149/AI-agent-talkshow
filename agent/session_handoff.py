from __future__ import annotations

import asyncio
import logging

from livekit.agents import AgentSession

from agent.agents.factory import build_agent
from agent.config import load_persona_tts
from agent.data import TalkShowData
from agent.participant_display import set_agent_display_name
from agent.supervisor import TurnController
logger = logging.getLogger(__name__)


async def wait_for_session_agent(session: AgentSession) -> None:
    """Wait until update_agent() has finished swapping AgentActivity."""
    task = getattr(session, "_update_activity_atask", None)
    if task is not None and not task.done():
        await asyncio.shield(task)


async def switch_to_role(
    session: AgentSession,
    data: TalkShowData,
    role: str,
    *,
    reason: str,
) -> None:
    """Hand off to role and wait before speaking (correct Piper + no stray on_enter)."""
    if role == data.active_role:
        return

    ctrl = TurnController(data.scenario, data)
    data.silent_handoff = True
    try:
        ctrl.record_handoff(to_role=role, reason=reason)
        ctrl.apply_persona_for_role(role)
        await set_agent_display_name(role)
        session.update_agent(build_agent(role, data))
        await wait_for_session_agent(session)
        tts_path = load_persona_tts(role, data.runtime.config).model_path
        logger.info("role ready role=%s piper=%s reason=%s", role, tts_path, reason)
    finally:
        data.silent_handoff = False


def queue_speech_ui(
    data: TalkShowData, role: str, text: str, *, step: str = ""
) -> None:
    """Queue UI update — emitted on speech_created when TTS is ready to play."""
    data.queue_transcript(role, text, step=step)


async def speak_panel_line(
    session: AgentSession,
    data: TalkShowData,
    *,
    speak_role: str,
    text: str,
    step: str,
) -> None:
    """TTS-only line: bypass StoredReplyLLM so panel text is never stolen or replaced."""
    from agent.floor_parser import strip_next_tag
    from agent.ui_events import emit_role_active, emit_transcript

    text = strip_next_tag(text.strip())
    if not text:
        return

    if speak_role != data.active_role:
        await switch_to_role(session, data, speak_role, reason=f"panel:{step}")
    await emit_role_active(speak_role)
    await emit_transcript(speak_role, text, step=step)
    logger.info("PANEL say step=%s role=%s text=%.80r", step, speak_role, text)
    handle = session.say(text, allow_interruptions=False)
    await handle.wait_for_playout()
