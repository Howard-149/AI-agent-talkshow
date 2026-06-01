from __future__ import annotations

import asyncio
import logging
import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

from agent.bootstrap import ensure_hf_hub_env

ensure_hf_hub_env()

from livekit.agents import AgentServer, JobContext, JobProcess, cli, room_io
from livekit.plugins import silero

from agent.agents.factory import build_agent
from agent.config import load_config, load_scenario
from agent.data import TalkShowData
from agent.hooks.logging import TurnJsonlLogger
from agent.panel_speech import run_panel_round
from agent.session_handoff import switch_to_role
from agent.runtime import build_runtime
from agent.session import build_agent_session
from agent.supervisor import TurnController

logger = logging.getLogger("talkshow")
logging.basicConfig(level=logging.INFO)

server = AgentServer()


def prewarm(proc: JobProcess) -> None:
    proc.userdata["vad"] = silero.VAD.load()


server.setup_fnc = prewarm


@server.rtc_session()
async def entrypoint(ctx: JobContext) -> None:
    ctx.log_context_fields = {"room": ctx.room.name}

    app_cfg = load_config()
    scenario = load_scenario()
    runtime = build_runtime(app_cfg)
    data = TalkShowData(scenario=scenario, runtime=runtime)
    controller = TurnController(scenario, data)

    session = build_agent_session(runtime, data)
    log_dir = Path(os.environ.get("LOG_DIR", "logs"))
    turn_log = TurnJsonlLogger(log_dir)

    @session.on("user_input_transcribed")
    def _on_user_transcript(ev) -> None:  # type: ignore[no-untyped-def]
        if not ev.is_final:
            return
        if data.panel_chain_running:
            return
        if controller.is_panel_mode():
            data.user_turn_pending_panel = True
            listen = controller.listen_role()
            controller.apply_listen_persona()
            if data.active_role != listen:
                asyncio.create_task(
                    _apply_handoff(listen, reason="panel_human_turn", silent=True)
                )
        else:
            data.user_turn_pending_rotation = True
        turn_log.log(
            "user_heard",
            text=ev.transcript,
            room=ctx.room.name,
            active_role=data.active_role,
        )

    async def _apply_handoff(next_role: str, *, reason: str, silent: bool) -> None:
        entry_from = data.active_role
        if silent:
            await switch_to_role(session, data, next_role, reason=reason)
        else:
            controller.record_handoff(to_role=next_role, reason=reason)
            controller.apply_persona_for_role(next_role)
            from agent.participant_display import set_agent_display_name

            await set_agent_display_name(next_role)
            session.update_agent(build_agent(next_role, data))
            from agent.session_handoff import wait_for_session_agent

            await wait_for_session_agent(session)
        turn_log.log(
            "handoff",
            text=f"{entry_from} -> {next_role}",
            room=ctx.room.name,
            active_role=next_role,
            reason=reason,
        )

    async def _run_panel_followups() -> None:
        """After human spoke and host replied: Lessac announces → Ryan → Amy → Lessac closes."""
        if data.panel_chain_running:
            return
        data.panel_chain_running = True
        try:
            logger.info("PANEL round start — watch logs for step=bridge_* / speech_*")
            await run_panel_round(session, data, controller)
            listen = controller.listen_role()
            await _apply_handoff(listen, reason="panel_wait_human", silent=True)
            logger.info("PANEL round done — waiting for human")
        finally:
            data.panel_chain_running = False
            data.user_turn_pending_panel = False

    async def _maybe_rotate_after_reply() -> None:
        tagged = runtime.turn_store.consume_handoff_to()
        if tagged:
            await _apply_handoff(tagged, reason="gemma_handoff_tag", silent=True)
            return
        next_role = controller.advance_rotation_after_assistant()
        if not next_role:
            return
        await _apply_handoff(next_role, reason="rotate_after_user", silent=True)

    @session.on("conversation_item_added")
    def _on_item(ev) -> None:  # type: ignore[no-untyped-def]
        item = ev.item
        role = getattr(item, "role", None)
        text = getattr(item, "text_content", None)
        if role != "assistant" or not text:
            return
        turn_log.log(
            "assistant_reply",
            text=text,
            room=ctx.room.name,
            active_role=data.active_role,
        )

        if controller.is_panel_mode():
            if (
                data.user_turn_pending_panel
                and data.active_role == controller.listen_role()
                and not data.panel_chain_running
            ):
                data.user_turn_pending_panel = False
                asyncio.create_task(_run_panel_followups())
            return

        if not data.user_turn_pending_rotation:
            return
        data.user_turn_pending_rotation = False
        asyncio.create_task(_maybe_rotate_after_reply())

    await session.start(
        agent=build_agent("host", data),
        room=ctx.room,
        room_options=room_io.RoomOptions(),
    )


if __name__ == "__main__":
    cli.run_app(server)
