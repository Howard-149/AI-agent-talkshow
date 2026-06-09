from __future__ import annotations

import asyncio
import logging
import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

from agent.bootstrap import ensure_cluster_runtime_env

ensure_cluster_runtime_env()

from livekit.agents import AgentServer, JobContext, JobProcess, cli, room_io

from agent.agents.factory import build_agent
from agent.config import load_config, load_scenario
from agent.data import TalkShowData
from agent.hooks.logging import TurnJsonlLogger
from agent.panel_speech import run_panel_round
from agent.session_handoff import switch_to_role
from agent.runtime import build_runtime
from agent.session import build_agent_session, load_vad
from agent.supervisor import TurnController

logger = logging.getLogger("talkshow")
logging.basicConfig(level=logging.INFO)

server = AgentServer()


def prewarm(proc: JobProcess) -> None:
    proc.userdata["vad"] = load_vad()


server.setup_fnc = prewarm


@server.rtc_session()
async def entrypoint(ctx: JobContext) -> None:
    ctx.log_context_fields = {"room": ctx.room.name}

    app_cfg = load_config()
    scenario = load_scenario()
    runtime = build_runtime(app_cfg)
    data = TalkShowData(scenario=scenario, runtime=runtime)
    data.room_name = ctx.room.name
    log_dir = Path(os.environ.get("LOG_DIR", "logs"))
    turn_log = TurnJsonlLogger(log_dir)
    data.turn_log = turn_log
    controller = TurnController(scenario, data)

    session = build_agent_session(
        runtime, data, prewarmed_vad=ctx.proc.userdata.get("vad")
    )
    data.agent_session = session

    @session.on("user_input_transcribed")
    def _on_user_transcript(ev) -> None:  # type: ignore[no-untyped-def]
        if not ev.is_final:
            return
        transcript = (ev.transcript or "").strip()
        if not transcript:
            logger.debug("ignore empty user transcript")
            return
        if data.panel_chain_running:
            return
        data.touch_activity()
        if controller.is_panel_mode():
            # Handoff + panel_followup_pending happen in GemmaAudioSTT (serialized)
            pass
        else:
            data.user_turn_pending_rotation = True
        turn_log.log(
            "user_heard",
            text=transcript,
            room=ctx.room.name,
            active_role=data.active_role,
            panel_followup_pending=data.panel_followup_pending,
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
        """After human spoke and host replied: host-moderated or fixed panel round."""
        if data.panel_chain_running:
            turn_log.log("panel_skip", reason="chain_running", room=ctx.room.name)
            return
        data.panel_chain_running = True
        try:
            turn_log.log(
                "panel_start",
                floor_next=data.floor_next_speaker or "host",
                room=ctx.room.name,
                active_role=data.active_role,
            )
            logger.info("PANEL round start next=%s", data.floor_next_speaker or "host")
            await run_panel_round(session, data, controller)
            listen = controller.listen_role()
            await _apply_handoff(listen, reason="panel_wait_human", silent=True)
            turn_log.log("panel_done", room=ctx.room.name, active_role=data.active_role)
            logger.info("PANEL round done — waiting for human")
        finally:
            data.panel_chain_running = False
            data.user_turn_pending_panel = False
            data.panel_followup_pending = False

    async def _maybe_rotate_after_reply() -> None:
        tagged = runtime.turn_store.consume_handoff_to()
        if tagged:
            await _apply_handoff(tagged, reason="gemma_handoff_tag", silent=True)
            return
        next_role = controller.advance_rotation_after_assistant()
        if not next_role:
            return
        await _apply_handoff(next_role, reason="rotate_after_user", silent=True)

    def _is_listening_state(state: object) -> bool:
        s = str(state).lower()
        return s == "listening" or s.endswith(".listening")

    @session.on("agent_state_changed")
    def _on_agent_state(ev) -> None:  # type: ignore[no-untyped-def]
        new_state = getattr(ev, "new_state", None)
        if _is_listening_state(new_state):
            from agent.ui_events import emit_role_idle

            asyncio.create_task(emit_role_idle())
        if not _is_listening_state(new_state):
            return
        if not controller.is_panel_mode():
            return
        if not data.panel_followup_pending or data.panel_chain_running:
            return
        if data.active_role != controller.listen_role():
            turn_log.log(
                "panel_skip",
                reason="wrong_role",
                active_role=data.active_role,
                listen=controller.listen_role(),
                room=ctx.room.name,
            )
            return
        data.panel_followup_pending = False
        data.user_turn_pending_panel = False
        turn_log.log(
            "panel_trigger",
            reason="host_reply_done",
            floor_next=data.floor_next_speaker or "host",
            room=ctx.room.name,
        )
        asyncio.create_task(_run_panel_followups())

    @session.on("speech_created")
    def _on_speech_created(ev) -> None:  # type: ignore[no-untyped-def]
        """Sync transcript + speaker highlight with TTS playout start."""
        pending = data.pop_pending_transcript()
        if not pending:
            return
        role, text, step = pending
        asyncio.create_task(
            _emit_speech_ui(role, text, step=step, source=str(ev.source))
        )

    async def _emit_speech_ui(
        role: str, text: str, *, step: str, source: str
    ) -> None:
        from agent.ui_events import emit_role_active, emit_transcript

        await emit_role_active(role)
        if text.strip():
            await emit_transcript(role, text, step=step)
        logger.debug(
            "speech UI synced role=%s step=%s source=%s text_len=%d",
            role,
            step,
            source,
            len(text),
        )

    @session.on("conversation_item_added")
    def _on_item(ev) -> None:  # type: ignore[no-untyped-def]
        item = ev.item
        role = getattr(item, "role", None)
        text = getattr(item, "text_content", None)
        if role != "assistant" or not text:
            return
        data.touch_activity()
        turn_log.log(
            "assistant_reply",
            text=text,
            room=ctx.room.name,
            active_role=data.active_role,
        )

        if controller.is_panel_mode():
            return

        if not data.user_turn_pending_rotation:
            return
        data.user_turn_pending_rotation = False
        asyncio.create_task(_maybe_rotate_after_reply())

    async def _bootstrap() -> None:
        from agent.bootstrap_session import bootstrap_after_connect

        await bootstrap_after_connect(ctx, session, data, controller, scenario)

    asyncio.create_task(_bootstrap())

    await session.start(
        agent=build_agent("host", data),
        room=ctx.room,
        room_options=room_io.RoomOptions(),
    )


if __name__ == "__main__":
    cli.run_app(server)
