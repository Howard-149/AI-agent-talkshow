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
from agent.session_lifecycle import register_session_lifecycle
from agent.supervisor import TurnController

logger = logging.getLogger("talkshow")
logging.basicConfig(level=logging.INFO)

# Keep ≥1 idle job process warm (Piper+VAD) after register — not only on first job.
# `dev` defaults to 0 idle processes otherwise.
_idle_processes = int(os.environ.get("TALKSHOW_IDLE_PROCESSES", "1"))
_init_timeout = float(os.environ.get("TALKSHOW_PROCESS_INIT_TIMEOUT", "180"))
server = AgentServer(
    num_idle_processes=max(_idle_processes, 0),
    initialize_process_timeout=max(_init_timeout, 30.0),
)
logger.info(
    "AgentServer idle_processes=%d init_timeout=%.0fs (Piper prewarm on idle spawn)",
    max(_idle_processes, 0),
    max(_init_timeout, 30.0),
)


def prewarm(proc: JobProcess) -> None:
    proc.userdata["vad"] = load_vad()
    try:
        from agent.adapters.piper_tts import prewarm_piper_voices

        prewarm_piper_voices()
    except Exception as exc:
        logger.warning("Piper prewarm failed: %s", exc)


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
    from agent.adapters.avatar_bridge import init_avatar_bridge

    init_avatar_bridge(turn_log=turn_log)
    dlg = scenario.dialogue
    turn_log.log(
        "session_start",
        room=ctx.room.name,
        log_file=turn_log.path.name,
        scenario_id=scenario.id,
        turn_mode=scenario.turn_control.mode,
        dialogue_library=dlg.library if dlg else None,
        dialogue_pick=dlg.pick if dlg else None,
    )
    controller = TurnController(scenario, data)
    data.ensure_panel_priority(
        scenario.turn_control.order,
        scenario.turn_control.listen_role,
    )

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

    def _is_speaking_state(state: object) -> bool:
        s = str(state).lower()
        return s == "speaking" or s.endswith(".speaking")

    register_session_lifecycle(session, data, ctx.room, job_ctx=ctx)

    @session.on("speech_created")
    def _on_speech_created(ev) -> None:  # type: ignore[no-untyped-def]
        """Emit role_idle after each playout so chunked lines idle between sentences."""
        handle = ev.speech_handle

        def _on_playout_done(_h) -> None:  # type: ignore[no-untyped-def]
            from agent.ui_events import emit_role_idle

            asyncio.create_task(emit_role_idle())

        handle.add_done_callback(_on_playout_done)

    @session.on("agent_state_changed")
    def _on_agent_state(ev) -> None:  # type: ignore[no-untyped-def]
        new_state = getattr(ev, "new_state", None)
        if _is_speaking_state(new_state):
            pending = data.pop_pending_speech_ui()
            if pending:
                role, text, step = pending
                asyncio.create_task(_emit_speech_ui(role, text, step=step))
        if not _is_listening_state(new_state):
            return
        if not controller.is_panel_mode():
            return
        if not data.panel_followup_pending or data.panel_chain_running:
            return
        # Avatar chunked lines idle between sentences — do not start panel mid-line.
        if data.speak_line_busy:
            data.panel_followup_deferred = True
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

    data.panel_followup_runner = lambda: asyncio.create_task(_run_panel_followups())

    async def _emit_speech_ui(role: str, text: str, *, step: str) -> None:
        from agent.ui_events import emit_role_active, emit_transcript

        await emit_role_active(role)
        if text.strip():
            await emit_transcript(role, text, step=step)

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
        room_input_options=room_io.RoomInputOptions(close_on_disconnect=True),
    )


if __name__ == "__main__":
    cli.run_app(server)
