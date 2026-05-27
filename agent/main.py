from __future__ import annotations

import logging
import os
from pathlib import Path

from dotenv import load_dotenv
from livekit.agents import AgentServer, JobContext, JobProcess, cli, room_io
from livekit.plugins import silero

from agent.agents.host import HostAgent
from agent.config import load_config, load_persona_instructions
from agent.hooks.logging import TurnJsonlLogger
from agent.session import build_agent_session, build_runtime

logger = logging.getLogger("talkshow")
logging.basicConfig(level=logging.INFO)

load_dotenv()

server = AgentServer()


def prewarm(proc: JobProcess) -> None:
    proc.userdata["vad"] = silero.VAD.load()


server.setup_fnc = prewarm


@server.rtc_session()
async def entrypoint(ctx: JobContext) -> None:
    ctx.log_context_fields = {"room": ctx.room.name}

    runtime = build_runtime(load_config())
    session = build_agent_session(runtime)
    persona = load_persona_instructions("host")

    log_dir = Path(os.environ.get("LOG_DIR", "logs"))
    turn_log = TurnJsonlLogger(log_dir)

    @session.on("user_input_transcribed")
    def _on_user_transcript(ev) -> None:  # type: ignore[no-untyped-def]
        if ev.is_final:
            turn_log.log("user_heard", text=ev.transcript, room=ctx.room.name)

    @session.on("conversation_item_added")
    def _on_item(ev) -> None:  # type: ignore[no-untyped-def]
        item = ev.item
        if item.role == "assistant" and item.text_content:
            turn_log.log("assistant_reply", text=item.text_content, room=ctx.room.name)

    await session.start(
        agent=HostAgent(instructions=persona),
        room=ctx.room,
        room_options=room_io.RoomOptions(),
    )


if __name__ == "__main__":
    # Run from repo root: python -m agent.main dev
    cli.run_app(server)
