from __future__ import annotations

import asyncio
import json
import logging
import os
from pathlib import Path

from agent.config import load_config, load_scenario
from agent.data import TalkShowData
from agent.floor import TurnController
from agent.floor.floor_control import apply_floor_next
from agent.floor.host_floor import host_speak_opening
from agent.interaction import ConsoleSpeechConsumer, ThinkActionInteraction
from agent.panel.panel_speech import run_panel_round
from agent.session.talkshow_runtime import build_runtime


async def run_text_opening(
    data: TalkShowData,
    controller: TurnController,
    human_inbox: asyncio.Queue[str] | None = None,
) -> None:
    """Run the host-moderated show in the console.

    Agents talk on their own; a line put on ``human_inbox`` (the ``human`` control
    command) interrupts the current speaker and the agents react to it. Once the
    host has closed the discussion, the show waits for the next human line.
    """
    from agent.show_graph.runner import run_show_beat

    human_inbox = human_inbox or asyncio.Queue()
    await host_speak_opening(None, data, controller)
    apply_floor_next(data, "host")

    interaction = data.floor_interaction
    while True:
        if human_inbox.empty() and not interaction.discussion_complete:
            await run_panel_round(None, data, controller)
            continue
        heard = (await human_inbox.get()).strip()
        if not heard:
            interaction.human_turn_abandoned()
            continue
        await run_show_beat(
            None,
            data,
            controller,
            entry="human",
            trigger="after_human",
            human_event={"heard": heard, "parsed": None, "done_event": "text_human_done"},
        )


class TextSessionManager:
    def __init__(self) -> None:
        job_id = os.environ.get("SLURM_JOB_ID", "local")

        self.control_dir = Path(
            os.environ.get(
                "TALKSHOW_TEXT_CONTROL_DIR",
                f"logs/text-control/{job_id}",
            )
        )

        self.control_dir.mkdir(parents=True, exist_ok=True)

        self.command_path = self.control_dir / "command.json"
        self.status_path = self.control_dir / "status.json"

        self.session_task: asyncio.Task | None = None
        self.session_id = 0
        self.interaction: ThinkActionInteraction | None = None
        self.data: TalkShowData | None = None
        self.human_inbox: asyncio.Queue[str] = asyncio.Queue()

    def write_status(self, state: str) -> None:
        self.status_path.write_text(
            json.dumps(
                {
                    "state": state,
                    "session_id": self.session_id,
                },
                ensure_ascii=False,
                indent=2,
            )
        )

    async def start_session(self) -> None:
        if self.session_task and not self.session_task.done():
            print("text session already running", flush=True)
            return

        self.session_id += 1

        self.session_task = asyncio.create_task(
            self.run_session(self.session_id)
        )

    async def run_session(self, session_id: int) -> None:
        self.write_status("running")

        print(
            f"\n=== text session {session_id} start ===",
            flush=True,
        )

        app_cfg = load_config()
        scenario = load_scenario()

        runtime = build_runtime(app_cfg)

        data = TalkShowData(
            scenario=scenario,
            runtime=runtime,
        )
        self.interaction = ThinkActionInteraction(ConsoleSpeechConsumer())
        self.human_inbox = asyncio.Queue()
        data.floor_interaction = self.interaction

        data.room_name = f"text-{session_id}"
        self.data = data

        controller = TurnController(
            scenario,
            data,
        )

        controller.apply_listen_persona()

        try:
            await run_text_opening(
                data,
                controller,
                self.human_inbox,
            )
        except asyncio.CancelledError:
            print(
                f"\n=== text session {session_id} stopped ===",
                flush=True,
            )
            raise
        except Exception:
            logging.getLogger(__name__).exception("text session %s failed", session_id)
            raise
        finally:
            await runtime.gemma_client.aclose()
            self.write_status("idle")

    async def stop_session(self) -> None:
        if not self.session_task:
            return

        if self.session_task.done():
            return

        self.session_task.cancel()

        try:
            await self.session_task
        except asyncio.CancelledError:
            pass

    def submit_human(self, data: TalkShowData | None, text: str) -> None:
        """A typed human line: take the floor now, the agents react once it is queued."""
        if self.interaction is None or not self.session_task or self.session_task.done():
            print("no text session running — human line ignored", flush=True)
            return
        if not text.strip():
            return
        self.interaction.human_turn_submitted(data)
        self.human_inbox.put_nowait(text)

    async def handle_command(self, command: str, payload: dict | None = None) -> None:
        if command == "human":
            self.submit_human(self.data, str((payload or {}).get("text") or ""))
            return
        if command == "start":
            await self.start_session()

        elif command == "stop":
            await self.stop_session()

        elif command == "restart":
            await self.stop_session()
            await self.start_session()

    async def run(self) -> None:
        self.write_status("idle")

        print(
            f"text session manager ready: {self.control_dir}",
            flush=True,
        )

        while True:
            if self.command_path.exists():
                try:
                    payload = json.loads(
                        self.command_path.read_text()
                    )
                    self.command_path.unlink()

                    await self.handle_command(
                        payload["command"], payload
                    )
                except Exception as exc:
                    print(
                        f"text control error: {exc}",
                        flush=True,
                    )

            await asyncio.sleep(0.5)


async def main() -> None:
    manager = TextSessionManager()
    await manager.run()