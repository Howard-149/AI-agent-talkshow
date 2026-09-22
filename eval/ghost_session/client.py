"""Join a LiveKit room as a silent human and inject scripted turns."""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import time
from dataclasses import dataclass, field
from datetime import timedelta
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

# Protocol topics — keep in sync with agent.ui.control_events / ui_events.
CONTROL_TOPIC = "talkshow/control"
UI_TOPIC = "talkshow/ui"

logger = logging.getLogger("ghost_session")

_REPO_ROOT = Path(__file__).resolve().parents[2]


@dataclass
class ScriptedTurn:
    text: str
    topic: str = ""


@dataclass
class GhostScript:
    locale: str = "en"
    identity: str = "ghost-guest"
    turns: list[ScriptedTurn] = field(default_factory=list)


def load_script(path: Path) -> GhostScript:
    import yaml

    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(raw, dict):
        raise ValueError(f"ghost script must be a mapping: {path}")
    turns: list[ScriptedTurn] = []
    for item in raw.get("turns") or []:
        if isinstance(item, str):
            text = item.strip()
            topic = ""
        elif isinstance(item, dict):
            text = str(item.get("text") or "").strip()
            topic = str(item.get("topic") or "").strip()
        else:
            continue
        if text:
            turns.append(ScriptedTurn(text=text, topic=topic))
    return GhostScript(
        locale=str(raw.get("locale") or "en").strip() or "en",
        identity=str(raw.get("identity") or "ghost-guest").strip() or "ghost-guest",
        turns=turns,
    )


def mint_ghost_token(
    *,
    room: str,
    identity: str,
    locale: str,
    ttl_sec: int = 3600,
    viewer: bool = False,
    can_publish: bool | None = None,
) -> tuple[str, str]:
    url = os.environ["LIVEKIT_URL"]
    api_key = os.environ["LIVEKIT_API_KEY"]
    api_secret = os.environ["LIVEKIT_API_SECRET"]
    from livekit import api

    if viewer:
        if not identity.startswith("ghost-viewer"):
            identity = "ghost-viewer"
    elif not identity.startswith("ghost-"):
        identity = f"ghost-{identity}"
    meta: dict[str, Any] = {"locale": locale, "ghost": True}
    if viewer:
        meta["ghostViewer"] = True
    if can_publish is None:
        can_publish = not viewer
    token = (
        api.AccessToken(api_key, api_secret)
        .with_identity(identity)
        .with_name(identity)
        .with_metadata(json.dumps(meta, ensure_ascii=False))
        .with_grants(
            api.VideoGrants(
                room_join=True,
                room=room,
                can_publish=can_publish,
                can_subscribe=True,
            )
        )
        .with_ttl(timedelta(seconds=ttl_sec))
        .to_jwt()
    )
    return url, token


def _decode_ui_event(raw: bytes | str | dict[str, Any]) -> dict[str, Any] | None:
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, bytes):
        try:
            text = raw.decode("utf-8")
        except UnicodeDecodeError:
            return None
    else:
        text = raw
    try:
        obj = json.loads(text)
    except json.JSONDecodeError:
        return None
    return obj if isinstance(obj, dict) else None


class GhostRoomClient:
    def __init__(self, audio_tap: Any | None = None) -> None:
        self.room: Any = None
        self.welcome = asyncio.Event()
        self.human_floor = asyncio.Event()
        self._ui_types: list[str] = []
        self._audio_tap = audio_tap

    async def connect(self, url: str, token: str) -> None:
        from livekit import rtc

        self.room = rtc.Room()

        def _on_data(*args: Any, **kwargs: Any) -> None:
            packet = args[0] if args else kwargs.get("ev")
            topic = getattr(packet, "topic", None)
            payload = getattr(packet, "data", None)
            if topic is None and len(args) >= 4:
                payload, _participant, _kind, topic = args[:4]
            if topic != UI_TOPIC or payload is None:
                return
            ev = _decode_ui_event(payload)
            if ev is None:
                return
            ev_type = str(ev.get("type") or "")
            self._ui_types.append(ev_type)
            if ev_type == "transcript" and ev.get("step") == "session_welcome":
                self.welcome.set()
                logger.info("welcome transcript received")
            if ev_type == "floor_grant" and ev.get("role") == "human":
                self.human_floor.set()
                logger.info("human floor granted reason=%s", ev.get("reason"))
            if ev_type == "transcript":
                speaker = ev.get("speaker") or ev.get("role")
                text = (ev.get("text") or "")[:80]
                logger.info("ui transcript %s step=%s %r", speaker, ev.get("step"), text)

        self.room.on("data_received", _on_data)
        if self._audio_tap is not None:
            self._audio_tap.attach(self.room)
        await self.room.connect(url, token)
        if self._audio_tap is not None:
            self._audio_tap.capture_existing(self.room)
        logger.info(
            "ghost joined room=%s identity=%s",
            getattr(self.room, "name", "?"),
            getattr(self.room.local_participant, "identity", "?"),
        )

    async def publish_control(self, payload: dict[str, Any]) -> None:
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        await self.room.local_participant.publish_data(
            data,
            reliable=True,
            topic=CONTROL_TOPIC,
        )

    async def raise_hand(self, *, topic: str, reason: str = "ghost_script") -> None:
        await self.publish_control(
            {
                "type": "hand_raise",
                "raised": True,
                "topic": topic,
                "reason": reason,
            }
        )
        logger.info("hand raised topic=%r", topic)

    async def inject_turn(self, text: str) -> None:
        await self.publish_control({"type": "ghost_human_turn", "text": text})
        logger.info("injected ghost turn chars=%d %r", len(text), text[:80])

    async def disconnect(self) -> None:
        if self.room is None:
            return
        try:
            await self.room.disconnect()
        except Exception:
            logger.debug("ghost disconnect raised", exc_info=True)


async def _wait_human_floor(
    client: GhostRoomClient,
    *,
    topic: str,
    timeout_sec: float,
    raise_every_sec: float,
) -> bool:
    """Wait for ``floor_grant`` human; optionally re-raise so late control works."""
    if client.human_floor.is_set():
        return True
    loop = asyncio.get_running_loop()
    deadline = loop.time() + max(0.0, timeout_sec)
    while True:
        remaining = deadline - loop.time()
        if remaining <= 0:
            return False
        slice_sec = remaining
        if raise_every_sec > 0:
            slice_sec = min(slice_sec, raise_every_sec)
        try:
            await asyncio.wait_for(client.human_floor.wait(), timeout=slice_sec)
            return True
        except asyncio.TimeoutError:
            if raise_every_sec > 0 and not client.human_floor.is_set():
                await client.raise_hand(topic=topic)


async def run_ghost_session(
    *,
    room: str,
    script: GhostScript,
    opening_only: bool = False,
    welcome_timeout_sec: float = 90.0,
    floor_timeout_sec: float = 180.0,
    hold_sec: float = 8.0,
    record: bool = False,
    frontend_url: str = "http://localhost:3000",
    start_frontend: bool = False,
    record_dir: Path | None = None,
    headed: bool = False,
) -> Path | None:
    from eval.ghost_session.record_frontend import (
        FrontendRecorder,
        RoomAudioTap,
        default_record_dir,
        mux_recording,
        viewer_page_url,
    )

    url, token = mint_ghost_token(
        room=room,
        identity=script.identity,
        locale=script.locale,
    )
    stamp = int(time.time())
    out_dir = record_dir or default_record_dir()
    recorder: FrontendRecorder | None = None
    audio_tap: RoomAudioTap | None = None
    recorded: Path | None = None
    if record:
        out_dir.mkdir(parents=True, exist_ok=True)
        audio_tap = RoomAudioTap(out_dir / f"ghost-{room}-{stamp}.wav")
        recorder = FrontendRecorder(
            frontend_url=frontend_url,
            record_dir=out_dir,
            start_frontend=start_frontend,
            headed=headed,
        )
        _viewer_url, viewer_token = mint_ghost_token(
            room=room,
            identity="ghost-viewer",
            locale=script.locale,
            viewer=True,
        )
        page_url = viewer_page_url(
            frontend_url=frontend_url,
            livekit_url=url,
            token=viewer_token,
            locale=script.locale,
        )
    else:
        page_url = ""

    client = GhostRoomClient(audio_tap=audio_tap)
    await client.connect(url, token)
    try:
        first_topic = script.turns[0].topic if script.turns else "ghost opening"
        await client.raise_hand(topic=first_topic)
        if recorder is not None:
            # Guest is already in the room so opening waits on a real raise, not the viewer.
            await recorder.start(page_url)
        try:
            await asyncio.wait_for(client.welcome.wait(), timeout=welcome_timeout_sec)
        except asyncio.TimeoutError:
            logger.warning(
                "no session_welcome UI event after %.0fs — continuing",
                welcome_timeout_sec,
            )
        # Control channel is registered during bootstrap; raise again after welcome
        # so an early packet is not dropped before the agent is listening.
        if not client.human_floor.is_set():
            await client.raise_hand(topic=first_topic)

        if opening_only or not script.turns:
            await _wait_human_floor(
                client,
                topic=first_topic,
                timeout_sec=floor_timeout_sec,
                raise_every_sec=8.0,
            )
            await asyncio.sleep(max(0.0, hold_sec))
        else:
            for idx, turn in enumerate(script.turns, start=1):
                topic = turn.topic or first_topic or f"ghost turn {idx}"
                if idx > 1:
                    client.human_floor.clear()
                    await client.raise_hand(topic=topic)
                granted = await _wait_human_floor(
                    client,
                    topic=topic,
                    timeout_sec=floor_timeout_sec,
                    raise_every_sec=8.0,
                )
                if not granted:
                    raise TimeoutError(
                        f"timed out waiting for human floor before turn {idx}"
                    )
                client.human_floor.clear()
                await client.inject_turn(turn.text)
                granted = await _wait_human_floor(
                    client,
                    topic=topic,
                    timeout_sec=floor_timeout_sec,
                    raise_every_sec=0.0,
                )
                if not granted:
                    logger.warning(
                        "turn %d: no next human floor grant after %.0fs",
                        idx,
                        floor_timeout_sec,
                    )
            await asyncio.sleep(max(0.0, hold_sec))
    finally:
        wav: Path | None = None
        video: Path | None = None
        if recorder is not None:
            # Close the page while the room is still up so wav/video stop together.
            video = await recorder.stop()
        await client.disconnect()
        if audio_tap is not None:
            wav = await audio_tap.finish()
        if recorder is not None and video is not None:
            dest = out_dir / f"ghost-{room}-{stamp}.mp4"
            recorded = mux_recording(
                video=video,
                audio=wav,
                dest=dest,
                audio_t0=getattr(audio_tap, "started_at", None),
                video_t0=recorder.started_at,
            )
            logger.info("ghost recording → %s", recorded)
    return recorded


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Join LiveKit as a silent human, inject scripted turns, "
            "and let TTS / avatar run on the Babel worker."
        )
    )
    parser.add_argument(
        "--room",
        default=os.environ.get("TALKSHOW_ROOM", "talkshow-dev"),
        help="LiveKit room (must match the running agent worker)",
    )
    parser.add_argument(
        "--script",
        type=Path,
        default=_REPO_ROOT / "eval" / "ghost_session" / "scripts" / "smoke.yaml",
        help="YAML with locale + turns",
    )
    parser.add_argument(
        "--opening-only",
        action="store_true",
        help="Join + raise hand + wait for welcome / grant; do not inject text",
    )
    parser.add_argument("--locale", default="", help="Override script locale")
    parser.add_argument(
        "--welcome-timeout",
        type=float,
        default=90.0,
        help="Seconds to wait for session_welcome",
    )
    parser.add_argument(
        "--floor-timeout",
        type=float,
        default=180.0,
        help="Seconds to wait for each human floor grant",
    )
    parser.add_argument(
        "--hold-sec",
        type=float,
        default=8.0,
        help="Stay connected after the last beat so TTS can finish",
    )
    parser.add_argument(
        "--log-dir",
        type=Path,
        default=None,
        help="If set (on Babel), summarize the newest matching session JSONL",
    )
    parser.add_argument(
        "--summarize",
        type=Path,
        default=None,
        help="Only print a recap of this JSONL (no LiveKit join)",
    )
    parser.add_argument(
        "--record",
        action="store_true",
        help="Headless-record talkshow-web (panel + transcript) and mux show audio",
    )
    parser.add_argument(
        "--frontend-url",
        default=os.environ.get("TALKSHOW_FRONTEND_URL", "http://localhost:3000"),
        help="talkshow-web origin for --record",
    )
    parser.add_argument(
        "--start-frontend",
        action="store_true",
        help="If --record and the origin is down, run `pnpm dev` in talkshow-web/",
    )
    parser.add_argument(
        "--record-dir",
        type=Path,
        default=None,
        help="Where to write ghost-*.mp4 (default logs/ghost-recordings)",
    )
    parser.add_argument(
        "--headed",
        action="store_true",
        help="Show the Chromium window while recording (debug)",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    load_dotenv(_REPO_ROOT / ".env")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    args = _build_parser().parse_args(argv)
    if args.summarize is not None:
        from eval.ghost_session.summarize import summarize_session

        print(summarize_session(args.summarize))
        return 0

    script = load_script(args.script)
    if args.locale:
        script.locale = args.locale
    logger.info(
        "ghost session room=%s locale=%s turns=%d opening_only=%s record=%s",
        args.room,
        script.locale,
        0 if args.opening_only else len(script.turns),
        args.opening_only,
        args.record,
    )
    recorded = asyncio.run(
        run_ghost_session(
            room=args.room,
            script=script,
            opening_only=args.opening_only,
            welcome_timeout_sec=args.welcome_timeout,
            floor_timeout_sec=args.floor_timeout,
            hold_sec=args.hold_sec,
            record=args.record,
            frontend_url=args.frontend_url,
            start_frontend=args.start_frontend,
            record_dir=args.record_dir,
            headed=args.headed,
        )
    )
    if recorded is not None:
        print(f"RECORDING={recorded}")
    log_dir = args.log_dir
    if log_dir is None:
        default_dir = Path(os.environ.get("LOG_DIR", "logs"))
        if default_dir.is_dir():
            log_dir = default_dir
    if log_dir is not None:
        from eval.ghost_session.summarize import find_session_log, summarize_session

        path = find_session_log(log_dir, args.room)
        if path is None:
            logger.warning("no session JSONL for room=%s in %s", args.room, log_dir)
        else:
            print(summarize_session(path))
            from eval.ghost_session.judge_pad import format_pad_judgment, judge_pad_events
            from eval.log_parse import load_events

            print(format_pad_judgment(judge_pad_events(load_events(path))))
    else:
        logger.info(
            "Session JSONL is on the agent worker (LOG_DIR, default logs/). "
            "On Babel: python -m eval.ghost_session --summarize logs/session-<id>.jsonl"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
