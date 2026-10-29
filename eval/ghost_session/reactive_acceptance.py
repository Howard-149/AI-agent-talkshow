"""Real DeepSeek + LiveKit + PCM acceptance. No prerecorded AI decisions.

Creates isolated local test rooms and records actual received audio. Reported
stop_ack_ms measures server speech-handle completion, not browser acoustic delay.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
from pathlib import Path
import time
import wave

from dotenv import load_dotenv
from eval.ghost_session.client import GhostRoomClient, UI_TOPIC, _decode_ui_event, _wait_human_floor, mint_ghost_token
from eval.ghost_session.record_frontend import RoomAudioTap


async def trial(mode: str, index: int, output: Path) -> dict:
    output.mkdir(parents=True, exist_ok=True)
    room = f"gefei-reactive-{mode}-{int(time.time())}-{index}"
    wav_path = output / f"{room}.wav"
    tap = RoomAudioTap(wav_path)
    client = GhostRoomClient(audio_tap=tap)
    events = []
    complete, heard, host_reply = asyncio.Event(), asyncio.Event(), asyncio.Event()
    sent_human = False
    human_mode = mode in {"human", "human_late", "human_gap"}
    ai_only = mode == "ai_only"
    ready = asyncio.Event()

    url, token = mint_ghost_token(room=room, identity="gefei-acceptance", locale="en",
                                 ai_only=ai_only, viewer=ai_only)
    await client.connect(url, token)

    def on_data(packet):
        if packet.topic != UI_TOPIC:
            return
        event = _decode_ui_event(packet.data)
        if not event:
            return
        events.append({"received_at": time.time(), **event})
        if event.get("type") == "reactive_ready" and event.get("enabled"):
            ready.set()
        if event.get("type") == "reactive_complete": complete.set()
        if event.get("type") == "reactive_state" and event.get("role") == "guest" and event.get("heard") and event.get("active"):
            if mode == "human" or (mode == "human_late" and event.get("step") == "demo_amy_response"):
                heard.set()
        if mode == "human_gap" and event.get("type") == "reactive_generating" and event.get("step") == "demo_host_followup":
            heard.set()
        if event.get("type") == "transcript" and event.get("step") == "host_reply":
            host_reply.set()
        if event.get("type") in {"reactive_decision", "reactive_complete", "reactive_error", "transcript"}:
            print(json.dumps({"room": room, **event}), flush=True)

    client.room.on("data_received", on_data)
    try:
        if ai_only:
            await asyncio.wait_for(ready.wait(), 60)
        else:
            await client.raise_hand(topic="AI homework")
            await asyncio.wait_for(client.welcome.wait(), 60)
            assert await _wait_human_floor(client, topic="AI homework", timeout_sec=90, raise_every_sec=5)
        client.human_floor.clear()
        await client.publish_control({"type": "reactive_demo", "mode": "live" if human_mode or ai_only else mode})
        if human_mode:
            await asyncio.wait_for(heard.wait(), 150)
            if mode != "human_gap": await asyncio.sleep(0.3)
            await client.publish_control({"type": "human_interrupt"})
        await asyncio.wait_for(complete.wait(), 240)
        if not ai_only:
            assert await _wait_human_floor(client, topic="AI homework", timeout_sec=45, raise_every_sec=0)
        if human_mode:
            # Prove that Send is accepted after the cut and the host actually replies.
            client.human_floor.clear()
            await client.inject_turn("I think students should try solving the problem themselves first, then ask AI for hints.")
            sent_human = True
            await asyncio.wait_for(host_reply.wait(), 90)
            # Bound this test: stop the following panel speaker rather than leave
            # background model work running after the observer disconnects.
            for _ in range(200):
                current = next((e for e in reversed(events) if e.get("type") == "reactive_state"), {})
                if current.get("active") and current.get("role") != "host":
                    await client.publish_control({"type": "human_interrupt"})
                    break
                await asyncio.sleep(0.1)
            assert await _wait_human_floor(client, topic="AI homework", timeout_sec=45, raise_every_sec=0)
        await asyncio.sleep(2)
    finally:
        await tap.finish()
        await client.disconnect()

    with wave.open(str(wav_path), "rb") as wav:
        duration = wav.getnframes() / wav.getframerate()
        pcm = wav.readframes(wav.getnframes())
    import numpy as np
    peak = int(np.abs(np.frombuffer(pcm, dtype=np.int16).astype(np.int32)).max()) if pcm else 0
    assert peak > 100 and duration > 5, "received audio is silent or missing"
    errors = [e for e in events if e.get("type") == "reactive_error"]
    assert not errors, errors
    decisions = [e for e in events if e.get("type") == "reactive_decision"]
    grants = [e for e in decisions if e.get("status") == "granted"]
    if mode in {"live", "ai_only"}: assert any(e.get("source") == "llm" for e in decisions), "missing real AI intent decision"
    if mode == "queue":
        assert any(e.get("status") == "queued" for e in decisions), "no AI request queued"
        assert not any(e.get("source") == "llm" for e in grants), "queue-only policy allowed a cut"
    if mode == "baseline": assert not decisions, "baseline made intent decisions"
    if human_mode:
        assert any(e.get("source") == "human" for e in grants), "human failed to take priority"
        assert sent_human and host_reply.is_set(), "host did not respond to post-cut input"

    # Read only this room's server telemetry; transcript cap never substitutes
    # for an epoch. The demo-end history is the actual committed shared memory.
    rows = []
    for path in Path("logs").glob("session-*.jsonl"):
        for raw in path.read_text().splitlines():
            try: row = json.loads(raw)
            except json.JSONDecodeError: continue
            if row.get("room") == room: rows.append(row)
    cut_lines = [r for r in rows if r.get("event") == "reactive_line_end" and r.get("interrupted")]
    starts = [r for r in rows if r.get("event") == "reactive_line_start"]
    for cut in cut_lines:
        start = next(r for r in starts if r["epoch"] == cut["epoch"])
        assert cut["heard"] == "" or start["text"].startswith(cut["heard"]), "committed non-prefix text"
        assert cut["discarded_chars"] > 0, "nothing was truncated"
    ends = [r for r in rows if r.get("event") == "reactive_demo_end"]
    assert ends, "missing completed demo telemetry"
    if ai_only:
        assert all(item["role"] != "human" for item in ends[0]["history"]), "fabricated human in AI-only history"
        assert not any(e.get("type") == "floor_grant" and e.get("role") == "human" for e in events), "AI-only returned floor to human"
        assert not any(e.get("step") in {"grant_human", "session_welcome"} for e in events), "human greeting played in AI-only room"
    demo_end_time = next(e["received_at"] for e in events if e.get("type") == "reactive_complete")
    demo_start_time = next(e["received_at"] for e in events if e.get("type") == "reactive_reset")
    demo_speeches = [e for e in events if e.get("type") == "transcript" and e.get("role") in {"guest", "commentator", "host"}
                    and demo_start_time <= e["received_at"] <= demo_end_time and (str(e.get("step", "")).startswith("demo_")
                                                             or str(e.get("step", "")).startswith("speech_")
                                                             or str(e.get("step", "")).startswith("reactive_takeover_"))]
    if not human_mode:
        assert [e["role"] for e in demo_speeches] == ["guest", "commentator", "guest", "host", "commentator", "guest", "host"], demo_speeches
        assert demo_speeches[-1]["step"] == "demo_host_wrap", "missing contextual host ending"
        assert not any(e.get("step") == "grant_human" and demo_start_time <= e["received_at"] <= demo_end_time for e in events), "extra canned handoff after wrap"
    if mode in {"human_late", "human_gap"}:
        assert any(e.get("step") == "demo_amy_response" for e in demo_speeches), "did not reach Amy's reply"
        assert not any(e.get("step") == "demo_host_wrap" for e in demo_speeches), "continued after human stop"
    for cut in cut_lines:
        if cut.get("omitted_fragment"):
            assert all(cut["omitted_fragment"] not in item["text"] for item in ends[0]["history"]), "unplayed tail leaked into history"
    result = {"room": room, "mode": mode, "passed": True, "audio_seconds": round(duration, 2),
              "audio_peak": peak, "audio": str(wav_path), "decisions": len(decisions),
              "grants": [{"role": e["role"], "source": e.get("source"), "rule": e.get("rule")} for e in grants],
              "stop_ack_ms": [r["stop_ack_ms"] for r in cut_lines],
              "intent_latency_ms": [r["latency_ms"] for r in rows if r.get("event") == "reactive_model_done"],
              "history_tail_leak": False, "ai_interrupted": any(e.get("source") == "llm" for e in grants),
              "human_reply_verified": human_mode and host_reply.is_set(),
              "demo_speeches": [{"role": e["role"], "step": e.get("step"), "text": e.get("text")} for e in demo_speeches],
              "natural_wrap_verified": not human_mode}
    (output / f"{room}.events.json").write_text(json.dumps(events, indent=2))
    return result


async def main(args):
    load_dotenv(Path(__file__).resolve().parents[2] / ".env")
    results = []
    for index, mode in enumerate(args.modes):
        result = await trial(mode, index, args.output)
        results.append(result)
        print("ACCEPTANCE " + json.dumps(result), flush=True)
        (args.output / "summary.json").write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--modes", nargs="+", choices=["live", "queue", "baseline", "human", "human_late", "human_gap", "ai_only"],
                        default=["live", "live", "queue", "baseline", "human"])
    parser.add_argument("--output", type=Path, default=Path("logs/reactive-acceptance"))
    logging.basicConfig(level=logging.WARNING)
    asyncio.run(main(parser.parse_args()))
