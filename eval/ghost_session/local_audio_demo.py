"""Run the upstream demo script and verify real LiveKit audio, without Playwright."""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
from pathlib import Path
import wave

from dotenv import load_dotenv
from eval.ghost_session.client import (
    GhostRoomClient, UI_TOPIC, _decode_ui_event, _wait_human_floor,
    load_script, mint_ghost_token,
)
from eval.ghost_session.record_frontend import RoomAudioTap


async def run(args: argparse.Namespace) -> None:
    load_dotenv(Path(__file__).resolve().parents[2] / '.env')
    script = load_script(args.script)
    audio = RoomAudioTap(args.output)
    client = GhostRoomClient(audio_tap=audio)
    url, token = mint_ghost_token(
        room=args.room, identity=script.identity, locale=script.locale,
    )
    await client.connect(url, token)
    roles: set[str] = set()

    def on_data(packet) -> None:
        if packet.topic != UI_TOPIC:
            return
        event = _decode_ui_event(packet.data)
        if event and event.get('type') == 'transcript' and event.get('final'):
            role = event.get('role', '')
            roles.add(role)
            print(json.dumps({'role': role, 'step': event.get('step'),
                              'text': event.get('text')}, ensure_ascii=False), flush=True)

    client.room.on('data_received', on_data)
    try:
        await client.raise_hand(topic=script.turns[0].topic)
        await asyncio.wait_for(client.welcome.wait(), timeout=60)
        await client.raise_hand(topic=script.turns[0].topic)
        for turn in script.turns:
            if not await _wait_human_floor(client, topic=turn.topic,
                                          timeout_sec=args.floor_timeout,
                                          raise_every_sec=8):
                raise TimeoutError('No human floor before scripted input')
            client.human_floor.clear()
            await client.inject_turn(turn.text)
            if not await _wait_human_floor(client, topic=turn.topic,
                                          timeout_sec=args.floor_timeout,
                                          raise_every_sec=0):
                raise TimeoutError('Panel did not return the human floor')
        await asyncio.sleep(3)
    finally:
        await audio.finish()
        await client.disconnect()
    with wave.open(str(args.output), 'rb') as wav:
        duration = wav.getnframes() / wav.getframerate()
        pcm = wav.readframes(wav.getnframes())
    import numpy as np
    peak = int(np.abs(np.frombuffer(pcm, dtype=np.int16).astype(np.int32)).max())
    assert {'host', 'guest', 'commentator'} <= roles, f'Missing panel roles: {roles}'
    assert duration > 5 and peak > 100, 'Missing or silent room audio'
    print(json.dumps({'verified': True, 'room': args.room, 'roles': sorted(roles),
                      'audio_seconds': round(duration, 2), 'audio_peak': peak,
                      'output': str(args.output)}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--room', default='talkshow-gefei-audio-test')
    parser.add_argument('--script', type=Path,
                        default=Path('eval/ghost_session/scripts/pad_swing.yaml'))
    parser.add_argument('--output', type=Path, default=Path('logs/local-demo.wav'))
    parser.add_argument('--floor-timeout', type=float, default=240)
    logging.basicConfig(level=logging.INFO)
    asyncio.run(run(parser.parse_args()))
