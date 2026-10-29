"""Synthesize and validate three real Fish role voices without exposing the key."""
import argparse
import asyncio
import json
from pathlib import Path
import time
import wave

from dotenv import load_dotenv
import numpy as np

from agent.adapters.fish_tts import FishTTS
from agent.config import load_config, load_persona_tts

TEXTS = {
    "host": "Welcome back to AI Talk Show. Today we're discussing whether students should use AI for homework. Amy, let's hear your view.",
    "guest": "I think AI can help students learn, but there is an important difference between asking for a hint and submitting an answer you do not understand.",
    "commentator": "Let me push back on that. If students never try solving the problem themselves, reading a correct answer is not the same as learning.",
}


async def main(output):
    load_dotenv(Path(__file__).resolve().parents[2] / ".env")
    output.mkdir(parents=True, exist_ok=True)
    config = load_config()
    results = []
    for role, text in TEXTS.items():
        adapter = FishTTS(load_persona_tts(role, config, locale="en"), role=role)
        started, first, chunks = time.monotonic(), None, []
        async with adapter.synthesize(text) as stream:
            async for event in stream:
                if first is None:
                    first = time.monotonic()
                chunks.append(bytes(event.frame.data))
        pcm = b"".join(chunks)
        samples = np.frombuffer(pcm, dtype=np.int16).astype(np.int32)
        assert len(pcm) > 1000 and np.abs(samples).max() > 100, "Fish PCM is empty or silent"
        path = output / f"fish-{role}.wav"
        with wave.open(str(path), "wb") as wav:
            wav.setnchannels(1)
            wav.setsampwidth(2)
            wav.setframerate(adapter.sample_rate)
            wav.writeframes(pcm)
        result = {"role": role, "model": adapter.model, "first_livekit_frame_ms": round((first - started) * 1000),
                  "synthesis_ms": round((time.monotonic() - started) * 1000),
                  "audio_seconds": round(len(pcm) / (adapter.sample_rate * 2), 2), "audio": str(path), "passed": True}
        results.append(result)
        print(json.dumps(result), flush=True)
    (output / "voice-smoke.json").write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("logs/fish-smoke"))
    asyncio.run(main(parser.parse_args().output))
