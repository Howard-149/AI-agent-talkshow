#!/usr/bin/env python3
"""CLI: bake DyStream speech clip or idle loop from portrait + audio."""
from __future__ import annotations

import argparse
import sys
import uuid
from pathlib import Path

from avatar.dystream_job import run_dystream_bake
from avatar.pcm_utils import silent_wav


def main() -> int:
    parser = argparse.ArgumentParser(description="Bake DyStream avatar clip (offline)")
    sub = parser.add_subparsers(dest="cmd", required=True)

    speech = sub.add_parser("speech", help="Portrait + WAV → MP4")
    speech.add_argument("--portrait", required=True)
    speech.add_argument("--audio", required=True)
    speech.add_argument("--output", required=True)
    speech.add_argument("--steps", type=int, default=5)

    idle = sub.add_parser("idle", help="Portrait + silent audio → idle loop MP4")
    idle.add_argument("--portrait", required=True)
    idle.add_argument("--output", required=True)
    idle.add_argument("--duration", type=float, default=3.0)
    idle.add_argument("--steps", type=int, default=5)

    args = parser.parse_args()
    portrait = Path(args.portrait)
    output = Path(args.output)
    if not portrait.is_file():
        print(f"Portrait not found: {portrait}", file=sys.stderr)
        return 1

    if args.cmd == "speech":
        audio = Path(args.audio)
        if not audio.is_file():
            print(f"Audio not found: {audio}", file=sys.stderr)
            return 1
        run_dystream_bake(
            portrait=portrait,
            audio_wav=audio,
            output_mp4=output,
            denoising_steps=args.steps,
        )
    else:
        work = output.parent / f".idle-{uuid.uuid4().hex[:8]}.wav"
        try:
            silent_wav(work, duration_sec=args.duration, sample_rate=16000)
            run_dystream_bake(
                portrait=portrait,
                audio_wav=work,
                output_mp4=output,
                denoising_steps=args.steps,
            )
        finally:
            work.unlink(missing_ok=True)

    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
