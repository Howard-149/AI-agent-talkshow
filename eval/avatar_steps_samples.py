#!/usr/bin/env python3
"""
Render one spoken line at several DyStream step counts, for a side-by-side quality check.

Run where the stack's sidecars are up (CosyVoice/Piper and DyStream):
    python -m eval.avatar_steps_samples                      # guest, steps 5,4,3,2
    python -m eval.avatar_steps_samples --role host --steps 5,3

Writes logs/avatar-steps/<role>-steps<N>.mp4 (with the line's audio when ffmpeg is
available) plus summary.json with each render's wall time and frames per second.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import shutil
import subprocess
import time
from pathlib import Path

from dotenv import load_dotenv

DEFAULT_TEXT = (
    "I think the real question is not whether one senior engineer can replace three, "
    "but who reviews their work when they are wrong."
)


async def _speech_wav(role: str, text: str, out_wav: Path) -> float:
    """Synthesize ``text`` in ``role``'s voice to a DyStream-rate wav; returns seconds of audio."""
    from agent.adapters.tts_synthesize import synthesize_pcm_for_config
    from agent.config import load_config, load_persona_tts
    from agent.adapters.dystream_bridge import DYSTREAM_AUDIO_RATE
    from avatar.pcm_utils import pcm16_to_wav, resample_pcm16_mono

    pcm, sample_rate, _ = await synthesize_pcm_for_config(
        load_persona_tts(role, load_config()), text, role=role, step="steps_sample"
    )
    if not pcm:
        raise RuntimeError("TTS returned no audio")
    pcm16 = resample_pcm16_mono(pcm, sample_rate, DYSTREAM_AUDIO_RATE)
    pcm16_to_wav(pcm16, out_wav, sample_rate=DYSTREAM_AUDIO_RATE)
    return len(pcm16) / 2 / DYSTREAM_AUDIO_RATE


def main(argv: list[str] | None = None) -> int:
    load_dotenv()
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--role", default="guest")
    ap.add_argument("--steps", default="5,4,3,2", help="comma-separated step counts")
    ap.add_argument("--text", default=DEFAULT_TEXT)
    ap.add_argument("--out", type=Path, default=Path("logs/avatar-steps"))
    args = ap.parse_args(argv)

    from agent.config import load_persona_avatar
    from avatar.dystream_job import run_dystream_bake, wait_for_dystream_sidecar_ready
    from avatar.paths import resolve_avatar_asset_path

    av = load_persona_avatar(args.role)
    if not av.portrait:
        raise SystemExit(f"no portrait configured for role {args.role}")
    portrait = resolve_avatar_asset_path(av.portrait)
    args.out.mkdir(parents=True, exist_ok=True)
    wav = args.out / f"{args.role}.wav"
    audio_s = asyncio.run(_speech_wav(args.role, args.text, wav))
    wait_for_dystream_sidecar_ready()
    ffmpeg = shutil.which("ffmpeg")

    summary = {"role": args.role, "text": args.text, "audio_s": round(audio_s, 3), "renders": []}
    for steps in [int(x) for x in args.steps.split(",") if x.strip()]:
        video = args.out / f"{args.role}-steps{steps}.video.mp4"
        t0 = time.monotonic()
        run_dystream_bake(portrait=portrait, audio_wav=wav, output_mp4=video, denoising_steps=steps)
        wall = time.monotonic() - t0
        final = args.out / f"{args.role}-steps{steps}.mp4"
        if ffmpeg:
            subprocess.run(
                [ffmpeg, "-y", "-loglevel", "error", "-i", str(video), "-i", str(wav),
                 "-map", "0:v", "-map", "1:a", "-c:v", "copy", "-c:a", "aac", "-shortest", str(final)],
                check=True,
            )
            video.unlink(missing_ok=True)
        else:
            video.rename(final)
        frames = audio_s * 25.0
        summary["renders"].append(
            {"steps": steps, "file": str(final), "render_s": round(wall, 2), "fps": round(frames / wall, 1)}
        )
        print(f"steps={steps}: {wall:.1f} s for {audio_s:.1f} s of audio (~{frames / wall:.1f} fps) -> {final}")
    (args.out / "summary.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    if not ffmpeg:
        print("ffmpeg not found: clips have no audio")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
