#!/usr/bin/env python3
"""One-shot DyStream offline inference (requires DYSTREAM_ROOT)."""
from __future__ import annotations

import argparse
import os
import shutil
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from avatar.paths import dystream_root


def main() -> int:
    parser = argparse.ArgumentParser(description="DyStream single-clip bake")
    parser.add_argument("--portrait", required=True, help="Reference face PNG/JPG")
    parser.add_argument("--audio", required=True, help="Speaker WAV (16 kHz preferred)")
    parser.add_argument("--output", required=True, help="Output MP4 path")
    parser.add_argument("--steps", type=int, default=5, help="Denoising steps")
    parser.add_argument("--npz", default="", help="Optional precomputed motion NPZ")
    args = parser.parse_args()

    root = dystream_root()
    portrait = Path(args.portrait).resolve()
    audio = Path(args.audio).resolve()
    output = Path(args.output).resolve()
    if not portrait.is_file():
        print(f"Portrait missing: {portrait}", file=sys.stderr)
        return 1
    if not audio.is_file():
        print(f"Audio missing: {audio}", file=sys.stderr)
        return 1

    os.chdir(root)
    sys.path.insert(0, str(root))

    from PIL import Image

    from app import run_inference

    npz = args.npz.strip() or None
    if npz and not Path(npz).is_file():
        npz = None

    video_path, _, _, _ = run_inference(
        Image.open(portrait).convert("RGB"),
        str(audio),
        None,
        args.steps,
        0.5,
        0.5,
        0.0,
        1.0,
        precomputed_npz_path=npz,
        video_audio_path=str(audio),
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(video_path, output)
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
