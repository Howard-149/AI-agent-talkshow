#!/usr/bin/env python3
"""Copy the MSP-Podcast emotional reference clips listed in config/msp_emotion_refs.yaml
into cosyvoice-prompts/msp/<role>/<Emotion>.wav (run where MSP-Podcast is mounted, e.g. Babel).

Each clip is scaled to the same loudness (MSP clips are quiet and uneven, and the clone copies
the level), and its transcript goes next to it as <Emotion>.txt for zero_shot mode.

  python deploy/bake_msp_emotion_refs.py [--msp-root /path/to/MSP-PODCAST-Publish-2.0]
"""
from __future__ import annotations

import argparse
import re
import sys
import wave
from pathlib import Path

import numpy as np
import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
TARGET_DBFS = -20.0


def clean_transcript(text: str) -> str:
    """Drop MSP markers such as "[inaudible 00:01:58]" or "[00:00:59]"; keep the spoken words."""
    text = re.sub(r"\[[^\]]*\]", " ", text)
    return re.sub(r"\s+", " ", text).strip(" .")


def copy_normalized(src: Path, dst: Path) -> float:
    """Write src to dst scaled to TARGET_DBFS RMS (peak-limited); return the gain in dB."""
    with wave.open(str(src)) as w:
        params = w.getparams()
        x = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16).astype(np.float32)
    rms = float(np.sqrt(np.mean(x**2))) or 1.0
    gain = 10 ** (TARGET_DBFS / 20) * 32768.0 / rms
    gain = min(gain, 32767.0 / max(float(np.abs(x).max()), 1.0))
    with wave.open(str(dst), "wb") as w:
        w.setparams(params)
        w.writeframes(np.clip(x * gain, -32768, 32767).astype(np.int16).tobytes())
    return 20 * np.log10(gain)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(REPO_ROOT / "config" / "msp_emotion_refs.yaml"))
    ap.add_argument("--msp-root", default="")
    ap.add_argument("--dest", default=str(REPO_ROOT / "cosyvoice-prompts" / "msp"))
    a = ap.parse_args()
    cfg = yaml.safe_load(Path(a.config).read_text(encoding="utf-8"))
    root = Path(a.msp_root or cfg["msp_root"])
    missing = 0
    for role, spec in cfg["roles"].items():
        out = Path(a.dest) / role
        out.mkdir(parents=True, exist_ok=True)
        for emotion, name in spec["refs"].items():
            src = root / "Audios" / name
            if not src.is_file():
                print(f"missing {src}", file=sys.stderr)
                missing += 1
                continue
            gain = copy_normalized(src, out / f"{emotion}.wav")
            tr = root / "Transcripts" / f"{Path(name).stem}.txt"
            text = clean_transcript(tr.read_text(encoding="utf-8")) if tr.is_file() else ""
            (out / f"{emotion}.txt").write_text(text + "\n", encoding="utf-8")
            print(f"{role:<12} {emotion:<10} ← {name}  gain {gain:+.1f} dB  | {text}")
    return 1 if missing else 0


if __name__ == "__main__":
    sys.exit(main())
