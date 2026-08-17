#!/usr/bin/env python3
"""Bake CosyVoice clone prompt WAVs from existing Piper ONNX voices.

Uses the same PIPER_MODEL_PATH* env vars as the talkshow agent so Lessac / Amy /
Ryan become CosyVoice instruct2 / zero_shot references. One wav per role —
clone refs are language-agnostic (zh/en dialogue share the same timbre).

Examples (Babel, talkshow conda with piper-tts):

  set -a && source .env && set +a
  python deploy/bake_cosyvoice_prompts_from_piper.py

  # Paths in config/personas/*.yaml → cosyvoice-prompts/{role}.wav
  #   TALKSHOW_TTS_ENGINE=cosyvoice
  #   COSYVOICE_MODE=instruct2
  #   COSYVOICE_SIDECAR_URL=http://127.0.0.1:8767
"""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import wave
from pathlib import Path

# Neutral line ~4–6s — clear, no emotion tags, good for speaker cloning.
PROMPT_TEXT = (
    "Hello everyone, welcome to our live talk show. "
    "Today we are discussing technology and everyday life with our panel."
)

ROLES = ("host", "guest", "commentator")
REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DEST = REPO_ROOT / "cosyvoice-prompts"


def _load_dotenv(repo_root: Path) -> None:
    env_path = repo_root / ".env"
    if not env_path.is_file():
        return
    for line in env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, val = line.partition("=")
        key = key.strip()
        val = val.strip()
        if "#" in val and not (val.startswith("'") or val.startswith('"')):
            val = val.split("#", 1)[0].rstrip()
        val = val.strip().strip("'").strip('"')
        os.environ.setdefault(key, val)


def _expand(path: str) -> str:
    return os.path.expanduser(path.replace("${USER}", os.environ.get("USER", "")))


def resolve_piper_onnx(role: str, locale: str = "en") -> Path | None:
    """Mirror agent/config.load_persona_tts Piper env priority."""
    role_key = role.upper()
    loc_key = locale.upper()
    candidates: list[str] = [
        os.environ.get(f"PIPER_MODEL_PATH_{loc_key}_{role_key}", "") or "",
    ]
    if role == "host":
        candidates.append(os.environ.get(f"PIPER_MODEL_PATH_{loc_key}", "") or "")
    if locale == "en":
        if role == "host":
            candidates.append(os.environ.get("PIPER_MODEL_PATH", "") or "")
        else:
            candidates.append(os.environ.get(f"PIPER_MODEL_PATH_{role_key}", "") or "")
    if locale == "zh" and role == "host":
        candidates.append(os.environ.get("PIPER_MODEL_PATH_ZH", "") or "")

    for raw in candidates:
        raw = raw.strip()
        if not raw:
            continue
        path = Path(_expand(raw))
        if path.is_file():
            return path
    return None


def _piper_config_path(model_path: Path) -> Path:
    modern = Path(f"{model_path}.json")
    legacy = model_path.with_suffix(".json")
    if modern.is_file():
        return modern
    if legacy.is_file():
        return legacy
    return modern


def synthesize_with_piper_python(model_path: Path, text: str) -> tuple[bytes, int]:
    import numpy as np
    from piper import PiperVoice
    from piper.config import SynthesisConfig

    config_path = _piper_config_path(model_path)
    if not config_path.is_file():
        raise FileNotFoundError(f"Missing Piper config beside model: {config_path}")

    voice = PiperVoice.load(str(model_path), config_path=str(config_path), use_cuda=False)
    rate = int(voice.config.sample_rate)
    syn = SynthesisConfig()
    phoneme_ids: list[int] = []
    if hasattr(voice, "phonemize") and hasattr(voice, "phoneme_ids_to_audio"):
        for phonemes in voice.phonemize(text):
            if phonemes:
                phoneme_ids.extend(voice.phonemes_to_ids(phonemes))
        if phoneme_ids:
            audio = voice.phoneme_ids_to_audio(phoneme_ids, syn_config=syn)
            if isinstance(audio, tuple):
                audio = audio[0]
            pcm = np.clip(audio * 32767.0, -32767, 32767).astype(np.int16).tobytes()
            return pcm, rate

    chunks: list[bytes] = []
    for chunk in voice.synthesize(text):
        audio = chunk.audio_float_array if hasattr(chunk, "audio_float_array") else chunk
        if hasattr(audio, "astype"):
            pcm = np.clip(audio * 32767.0, -32767, 32767).astype(np.int16).tobytes()
        else:
            pcm = bytes(audio)
        chunks.append(pcm)
    if not chunks:
        raise RuntimeError(f"Piper produced no audio for {model_path.name}")
    return b"".join(chunks), rate


def synthesize_with_piper_cli(model_path: Path, text: str, out_wav: Path) -> None:
    piper_bin = shutil.which("piper")
    if not piper_bin:
        raise RuntimeError("piper-tts Python package and piper CLI both unavailable")
    out_wav.parent.mkdir(parents=True, exist_ok=True)
    cmd = [piper_bin, "--model", str(model_path), "--output_file", str(out_wav)]
    subprocess.run(cmd, input=text.encode("utf-8"), check=True)


def write_wav(path: Path, pcm: bytes, sample_rate: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm)


def bake_one(model_path: Path, text: str, out_wav: Path) -> float:
    try:
        pcm, rate = synthesize_with_piper_python(model_path, text)
        write_wav(out_wav, pcm, rate)
    except ImportError:
        synthesize_with_piper_cli(model_path, text, out_wav)
        with wave.open(str(out_wav), "rb") as wf:
            rate = wf.getframerate()
            nframes = wf.getnframes()
            return nframes / float(rate) if rate else 0.0
    with wave.open(str(out_wav), "rb") as wf:
        return wf.getnframes() / float(wf.getframerate())


def main() -> int:
    _load_dotenv(REPO_ROOT)

    parser = argparse.ArgumentParser(
        description="Bake CosyVoice prompt WAVs from Piper voices (one per role)"
    )
    parser.add_argument(
        "--dest",
        default=str(DEFAULT_DEST),
        help=f"Output directory (default: {DEFAULT_DEST})",
    )
    parser.add_argument(
        "--zh",
        action="store_true",
        help="Deprecated no-op: clone refs are language-agnostic (kept for old scripts)",
    )
    parser.add_argument(
        "--text",
        default=PROMPT_TEXT,
        help="Prompt utterance (any language; used only to bake the ref wav)",
    )
    parser.add_argument(
        "--text-en",
        default="",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--text-zh",
        default="",
        help=argparse.SUPPRESS,
    )
    args = parser.parse_args()
    if args.zh:
        print(
            "Note: --zh ignored — CosyVoice instruct2 shares one ref wav per role "
            "across locales.",
            file=sys.stderr,
        )
    text = (args.text_en or args.text or PROMPT_TEXT).strip() or PROMPT_TEXT
    dest = Path(os.path.expanduser(args.dest))
    if not dest.is_absolute():
        dest = REPO_ROOT / dest
    dest.mkdir(parents=True, exist_ok=True)

    baked: list[tuple[str, Path, float]] = []
    missing: list[str] = []

    for role in ROLES:
        onnx = resolve_piper_onnx(role, "en")
        if onnx is None:
            missing.append(role)
            continue
        out_path = dest / f"{role}.wav"
        legacy = dest / f"en_{role}.wav"
        print(f"Baking {out_path.name} ← {onnx.name} …", flush=True)
        try:
            dur = bake_one(onnx, text, out_path)
        except Exception as exc:  # noqa: BLE001
            print(f"  FAILED: {exc}", file=sys.stderr)
            return 1
        shutil.copy2(out_path, legacy)
        print(f"  → {out_path} (+ {legacy.name}) ({dur:.2f}s)")
        baked.append((role, out_path, dur))

    if missing:
        print("\nSkipped (no PIPER_MODEL_PATH*): " + ", ".join(missing), file=sys.stderr)
    if not baked:
        print("Nothing baked. Set PIPER_MODEL_PATH* in .env first.", file=sys.stderr)
        return 1

    print("\n# Prompt paths: config/personas/*.yaml → cosyvoice-prompts/{role}.wav")
    print("# .env needs (no per-locale WAV vars):")
    print("TALKSHOW_TTS_ENGINE=cosyvoice")
    print("COSYVOICE_MODE=instruct2")
    print("# COSYVOICE_MODEL_DIR=.../pretrained_models/Fun-CosyVoice3-0.5B")
    print("# COSYVOICE_SIDECAR_URL=http://127.0.0.1:8767")
    print(f"# Wrote {len(baked)} role ref(s) under {dest}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
