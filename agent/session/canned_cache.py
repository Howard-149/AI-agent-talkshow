"""Pre-rendered canned host lines, replayed instead of rendered on every use.

The host's canned lines (session welcome, open floor, speaker intros, the
human grant, the round close) are fixed text, yet each one went through TTS
and a DyStream render every time it was spoken. With
``TALKSHOW_PRERENDER_CANNED=1`` they are rendered once into a disk cache by
``python -m agent.session.canned_warm`` and replayed from it; a line that is
not cached is rendered live, as before.

A cached clip is the line's int16 PCM plus its avatar frames as JPEG (the
LiveKit track is I420, so alpha is never sent). The key covers everything that
changes the audio or the face: role, locale, text, TTS engine/voice settings
and the portrait, so a changed voice or portrait simply misses.

Canned lines are rendered with a neutral emotion; the host's live PAD mood is
not applied to them.
"""

from __future__ import annotations

import hashlib
import io
import json
import logging
import os
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

logger = logging.getLogger(__name__)

CACHE_VERSION = 1
JPEG_QUALITY = 90

# LineSpec kinds (agent/show_graph/state.py) whose text comes from a fixed list.
CANNED_KINDS = frozenset({"welcome", "open_floor", "intro", "grant_human", "close"})


def prerender_enabled() -> bool:
    raw = os.environ.get("TALKSHOW_PRERENDER_CANNED", "").strip().lower()
    return raw in ("1", "true", "yes", "on")


def cache_dir() -> Path:
    raw = os.environ.get("TALKSHOW_CANNED_CACHE_DIR", "").strip()
    if raw:
        return Path(raw).expanduser()
    from avatar.paths import REPO_ROOT

    return REPO_ROOT / "avatar" / "runtime" / "canned"


@dataclass(frozen=True)
class CannedClip:
    key: str
    role: str
    locale: str
    text: str
    pcm: bytes
    sample_rate: int
    # JPEG-encoded frames; empty when the clip was rendered with the avatar off.
    frames_jpeg: tuple[bytes, ...]
    fps: float
    width: int
    height: int

    @property
    def duration_sec(self) -> float:
        return (len(self.pcm) / 2) / float(self.sample_rate) if self.sample_rate else 0.0


def _file_stamp(path: str | Path | None) -> str:
    """Path + size + mtime, so a re-recorded prompt wav or portrait misses."""
    if not path:
        return ""
    p = Path(path)
    try:
        st = p.stat()
    except OSError:
        return str(p)
    return f"{p}:{st.st_size}:{int(st.st_mtime)}"


def clip_key(
    *,
    role: str,
    locale: str,
    text: str,
    voice: dict[str, str],
    portrait: str | Path | None,
    width: int,
    height: int,
) -> str:
    """Stable key for one canned line. ``portrait`` is None when the avatar is off."""
    ident = {
        "v": CACHE_VERSION,
        "role": role,
        "locale": locale,
        "text": text.strip(),
        "voice": dict(sorted(voice.items())),
        "portrait": _file_stamp(portrait) if portrait else "",
        "size": [width, height] if portrait else [],
    }
    blob = json.dumps(ident, sort_keys=True, ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()[:32]


def voice_identity(tts_cfg: object, *, role: str, locale: str) -> dict[str, str]:
    """TTS settings that change a canned line's audio (engine, voice, speed)."""
    from agent.adapters.tts_synthesize import resolve_tts_engine

    engine = resolve_tts_engine(tts_cfg)  # type: ignore[arg-type]
    model_path = str(getattr(tts_cfg, "model_path", "") or "")
    voice = {"engine": engine, "model": _file_stamp(model_path), "emotion": "neutral"}
    if engine == "cosyvoice":
        from agent.adapters.cosyvoice_voices import cosyvoice_mode, resolve_spk_id

        voice["mode"] = cosyvoice_mode()
        voice["spk_id"] = resolve_spk_id(role=role, locale=locale, model_path=model_path) or ""
        voice["speed"] = os.environ.get("COSYVOICE_SPEED", "").strip()
    return voice


def video_size() -> tuple[int, int]:
    return (
        int(os.environ.get("TALKSHOW_AVATAR_VIDEO_WIDTH", "512")),
        int(os.environ.get("TALKSHOW_AVATAR_VIDEO_HEIGHT", "512")),
    )


# --- frames <-> JPEG ---------------------------------------------------------


def encode_frames(frames: Iterable[bytes], width: int, height: int) -> list[bytes]:
    from PIL import Image

    out = []
    for fr in frames:
        img = Image.frombytes("RGBA", (width, height), fr).convert("RGB")
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=JPEG_QUALITY)
        out.append(buf.getvalue())
    return out


def decode_frames(frames_jpeg: Iterable[bytes], width: int, height: int) -> list[bytes]:
    from PIL import Image

    out = []
    for jpg in frames_jpeg:
        img = Image.open(io.BytesIO(jpg)).convert("RGBA")
        if img.size != (width, height):
            img = img.resize((width, height))
        out.append(img.tobytes())
    return out


# --- disk ---------------------------------------------------------------------


def load(key: str) -> CannedClip | None:
    """Read one clip, or None when it is missing or unreadable."""
    d = cache_dir() / key
    try:
        meta = json.loads((d / "meta.json").read_text(encoding="utf-8"))
        if meta.get("v") != CACHE_VERSION:
            return None
        pcm = (d / "audio.pcm").read_bytes()
        blob = (d / "frames.bin").read_bytes() if meta.get("frame_sizes") else b""
    except (OSError, ValueError):
        return None
    frames, pos = [], 0
    for n in meta.get("frame_sizes") or []:
        frames.append(blob[pos : pos + n])
        pos += n
    return CannedClip(
        key=key,
        role=meta["role"],
        locale=meta["locale"],
        text=meta["text"],
        pcm=pcm,
        sample_rate=int(meta["sample_rate"]),
        frames_jpeg=tuple(frames),
        fps=float(meta.get("fps") or 25.0),
        width=int(meta.get("width") or 0),
        height=int(meta.get("height") or 0),
    )


def store(clip: CannedClip, *, extra: dict | None = None) -> Path:
    """Write a clip atomically (temp dir, then rename)."""
    root = cache_dir()
    root.mkdir(parents=True, exist_ok=True)
    final = root / clip.key
    tmp = Path(tempfile.mkdtemp(prefix=f".{clip.key}.", dir=root))
    try:
        (tmp / "audio.pcm").write_bytes(clip.pcm)
        if clip.frames_jpeg:
            (tmp / "frames.bin").write_bytes(b"".join(clip.frames_jpeg))
        meta = {
            "v": CACHE_VERSION,
            "role": clip.role,
            "locale": clip.locale,
            "text": clip.text,
            "sample_rate": clip.sample_rate,
            "fps": clip.fps,
            "width": clip.width,
            "height": clip.height,
            "frame_sizes": [len(f) for f in clip.frames_jpeg],
            "duration_sec": round(clip.duration_sec, 3),
            **(extra or {}),
        }
        (tmp / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")
        if final.exists():
            shutil.rmtree(final)
        tmp.rename(final)
    except BaseException:
        shutil.rmtree(tmp, ignore_errors=True)
        raise
    return final


# --- which lines are canned -----------------------------------------------------


def canned_lines(panel_names: Iterable[str], *, close_line: str, welcome: str) -> list[tuple[str, str]]:
    """(kind, English text) for every fixed host line, most frequent first.

    Grant lines that name the human's topic are not fixed text and stay live.
    """
    from agent.floor.host_lines import _HUMAN_FLOOR, _INTRO_SPEAKER, _OPEN_FLOOR

    lines = [("open_floor", t) for t in _OPEN_FLOOR]
    for name in panel_names:
        lines += [("intro", t.format(name=name)) for t in _INTRO_SPEAKER]
    lines += [("grant_human", t) for t in _HUMAN_FLOOR]
    lines += [("close", close_line), ("welcome", welcome)]
    return lines
