"""Avatar asset/clip path helpers and DyStream root / public URL resolution."""

from __future__ import annotations

import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
AVATAR_DIR = REPO_ROOT / "avatar"
DEFAULT_ASSETS_DIR = AVATAR_DIR / "assets"
DEFAULT_CLIPS_DIR = AVATAR_DIR / "runtime" / "clips"

SPEECH_CLIP_SLOT = "current"


def _safe_slot(slot: str) -> str:
    cleaned = "".join(c for c in slot if c.isalnum() or c in ("-", "_"))
    return cleaned or SPEECH_CLIP_SLOT


def _expand_user_path(raw: str) -> Path:
    return Path(raw.replace("${USER}", os.environ.get("USER", ""))).expanduser()


_env_loaded = False


def load_repo_env() -> None:
    """Load repo root .env (for CLI / dystream conda env without manual export)."""
    global _env_loaded
    if _env_loaded:
        return
    _env_loaded = True
    env_file = REPO_ROOT / ".env"
    if not env_file.is_file():
        return
    try:
        from dotenv import load_dotenv

        load_dotenv(env_file, override=False)
        return
    except ImportError:
        pass
    for line in env_file.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, val = line.partition("=")
        key = key.strip()
        if not key or key in os.environ:
            continue
        val = val.strip().strip('"').strip("'")
        os.environ[key] = val.replace("${USER}", os.environ.get("USER", ""))


def avatar_assets_dir() -> Path:
    """Portraits + idle loops (fixed assets under repo by default)."""
    load_repo_env()
    raw = os.environ.get("AVATAR_ASSETS_DIR", "").strip()
    if raw:
        return _expand_user_path(raw)
    return DEFAULT_ASSETS_DIR


def avatar_clips_dir() -> Path:
    """Speech clip slots under avatar/runtime/clips/ (current.mp4 + chunk cN.mp4)."""
    load_repo_env()
    raw = os.environ.get("AVATAR_CLIPS_DIR", "").strip()
    if raw:
        return _expand_user_path(raw)
    return DEFAULT_CLIPS_DIR


def speech_clip_path(slot: str = SPEECH_CLIP_SLOT) -> Path:
    return avatar_clips_dir() / f"{_safe_slot(slot)}.mp4"


def speech_wav_path(slot: str = SPEECH_CLIP_SLOT) -> Path:
    return avatar_clips_dir() / f"{_safe_slot(slot)}.wav"


def prune_stale_speech_clips() -> None:
    """Drop leftover speech MP4/WAV/work from older runs; keep only current.* slot."""
    import shutil

    clips = avatar_clips_dir()
    if not clips.is_dir():
        return
    keep = {
        f"{SPEECH_CLIP_SLOT}.mp4",
        f"{SPEECH_CLIP_SLOT}.wav",
        ".gitkeep",
    }
    for entry in clips.iterdir():
        if entry.name in keep:
            continue
        if entry.is_dir():
            if entry.name == "work":
                shutil.rmtree(entry, ignore_errors=True)
            continue
        if entry.is_file() and entry.suffix.lower() in (
            ".mp4",
            ".wav",
            ".rgba",
            ".json",
        ):
            try:
                entry.unlink()
            except OSError:
                pass


def cleanup_speech_slots(slots: list[str]) -> None:
    """Remove temporary chunk WAV/MP4/RGBA slots after a panel line finishes."""
    for slot in slots:
        if _safe_slot(slot) == SPEECH_CLIP_SLOT:
            continue
        speech_clip_path(slot).unlink(missing_ok=True)
        speech_wav_path(slot).unlink(missing_ok=True)
        # Scrub leftover offline-pack files from older builds.
        clips = avatar_clips_dir()
        safe = _safe_slot(slot)
        (clips / f"{safe}.rgba").unlink(missing_ok=True)
        (clips / f"{safe}.meta.json").unlink(missing_ok=True)


def ensure_avatar_dirs() -> None:
    """Create clips + assets dirs; prune stale speech files in clips."""
    avatar_assets_dir().mkdir(parents=True, exist_ok=True)
    avatar_clips_dir().mkdir(parents=True, exist_ok=True)
    prune_stale_speech_clips()


def dystream_root() -> Path:
    """External DyStream install — set DYSTREAM_ROOT (checkpoints too large for repo)."""
    load_repo_env()
    raw = os.environ.get("DYSTREAM_ROOT", "").strip()
    if not raw:
        raise FileNotFoundError(
            "DYSTREAM_ROOT is not set. Example (Babel):\n"
            "  export DYSTREAM_ROOT=/data/user_data/$USER/dystream\n"
            "  # or add DYSTREAM_ROOT=... to repo .env\n"
            "  bash deploy/clone-dystream.sh"
        )
    path = _expand_user_path(raw)
    if not path.is_dir() or not (path / "app.py").is_file():
        raise FileNotFoundError(
            f"DYSTREAM_ROOT invalid (missing app.py): {path}\n"
            "Run: bash deploy/clone-dystream.sh"
        )
    return path


def resolve_avatar_asset_path(spec: str) -> Path:
    """Persona yaml paths relative to AVATAR_ASSETS_DIR, or absolute."""
    spec = spec.strip()
    if not spec:
        raise ValueError("empty avatar asset path")
    expanded = spec.replace("${USER}", os.environ.get("USER", ""))
    path = Path(expanded).expanduser()
    if path.is_absolute():
        return path
    return avatar_assets_dir() / path


def avatar_clips_base_url() -> str:
    return os.environ.get(
        "AVATAR_CLIPS_BASE_URL", "http://127.0.0.1:8765/clips"
    ).rstrip("/")


def avatar_clip_public_url(filename: str) -> str:
    return f"{avatar_clips_base_url()}/{filename.lstrip('/')}"
