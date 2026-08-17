"""Call DyStream sidecar (or subprocess) for bake and streaming RGBA synthesize."""

from __future__ import annotations

import json
import logging
import os
import struct
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

from avatar.paths import REPO_ROOT, dystream_root

logger = logging.getLogger(__name__)


def sidecar_url() -> str | None:
    raw = os.environ.get("DYSTREAM_SIDECAR_URL", "").strip().rstrip("/")
    return raw or None


def wait_for_dystream_sidecar_ready(
    *,
    timeout_sec: float | None = None,
    poll_sec: float = 2.0,
) -> bool:
    """Block until sidecar /health reports models_warmed (startup warm complete)."""
    sidecar = sidecar_url()
    if not sidecar:
        return True

    if timeout_sec is None:
        timeout_sec = float(os.environ.get("DYSTREAM_SIDECAR_WAIT_SEC", "900"))

    t0 = time.monotonic()
    logger.info("waiting for DyStream sidecar warmup at %s ...", sidecar)
    while time.monotonic() - t0 < timeout_sec:
        try:
            req = urllib.request.Request(f"{sidecar}/health", method="GET")
            with urllib.request.urlopen(req, timeout=5) as resp:
                body = json.loads(resp.read().decode("utf-8"))
            if body.get("models_warmed"):
                warmup = body.get("warmup_sec")
                logger.info(
                    "DyStream sidecar ready after %.1fs wait (warmup_sec=%s)",
                    time.monotonic() - t0,
                    warmup,
                )
                return True
        except Exception as exc:
            logger.debug("sidecar health poll: %s", exc)
        time.sleep(poll_sec)

    logger.warning(
        "DyStream sidecar not warmed after %.0fs — start tmux 3: bash deploy/run-dystream-sidecar.sh",
        timeout_sec,
    )
    return False


def _bake_via_sidecar(
    *,
    sidecar: str,
    portrait: Path,
    audio_wav: Path,
    output_mp4: Path,
    denoising_steps: int,
    npz: Path | None,
    timeout_sec: float,
) -> float:
    payload = {
        "portrait": str(portrait.resolve()),
        "audio": str(audio_wav.resolve()),
        "output": str(output_mp4.resolve()),
        "steps": denoising_steps,
    }
    if npz is not None and npz.is_file():
        payload["npz"] = str(npz.resolve())

    req = urllib.request.Request(
        f"{sidecar}/bake",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    t0 = time.monotonic()
    try:
        with urllib.request.urlopen(req, timeout=timeout_sec) as resp:
            body = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")[:800]
        raise RuntimeError(f"DyStream sidecar HTTP {exc.code}: {detail}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"DyStream sidecar unreachable at {sidecar}: {exc}") from exc

    elapsed = time.monotonic() - t0
    if not body.get("ok"):
        raise RuntimeError(f"DyStream sidecar bake failed: {body.get('error', body)}")
    if not output_mp4.is_file():
        raise RuntimeError(f"DyStream sidecar produced no file: {output_mp4}")
    return elapsed


def _bake_via_subprocess(
    *,
    portrait: Path,
    audio_wav: Path,
    output_mp4: Path,
    denoising_steps: int,
    npz: Path | None,
    timeout_sec: float,
) -> float:
    root = dystream_root()
    device = os.environ.get("DYSTREAM_CUDA_DEVICE", "1").strip() or "1"
    python = os.environ.get("DYSTREAM_PYTHON", "").strip() or sys.executable
    env = os.environ.copy()
    env["DYSTREAM_ROOT"] = str(root)
    env["CUDA_VISIBLE_DEVICES"] = device

    cmd = [
        python,
        "-m",
        "avatar.dystream_once",
        "--portrait",
        str(portrait.resolve()),
        "--audio",
        str(audio_wav.resolve()),
        "--output",
        str(output_mp4.resolve()),
        "--steps",
        str(denoising_steps),
    ]
    if npz is not None and npz.is_file():
        cmd.extend(["--npz", str(npz.resolve())])

    t0 = time.monotonic()
    proc = subprocess.run(
        cmd,
        cwd=str(REPO_ROOT),
        env=env,
        capture_output=True,
        text=True,
        timeout=timeout_sec,
        check=False,
    )
    elapsed = time.monotonic() - t0
    if proc.returncode != 0:
        stderr = (proc.stderr or proc.stdout or "").strip()
        raise RuntimeError(
            f"DyStream bake failed rc={proc.returncode} after {elapsed:.1f}s: {stderr[:800]}"
        )
    if not output_mp4.is_file():
        raise RuntimeError(f"DyStream bake produced no file: {output_mp4}")
    return elapsed


def run_dystream_bake(
    *,
    portrait: Path,
    audio_wav: Path,
    output_mp4: Path,
    denoising_steps: int = 5,
    npz: Path | None = None,
    timeout_sec: float = 600.0,
) -> tuple[Path, float]:
    """Bake on GPU1 via sidecar (warm models) or one-shot subprocess fallback."""
    sidecar = sidecar_url()
    mode = "sidecar" if sidecar else "subprocess"
    logger.info(
        "dystream bake start mode=%s portrait=%s gpu=%s",
        mode,
        portrait.name,
        os.environ.get("DYSTREAM_CUDA_DEVICE", "1"),
    )

    if sidecar:
        elapsed = _bake_via_sidecar(
            sidecar=sidecar,
            portrait=portrait,
            audio_wav=audio_wav,
            output_mp4=output_mp4,
            denoising_steps=denoising_steps,
            npz=npz,
            timeout_sec=timeout_sec,
        )
    else:
        elapsed = _bake_via_subprocess(
            portrait=portrait,
            audio_wav=audio_wav,
            output_mp4=output_mp4,
            denoising_steps=denoising_steps,
            npz=npz,
            timeout_sec=timeout_sec,
        )

    logger.info("dystream bake done output=%s elapsed=%.1fs mode=%s", output_mp4.name, elapsed, mode)
    return output_mp4, elapsed


def _read_exact(fp, n: int) -> bytes:
    buf = bytearray()
    while len(buf) < n:
        chunk = fp.read(n - len(buf))
        if not chunk:
            raise EOFError(f"RGBA stream truncated want={n} got={len(buf)}")
        buf.extend(chunk)
    return bytes(buf)


def iter_sidecar_rgba_stream(
    *,
    sidecar: str,
    portrait: Path,
    audio_wav: Path,
    denoising_steps: int,
    npz: Path | None,
    cache_key: str,
    timeout_sec: float,
):
    """Yield ("meta", dict) once, then ("frame", bytes), then stop at EOS.

    Framing: uint32_be 0xFFFFFFFF + uint32_be meta_len + JSON;
    then uint32_be len + RGBA; uint32_be 0 = EOS.
    """
    payload = {
        "portrait": str(portrait.resolve()),
        "audio": str(audio_wav.resolve()),
        "steps": denoising_steps,
        "cache_key": cache_key,
    }
    if npz is not None and npz.is_file():
        payload["npz"] = str(npz.resolve())

    req = urllib.request.Request(
        f"{sidecar}/synthesize/stream",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        resp = urllib.request.urlopen(req, timeout=timeout_sec)
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")[:800]
        raise RuntimeError(f"DyStream sidecar HTTP {exc.code}: {detail}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"DyStream sidecar unreachable at {sidecar}: {exc}") from exc

    with resp:
        while True:
            hdr = _read_exact(resp, 4)
            (length,) = struct.unpack(">I", hdr)
            if length == 0:
                return
            if length == 0xFFFFFFFF:
                meta_len_raw = _read_exact(resp, 4)
                (meta_len,) = struct.unpack(">I", meta_len_raw)
                meta_bytes = _read_exact(resp, meta_len)
                meta = json.loads(meta_bytes.decode("utf-8"))
                yield ("meta", meta)
                continue
            if length > 64 * 1024 * 1024:
                raise RuntimeError(f"RGBA frame length absurd: {length}")
            yield ("frame", _read_exact(resp, length))


def run_dystream_synthesize_stream(
    *,
    portrait: Path,
    audio_wav: Path,
    denoising_steps: int = 5,
    npz: Path | None = None,
    cache_key: str = "default",
    timeout_sec: float = 600.0,
):
    """Online stream iterator via sidecar /synthesize/stream."""
    sidecar = sidecar_url()
    if not sidecar:
        raise RuntimeError(
            "Online stream requires DYSTREAM_SIDECAR_URL "
            "(or set TALKSHOW_AVATAR_SYNTH=mp4 for legacy bake)"
        )
    logger.info(
        "dystream stream start portrait=%s cache_key=%s gpu=%s",
        portrait.name,
        cache_key,
        os.environ.get("DYSTREAM_CUDA_DEVICE", "1"),
    )
    return iter_sidecar_rgba_stream(
        sidecar=sidecar,
        portrait=portrait,
        audio_wav=audio_wav,
        denoising_steps=denoising_steps,
        npz=npz,
        cache_key=cache_key,
        timeout_sec=timeout_sec,
    )
