#!/usr/bin/env python3
"""DyStream sidecar — warm models on GPU 1.

Endpoints:
  GET  /health
  POST /bake              — legacy MP4 bake (compat)
  POST /synthesize/stream — online: length-prefixed RGBA while generating
                            Framing: uint32_be 0xFFFFFFFF + uint32_be meta_len + JSON meta,
                            then uint32_be len + RGBA…, then uint32_be 0 (EOS).
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import struct
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from avatar.paths import dystream_root, load_repo_env

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("dystream_sidecar")

_bake_lock = threading.Lock()
_models_warmed = False
_warmup_sec: float | None = None
_run_inference = None


def _setup_dystream() -> None:
    global _run_inference
    load_repo_env()
    root = dystream_root()
    os.chdir(root)
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    from app import run_inference  # noqa: WPS433

    _run_inference = run_inference


def warm_models() -> None:
    global _models_warmed, _warmup_sec
    if _models_warmed:
        return
    _setup_dystream()
    from app import (  # noqa: WPS433
        load_dystream_model,
        load_face_detector,
        load_visualization_model,
    )

    logger.info("warming DyStream models — blocking until ready (may take several minutes)...")
    t0 = time.monotonic()
    load_dystream_model()
    load_visualization_model()
    load_face_detector()

    # Per-role portrait latents + optional short AR probe (env-gated).
    warm_roles = os.environ.get("DYSTREAM_SIDECAR_WARM_ROLES", "1").strip().lower() not in (
        "0",
        "false",
        "no",
        "off",
    )
    warm_probe = os.environ.get("DYSTREAM_SIDECAR_WARM_PROBE", "1").strip().lower() not in (
        "0",
        "false",
        "no",
        "off",
    )
    if warm_roles:
        try:
            from avatar.dystream_synth import warm_ar_probe, warm_role_caches

            warm_role_caches(probe_render=True)
            if warm_probe:
                warm_ar_probe()
        except Exception:
            logger.exception("role/AR warmup failed — continuing (first turn may be slower)")

    _warmup_sec = time.monotonic() - t0
    _models_warmed = True
    try:
        from avatar.dystream_synth import cuda_mem_mb

        mem = cuda_mem_mb()
    except Exception:
        mem = None
    logger.info(
        "models ready in %.1fs — sidecar can accept /bake /synthesize/stream "
        "(cuda_alloc_mb=%s warm_roles=%s warm_probe=%s)",
        _warmup_sec,
        mem,
        warm_roles,
        warm_probe,
    )


def bake_clip(
    *,
    portrait: Path,
    audio: Path,
    output: Path,
    steps: int,
    npz: Path | None = None,
) -> float:
    from PIL import Image

    if not _models_warmed:
        warm_models()
    assert _run_inference is not None

    if not portrait.is_file():
        raise FileNotFoundError(f"portrait missing: {portrait}")
    if not audio.is_file():
        raise FileNotFoundError(f"audio missing: {audio}")

    try:
        from avatar.dystream_synth import release_cuda

        t0 = time.monotonic()
        video_path, _, _, _ = _run_inference(
            Image.open(portrait).convert("RGB"),
            str(audio),
            None,
            steps,
            0.5,
            0.5,
            0.0,
            1.0,
            precomputed_npz_path=str(npz) if npz and npz.is_file() else None,
            video_audio_path=str(audio),
        )
        output.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(video_path, output)
        return time.monotonic() - t0
    finally:
        try:
            from avatar.dystream_synth import release_cuda

            release_cuda()
        except Exception:
            pass


def _parse_common(data: dict[str, Any]) -> dict[str, Any]:
    portrait = Path(str(data["portrait"])).expanduser().resolve()
    audio = Path(str(data["audio"])).expanduser().resolve()
    steps = int(data.get("steps", 5))
    npz_raw = data.get("npz")
    npz = Path(str(npz_raw)).expanduser().resolve() if npz_raw else None
    cache_key = str(data.get("cache_key") or data.get("role") or "default")
    return {
        "portrait": portrait,
        "audio": audio,
        "steps": steps,
        "npz": npz,
        "cache_key": cache_key,
    }


class SidecarHandler(BaseHTTPRequestHandler):
    server_version = "DyStreamSidecar/1.1"

    def log_message(self, fmt: str, *args: Any) -> None:
        logger.info("%s - %s", self.address_string(), fmt % args)

    def _json(self, code: int, payload: dict[str, Any]) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _read_json(self) -> dict[str, Any] | None:
        length = int(self.headers.get("Content-Length", "0"))
        try:
            raw = self.rfile.read(length)
            return json.loads(raw.decode("utf-8"))
        except (json.JSONDecodeError, ValueError) as exc:
            self._json(400, {"ok": False, "error": f"invalid json: {exc}"})
            return None

    def do_GET(self) -> None:
        path = urlparse(self.path).path.rstrip("/") or "/"
        if path in ("", "/", "/health"):
            self._json(
                200,
                {
                    "ok": True,
                    "models_warmed": _models_warmed,
                    "warmup_sec": round(_warmup_sec, 1)
                    if _warmup_sec is not None
                    else None,
                    "service": "dystream_sidecar",
                    "apis": ["bake", "synthesize/stream"],
                    "synth_mode": "online_stream",
                },
            )
            return
        self._json(404, {"ok": False, "error": "not found"})

    def do_POST(self) -> None:
        path = urlparse(self.path).path.rstrip("/")
        data = self._read_json()
        if data is None:
            return

        if path == "/bake":
            self._handle_bake(data)
            return
        if path == "/synthesize/stream":
            self._handle_synthesize_stream(data)
            return
        self._json(404, {"ok": False, "error": "not found"})

    def _handle_bake(self, data: dict[str, Any]) -> None:
        try:
            common = _parse_common(data)
            output = Path(str(data["output"])).expanduser().resolve()
        except (KeyError, TypeError, ValueError) as exc:
            self._json(400, {"ok": False, "error": f"bad request: {exc}"})
            return

        with _bake_lock:
            try:
                elapsed = bake_clip(
                    portrait=common["portrait"],
                    audio=common["audio"],
                    output=output,
                    steps=common["steps"],
                    npz=common["npz"],
                )
            except Exception as exc:
                logger.exception("bake failed")
                self._json(500, {"ok": False, "error": str(exc)[:800]})
                return

        self._json(
            200,
            {
                "ok": True,
                "output": str(output),
                "bake_ms": round(elapsed * 1000),
            },
        )

    def _handle_synthesize_stream(self, data: dict[str, Any]) -> None:
        """Online stream: write each RGBA frame as AR+render completes."""
        try:
            common = _parse_common(data)
        except (KeyError, TypeError, ValueError) as exc:
            self._json(400, {"ok": False, "error": f"bad request: {exc}"})
            return

        if not common["portrait"].is_file() or not common["audio"].is_file():
            self._json(400, {"ok": False, "error": "portrait or audio missing"})
            return

        # Acquire lock before headers so concurrent roles serialize on one GPU.
        if not _bake_lock.acquire(blocking=True):
            self._json(503, {"ok": False, "error": "sidecar busy"})
            return

        headers_sent = False
        try:
            if not _models_warmed:
                warm_models()
            from avatar.dystream_synth import (  # noqa: WPS433
                OnlineMeta,
                iter_synthesize_online,
                release_cuda,
            )

            def _send(blob: bytes) -> None:
                self.wfile.write(f"{len(blob):X}\r\n".encode("ascii"))
                self.wfile.write(blob)
                self.wfile.write(b"\r\n")
                self.wfile.flush()

            frame_count = 0
            for item in iter_synthesize_online(
                portrait=common["portrait"],
                audio_wav=common["audio"],
                denoising_steps=common["steps"],
                npz=common["npz"]
                if common["npz"] and common["npz"].is_file()
                else None,
                cache_key=common["cache_key"],
            ):
                if isinstance(item, OnlineMeta):
                    meta = {
                        "ok": True,
                        "width": item.width,
                        "height": item.height,
                        "fps": item.fps,
                        "expected_frames": item.expected_frames,
                        "audio_feat_ms": item.audio_feat_ms,
                        "stream": "online",
                    }
                    meta_bytes = json.dumps(meta).encode("utf-8")
                    if not headers_sent:
                        self.send_response(200)
                        self.send_header(
                            "Content-Type", "application/x-talkshow-rgba-stream"
                        )
                        self.send_header("Transfer-Encoding", "chunked")
                        self.end_headers()
                        headers_sent = True
                    # Prelude: marker + meta_len + JSON
                    _send(
                        struct.pack(">I", 0xFFFFFFFF)
                        + struct.pack(">I", len(meta_bytes))
                        + meta_bytes
                    )
                    continue

                if not headers_sent:
                    # Should not happen (meta precedes frames); fail safe.
                    self.send_response(500)
                    self.send_header("Content-Type", "application/json")
                    self.end_headers()
                    self.wfile.write(
                        b'{"ok":false,"error":"stream missing meta"}'
                    )
                    return

                fr: bytes = item
                _send(struct.pack(">I", len(fr)) + fr)
                frame_count += 1

            if not headers_sent:
                self._json(
                    500,
                    {"ok": False, "error": "online synth produced no frames"},
                )
                return

            _send(struct.pack(">I", 0))  # EOS
            self.wfile.write(b"0\r\n\r\n")
            stats = getattr(iter_synthesize_online, "last_stats", None)
            logger.info(
                "synthesize/stream done frames=%d stats=%s",
                frame_count,
                stats,
            )
        except Exception as exc:
            logger.exception("synthesize/stream failed")
            if not headers_sent:
                self._json(500, {"ok": False, "error": str(exc)[:800]})
            else:
                # Mid-stream failure: close without clean EOS; client errors out.
                try:
                    self.close_connection = True
                except Exception:
                    pass
        finally:
            try:
                from avatar.dystream_synth import release_cuda

                release_cuda()
            except Exception:
                pass
            _bake_lock.release()


def main() -> int:
    parser = argparse.ArgumentParser(description="DyStream model sidecar (GPU 1)")
    parser.add_argument("--host", default=os.environ.get("DYSTREAM_SIDECAR_HOST", "127.0.0.1"))
    parser.add_argument(
        "--port",
        type=int,
        default=int(os.environ.get("DYSTREAM_SIDECAR_PORT", "8766")),
    )
    args = parser.parse_args()

    device = os.environ.get("DYSTREAM_CUDA_DEVICE", "1").strip() or "1"
    os.environ["CUDA_VISIBLE_DEVICES"] = device
    logger.info("CUDA_VISIBLE_DEVICES=%s", device)

    skip_warm = os.environ.get("DYSTREAM_SIDECAR_WARM", "1").strip().lower() in (
        "0",
        "false",
        "no",
    )
    if skip_warm:
        logger.warning("DYSTREAM_SIDECAR_WARM=0 — models load on first request (slow)")
    else:
        warm_models()

    # Ensure talkshow avatar package is importable when cwd is DYSTREAM_ROOT
    if str(_REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(_REPO_ROOT))

    server = ThreadingHTTPServer((args.host, args.port), SidecarHandler)
    logger.info(
        "listening http://%s:%d  GET /health  POST /bake /synthesize/stream  (warmed=%s)",
        args.host,
        args.port,
        _models_warmed,
    )
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        logger.info("shutting down")
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
