#!/usr/bin/env python3
"""AVTR-1 avatar sidecar with the DyStream sidecar's HTTP interface, so the agent can use
either (same DYSTREAM_SIDECAR_URL, no agent change).

  POST /synthesize/stream  {"portrait": path, "audio": wav path, "cache_key": role, ...}
    → chunked body: uint32_be 0xFFFFFFFF + uint32_be len + meta JSON,
      then uint32_be len + RGBA frame (repeated), then uint32_be 0 (EOS)
  GET /health → {"ok": true, "models_warmed": true, ...}

AVTR-1 (github.com/avaturn-live/avtr-1) renders 25 fps in 5-frame chunks from 16 kHz audio.
Its frames are 16:9 with the head in the middle; the sidecar sends the centre square,
scaled to TALKSHOW_AVATAR_VIDEO_WIDTH (512), like DyStream's square frames.

Run inside the AVTR-1 pixi environment, with AVTR1_LOCAL_STORAGE set to the storage whose
TensorRT engines were built for this GPU, and the role portraits copied into its
avatars_artifacts/reference_frames as <avatar id>.png:

  cd $AVTR1_REPO && pixi run python ~/AI-agent-talkshow/avatar/avtr1_sidecar.py --port 8766
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import struct
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

import numpy as np

logger = logging.getLogger("avtr1_sidecar")

SAMPLE_RATE = 16000
FPS = 25
# Role → AVTR-1 avatar id, for a portrait whose file name is not an avatar id
# (the host's portrait is lessac.png; it is registered as "claire").
ROLE_AVATAR = {"host": "claire", "commentator": "ryan", "guest": "amy"}

_lock = threading.Lock()  # one GPU: one stream at a time, like the DyStream sidecar
_pipeline = None
_registry: dict = {}
_states: dict[str, object] = {}  # avatar id → motion state, so sentences join up
_warm_s: float | None = None


def _load(avatar_ids: list[str]) -> None:
    global _pipeline, _registry, _warm_s
    from avtr1_renderer.pipeline import Pipeline

    t0 = time.monotonic()
    _pipeline, _registry = Pipeline.from_artifacts(avatar_ids=avatar_ids)
    _warm_s = time.monotonic() - t0
    logger.info("AVTR-1 loaded avatars=%s in %.1fs", sorted(_registry), _warm_s)


def _avatar_id(portrait: str, cache_key: str) -> str:
    stem = Path(portrait).stem
    if stem in _registry:
        return stem
    if cache_key in ROLE_AVATAR and ROLE_AVATAR[cache_key] in _registry:
        return ROLE_AVATAR[cache_key]
    raise ValueError(f"no AVTR-1 avatar for portrait={portrait} role={cache_key}")


def _load_audio(path: str) -> np.ndarray:
    import soundfile as sf
    import soxr

    audio, sr = sf.read(path, dtype="float32", always_2d=True)
    audio = audio.mean(axis=1)
    if sr != SAMPLE_RATE:
        audio = soxr.resample(audio, sr, SAMPLE_RATE, quality="HQ")
    return audio.astype(np.float32)


def _square_rgba(frame_data: np.ndarray, size: int) -> bytes:
    """I420 (1.5H, W) → centre square → size×size RGBA bytes."""
    import cv2

    rgba = cv2.cvtColor(frame_data, cv2.COLOR_YUV2RGBA_I420)
    h, w = rgba.shape[:2]
    side = min(h, w)
    x0, y0 = (w - side) // 2, (h - side) // 2
    square = rgba[y0 : y0 + side, x0 : x0 + side]
    if side != size:
        square = cv2.resize(square, (size, size), interpolation=cv2.INTER_AREA)
    return np.ascontiguousarray(square).tobytes()


def _chunks(audio: np.ndarray) -> list[np.ndarray]:
    mg = _pipeline._motion_generator
    window = (mg.chunk_size + mg.future_size) * mg.frame_len + mg.audio_shift
    step = mg.chunk_size * mg.frame_len
    n = max(1, (len(audio) + step - 1) // step)
    out = []
    for i in range(n):
        piece = audio[i * step : i * step + window]
        if len(piece) < window:
            piece = np.pad(piece, (0, window - len(piece)))
        out.append(piece)
    return out


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt: str, *args: object) -> None:
        logger.info("%s - %s", self.address_string(), fmt % args)

    def _json(self, code: int, body: dict) -> None:
        raw = json.dumps(body).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def do_GET(self) -> None:
        if (urlparse(self.path).path.rstrip("/") or "/") in ("/", "/health"):
            self._json(200, {
                "ok": True,
                "models_warmed": _pipeline is not None,
                "warmup_sec": round(_warm_s, 1) if _warm_s else None,
                "service": "avtr1_sidecar",
                "apis": ["synthesize/stream"],
                "synth_mode": "online_stream",
                "avatars": sorted(_registry),
            })
            return
        self._json(404, {"ok": False, "error": "not found"})

    def do_POST(self) -> None:
        if urlparse(self.path).path.rstrip("/") != "/synthesize/stream":
            self._json(404, {"ok": False, "error": "not found"})
            return
        try:
            body = json.loads(self.rfile.read(int(self.headers.get("Content-Length") or 0)))
            avatar_id = _avatar_id(str(body["portrait"]), str(body.get("cache_key") or ""))
            speech = _load_audio(str(body["audio"]))
            # Optional: what the others are saying meanwhile (AVTR-1's listening input).
            listen = _load_audio(str(body["listen"])) if body.get("listen") else None
        except Exception as exc:
            self._json(400, {"ok": False, "error": str(exc)[:500]})
            return
        size = int(os.environ.get("TALKSHOW_AVATAR_VIDEO_WIDTH", "512"))
        with _lock:
            self._stream(avatar_id, speech, size, listen)

    def _send(self, blob: bytes) -> None:
        self.wfile.write(f"{len(blob):X}\r\n".encode("ascii"))
        self.wfile.write(blob)
        self.wfile.write(b"\r\n")
        self.wfile.flush()

    def _stream(
        self, avatar_id: str, speech: np.ndarray, size: int, listen: np.ndarray | None = None
    ) -> None:
        from avtr1_renderer.types import Chunk, RenderOptions

        avatar = _registry[avatar_id]
        expected = int(np.ceil(len(speech) / SAMPLE_RATE * FPS))
        self.send_response(200)
        self.send_header("Content-Type", "application/x-talkshow-rgba-stream")
        self.send_header("Transfer-Encoding", "chunked")
        self.end_headers()
        meta = json.dumps({
            "ok": True, "width": size, "height": size, "fps": FPS,
            "expected_frames": expected, "audio_feat_ms": 0, "stream": "online",
        }).encode("utf-8")
        options = RenderOptions(pixel_format="yuv_i420", bg_id="plain_white", stream_frames=True)
        sent = 0
        t0 = time.monotonic()
        state = _states.get(avatar_id)
        try:
            self._send(struct.pack(">I", 0xFFFFFFFF) + struct.pack(">I", len(meta)) + meta)
            if listen is None:
                listen = np.zeros_like(speech)
            listen = np.pad(listen, (0, max(0, len(speech) - len(listen))))[: len(speech)]
            for piece, heard in zip(_chunks(speech), _chunks(listen)):
                state, frames = _pipeline.process_chunk(
                    avatar, Chunk(audio_speech=piece, audio_listen=heard), state, options
                )
                for frame in frames:
                    if sent >= expected:
                        continue  # the last chunk runs past the audio
                    rgba = _square_rgba(frame.data, size)
                    self._send(struct.pack(">I", len(rgba)) + rgba)
                    sent += 1
            _states[avatar_id] = state
            self._send(struct.pack(">I", 0))
            self.wfile.write(b"0\r\n\r\n")
            logger.info("stream done avatar=%s frames=%d in %.2fs", avatar_id, sent, time.monotonic() - t0)
        except (BrokenPipeError, ConnectionResetError):
            # The agent hung up (the line was cut); keep the motion state for the next one.
            _states[avatar_id] = state
            logger.info("client hung up after %d frames", sent)


def main() -> int:
    parser = argparse.ArgumentParser(description="AVTR-1 sidecar (DyStream-compatible API)")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8766)
    parser.add_argument("--avatars", default="claire,ryan,amy")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    _load([a.strip() for a in args.avatars.split(",") if a.strip()])
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    logger.info("listening on http://%s:%d", args.host, args.port)
    server.serve_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
