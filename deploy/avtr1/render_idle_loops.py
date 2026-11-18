"""Render the idle loops with the running AVTR-1 sidecar, so idle and speaking frames share
one framing (DyStream's loops were cropped differently per role). Each face listens to
someone talking (AVTR-1's listening input): silence alone left them staring down.

  pixi run python deploy/avtr1/render_idle_loops.py <amy_line.wav> <ryan_line.wav>

(any lines in those voices; Claire and Amy listen to Ryan's, Ryan to Amy's)

Writes avatar/assets/loops/<portrait>-idle-avtr1.mp4 (512x512, 25 fps, played forward then
backward so the loop has no seam); the frontend uses them with NEXT_PUBLIC_AVATAR_ENGINE=avtr1.
"""
import json
import struct
import sys
import urllib.request
import wave
from pathlib import Path

import imageio_ffmpeg
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
SIDECAR = "http://127.0.0.1:8766"
SECONDS = 8
ROLES = {"host": "lessac", "commentator": "ryan", "guest": "amy"}


def _stream(portrait: Path, speech: Path, listen: Path, role: str) -> list[np.ndarray]:
    body = {"portrait": str(portrait), "audio": str(speech), "listen": str(listen), "cache_key": role}
    req = urllib.request.Request(f"{SIDECAR}/synthesize/stream", method="POST",
                                 data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    frames = []
    with urllib.request.urlopen(req, timeout=120) as resp:
        def exact(n: int) -> bytes:
            b = b""
            while len(b) < n:
                b += resp.read(n - len(b))
            return b
        while True:
            (n,) = struct.unpack(">I", exact(4))
            if n == 0:
                return frames
            if n == 0xFFFFFFFF:
                (m,) = struct.unpack(">I", exact(4))
                meta = json.loads(exact(m))
                size = int(meta.get("width") or 512)
                continue
            frames.append(np.frombuffer(exact(n), dtype=np.uint8).reshape(size, size, 4)[:, :, :3])


def main() -> None:
    amy_line, ryan_line = (Path(p).resolve() for p in sys.argv[1:3])
    hears = {"host": ryan_line, "commentator": amy_line, "guest": ryan_line}
    out = ROOT / "avatar/assets/loops"
    silence = Path("/tmp") / f"avtr1-silence-{SECONDS}s.wav"
    with wave.open(str(silence), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(16000)
        w.writeframes(b"\0\0" * 16000 * SECONDS)
    for role, name in ROLES.items():
        frames = _stream(ROOT / "avatar/assets/portraits" / f"{name}.png", silence, hears[role], role)
        seq = frames + frames[-2:0:-1]
        size = frames[0].shape[0]
        writer = imageio_ffmpeg.write_frames(str(out / f"{name}-idle-avtr1.mp4"), size=(size, size), fps=25,
                                             codec="libx264", pix_fmt_out="yuv420p", quality=8, macro_block_size=1)
        writer.send(None)
        for f in seq:
            writer.send(np.ascontiguousarray(f).tobytes())
        writer.close()
        print(name, len(seq), "frames →", out / f"{name}-idle-avtr1.mp4")


if __name__ == "__main__":
    main()
