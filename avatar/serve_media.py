#!/usr/bin/env python3
"""HTTP server for avatar assets + runtime clips (Babel — no talkshow-web)."""
from __future__ import annotations

import argparse
import os
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from avatar.paths import avatar_assets_dir, avatar_clips_dir


class AvatarMediaHandler(SimpleHTTPRequestHandler):
    assets_root: Path = avatar_assets_dir()
    clips_root: Path = avatar_clips_dir()

    def translate_path(self, path: str) -> str:
        clean = path.split("?", 1)[0].split("#", 1)[0]
        if clean.startswith("/assets/"):
            rel = clean[len("/assets/") :]
            return str((self.assets_root / rel).resolve())
        if clean.startswith("/clips/"):
            rel = clean[len("/clips/") :]
            return str((self.clips_root / rel).resolve())
        if clean in ("/assets", "/clips"):
            return str(self.assets_root if clean == "/assets" else self.clips_root)
        return super().translate_path(path)

    def end_headers(self) -> None:
        self.send_header("Access-Control-Allow-Origin", "*")
        super().end_headers()

    def log_message(self, fmt: str, *args: object) -> None:
        if args and str(args[0]).startswith("GET /"):
            super().log_message(fmt, *args)


def main() -> None:
    parser = argparse.ArgumentParser(description="Serve avatar assets + clips")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=int(os.environ.get("AVATAR_MEDIA_PORT", "8765")))
    args = parser.parse_args()

    AvatarMediaHandler.assets_root = avatar_assets_dir()
    AvatarMediaHandler.clips_root = avatar_clips_dir()
    AvatarMediaHandler.assets_root.mkdir(parents=True, exist_ok=True)
    AvatarMediaHandler.clips_root.mkdir(parents=True, exist_ok=True)

    assets = AvatarMediaHandler.assets_root
    clips = AvatarMediaHandler.clips_root
    print(f"assets → http://{args.host}:{args.port}/assets/  ({assets})")
    print(f"clips  → http://{args.host}:{args.port}/clips/   ({clips})")

    server = ThreadingHTTPServer((args.host, args.port), AvatarMediaHandler)
    server.serve_forever()


if __name__ == "__main__":
    main()
