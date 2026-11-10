"""Resolve a Hugging Face model name to its local snapshot directory when running offline.

With HF_HUB_OFFLINE=1, huggingface_hub checks a cached snapshot for files it expects (e.g. .gitattributes, README.md);
a snapshot copied without them fails that check although config.json and the weights are there. Passing the snapshot
directory instead of the name skips the check. ``local_snapshot`` does that for any "org/name" with a cached snapshot
that has a config.json, and returns the name unchanged otherwise (online runs, paths, models not in the cache).
"""

from __future__ import annotations

import os
import re
from pathlib import Path


def hub_cache() -> Path:
    if os.environ.get("HF_HUB_CACHE"):
        return Path(os.environ["HF_HUB_CACHE"])
    home = os.environ.get("HF_HOME") or os.path.join(os.path.expanduser("~"), ".cache", "huggingface")
    return Path(home) / "hub"


def offline() -> bool:
    return os.environ.get("HF_HUB_OFFLINE", "").strip().lower() in ("1", "true", "yes", "on")


def local_snapshot(name: str) -> str:
    """The cached snapshot directory for ``name`` when offline (the one refs/main names, else the newest with a
    config.json), else ``name``."""
    if not offline() or not re.fullmatch(r"[A-Za-z0-9_.\-]+/[A-Za-z0-9_.\-]+", name or "") or Path(name).exists():
        return name
    repo = hub_cache() / f"models--{name.replace('/', '--')}"
    snaps = repo / "snapshots"
    if not snaps.is_dir():
        return name
    ref = repo / "refs" / "main"
    if ref.is_file():
        main = snaps / ref.read_text().strip()
        if (main / "config.json").is_file():
            return str(main)
    found = sorted((p for p in snaps.iterdir() if (p / "config.json").is_file()), key=lambda p: p.stat().st_mtime)
    return str(found[-1]) if found else name
