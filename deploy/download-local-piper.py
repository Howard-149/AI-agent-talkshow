"""Download the original three English Piper voices for an audio-only demo."""
from pathlib import Path
import shutil
import ssl
import urllib.request

import certifi

destination = Path(__file__).resolve().parents[1] / "models" / "piper"
destination.mkdir(parents=True, exist_ok=True)
context = ssl.create_default_context(cafile=certifi.where())
base = "https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US"
for voice in ("lessac", "ryan", "amy"):
    for suffix in (".onnx", ".onnx.json"):
        name = f"en_US-{voice}-medium{suffix}"
        target = destination / name
        if target.is_file() and target.stat().st_size > 0:
            print(f"Already downloaded: {name}", flush=True)
            continue
        print(f"Downloading: {name}", flush=True)
        request = urllib.request.Request(f"{base}/{voice}/medium/{name}")
        temporary = target.with_name(f"{target.name}.part")
        with urllib.request.urlopen(request, context=context, timeout=120) as response:
            with temporary.open("wb") as output:
                shutil.copyfileobj(response, output)
        temporary.replace(target)
        print(f"Ready: {name} ({target.stat().st_size:,} bytes)", flush=True)
