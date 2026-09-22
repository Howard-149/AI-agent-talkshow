# Ghost session (automated talkshow smoke)

Join LiveKit as a **silent human**, let the real worker run welcome + TTS + DyStream, then inject canned guest lines over `talkshow/control`. No laptop mic.

This is an integration harness, not a pytest. The Babel stack (agent worker, vLLM, TTS, optional DyStream) must already be running.

## What it covers

| Path | Real / skipped |
|------|----------------|
| Room join, `wait_for_remote_humans`, locale metadata | Real |
| Session welcome, host grant, panel follow-ups | Real |
| Host + panelist Gemma `complete_text` | Real |
| `speak_panel_line` → Piper/CosyVoice + DyStream | Real |
| Session JSONL (`ghost_human_*`, panel, avatar) | Real |
| Gemma **audio-in** / VAD / mic | Skipped (text inject) |

Opening-only (`--opening-only`) stops after welcome + human floor grant. Default smoke script injects 3Q Test 1 (one human turn, then the panel).

## Run (Babel, recommended)

Worker already up (`python -m agent.main dev`). Use the same room name (default `talkshow-dev`).

```bash
conda activate talkshow
cd ~/AI-agent-talkshow
python -m eval.ghost_session
# or just the opening beat:
python -m eval.ghost_session --opening-only
```

After the run, the client prints a recap if `logs/` is local. Otherwise:

```bash
python -m eval.ghost_session --summarize logs/session-<id>.jsonl
python eval/latency_plot.py logs/session-<id>.jsonl
```

Kill switch on the worker: `TALKSHOW_GHOST_TURNS=0`. Injects are ignored unless the sender identity is `ghost-*` or token metadata has `"ghost": true`.

## Laptop

Needs the RTC SDK (not in `requirements-laptop.txt`):

```bash
pip install livekit pyyaml
python -m eval.ghost_session --room talkshow-dev
```

JSONL still lands on Babel (`LOG_DIR`). Summarize there.

## Script format

See `eval/ghost_session/scripts/smoke.yaml`:

```yaml
locale: en
identity: ghost-guest
turns:
  - text: >
      One guest paragraph the panel should answer.
    topic: short queue topic
```

`--script` can point at another YAML. Raise-hand happens on join so opening FIFO grants the human instead of a host direct-call.

## Record the frontend (no tab)

Laptop only — `talkshow-web` is not synced to Babel. Playwright opens `/custom` as `ghost-viewer` (hidden on the panel), records the stage + transcript, and muxes LiveKit TTS audio into an MP4.

```bash
pip install livekit playwright pyyaml
python -m playwright install chromium   # venv Playwright, not the Node CLI

# talkshow-web already on :3000, or let the harness start it:
python -m eval.ghost_session --record
python -m eval.ghost_session --record --start-frontend
python -m eval.ghost_session --record --opening-only --headed   # watch Chromium
```

Output: `logs/ghost-recordings/ghost-<room>-<ts>.mp4` (printed as `RECORDING=...`). Needs `ffmpeg` on PATH to mux audio; otherwise you get video-only `.webm` plus a `.wav` sidecar.

The wav is LiveKit PCM from the ghost client, not Chromium tab audio. Mux uses the Playwright page-start vs first-audio timestamps so the two clocks line up (ghost joins before Chromium, so leading wav is trimmed).

`--record` is not LiveKit Cloud egress (that path needs a public URL + S3). This is a local headless viewer of the real talkshow UI.
