# Ghost session (automated talkshow smoke)

Join LiveKit as a **silent human**, let the real worker run welcome + TTS + DyStream, then inject canned guest lines over `talkshow/control`. No laptop mic.

This is an integration harness, not a pytest. The Babel stack (agent worker, vLLM, TTS, optional DyStream) must already be running.

## Common commands

Typical loop: start the Babel stack, run the ghost from the laptop, then read the logs on Babel.

```bash
# 1. Babel login node: (re)start the stack (vLLM + CosyVoice + DyStream + agent worker)
cd ~/AI-agent-talkshow
squeue -u $USER                                   # find the old job id
scancel <old-jobid>
sbatch deploy/slurm-talkshow-3gpu.sh              # add --exclude=<node> to skip a bad node
grep "registered worker" slurm-logs/slurm-talkshow-<jobid>.out   # ready when this prints
deploy/talkshow-ctl.sh restart agent              # after a code / .env change: no resubmit

# 2. Laptop: run a ghost session against it
python -m eval.ghost_session                                               # smoke.yaml (polite debate)
python -m eval.ghost_session --script eval/ghost_session/scripts/pad_swing.yaml   # emotion swing
python -m eval.ghost_session --record --start-frontend \
    --script eval/ghost_session/scripts/pad_swing.yaml                     # + MP4 in logs/ghost-recordings/

# 3. Babel: read the newest session log
python -m eval.ghost_session --summarize logs/session-<id>.jsonl   # turn-by-turn recap
python eval/pad_report.py --last 1                                 # [pad] compliance + PAD trajectory
python eval/gap_breakdown.py logs/session-<id>.jsonl               # where dead air goes (LLM / TTS / avatar)
tail -f slurm-logs/slurm-talkshow-<jobid>.out                      # live worker log
```

Scripts: `scripts/smoke.yaml` (hiring-freeze debate, PAD stays near 0) and
`scripts/pad_swing.yaml` (public humiliation → relief; exercises negative and recovering emotion).

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

Worker already up (`python -m agent.main dev`). Each run uses a new room (`talkshow-ghost-<epoch>`) unless `--room` or `TALKSHOW_ROOM` is set: a reused name can hit the empty room LiveKit keeps open after the last session (no agent), and a shared name collides with other users' ghosts (`DuplicateIdentity`). If the worker sets `TALKSHOW_AGENT_NAME`, set the same value where the ghost runs.

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
python -m eval.ghost_session            # new room per run; needs the Babel TALKSHOW_AGENT_NAME if set
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
