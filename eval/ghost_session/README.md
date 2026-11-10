# Ghost session (automated talkshow smoke)

Join LiveKit as the **human guest**, let the real worker run the show, and speak scripted guest turns into it. The floor is Think & Action: the guest never waits for a floor grant. Each turn is synthesized with CosyVoice from a recorded voice and played on a microphone track, so the worker hears it through VAD, barge-in and STT like a real guest, and `--record` captures the guest's voice too. Replay scripts and `--text-turns` send the text over `talkshow/control` instead (no audio, no STT, the exact words every run).

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
| Opening, Think & Action floor, panel lines | Real |
| Host + panelist Gemma `complete_text` | Real |
| `speak_panel_line` → Piper/CosyVoice + DyStream | Real |
| Session JSONL (`ghost_human_*`, panel, avatar) | Real |
| Human barge-in (VAD), endpointing, Gemma **audio-in** STT | Real (skipped for text turns) |

When a turn is spoken: a turn with `after_lines` cuts in when the script says (see Script format). A turn without it
waits for the beat to end: `--beat-panel N` lets the show run N panel lines first, with `--quiet-sec` as the fallback.

`--opening-only` joins, waits for the first agent line and holds. The default smoke script speaks 3Q Test 1 right
after the host's opening.

## The guest's voice

Record yourself once (any quiet room, 16-bit wav) and put the files in `eval/ghost_session/voice/` (git-ignored: it is
a person's voice; the repo is public), or point `GHOST_VOICE_DIR` at another folder:

| File | What | Read |
|------|------|------|
| `Neutral.wav` (required) | calm, 12–15 s | "Hi, thanks for having me on the show tonight. I work in software, and I've been at the same company for about six years now. I'm curious to hear what the panel thinks about loyalty at work these days." |
| `Anger.wav`, `Sadness.wav`, `Happiness.wav`, … (optional) | that emotion, 6–10 s | any short line in that mood |

A turn with `emotion: Anger` uses `Anger.wav` as the reference when it exists, else `Neutral.wav`, plus the CosyVoice
instruct for that emotion. Synthesis needs the CosyVoice sidecar (`COSYVOICE_SIDECAR_URL`), and the sidecar must be
able to read the reference path, so on Babel keep the folder where compute nodes see it (e.g. `~/ghost_voice`, mode
700). Turns are synthesized before joining and cached in `logs/ghost-voice-cache/`. A turn can name a recording
instead with `audio: path/to.wav`.

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

The laptop cannot reach the CosyVoice sidecar, so spoken turns need their audio already in the cache: run the script
once on Babel (it fills `logs/ghost-voice-cache/`) and copy that folder over, or name recordings with `audio:`, or use
`--text-turns`.

## Script format

See `eval/ghost_session/scripts/smoke.yaml`:

```yaml
locale: en
identity: ghost-guest
turns:
  - text: >
      One guest paragraph the panel should answer.
    after_lines: 1      # cut in once this many agent lines started since the guest last spoke
    when: line_end      # line_end: after that line finishes | mid_line: after_sec into it
    after_sec: 0
    emotion: Anger      # CosyVoice delivery; default Neutral
```

All four timing fields are optional; without `after_lines` the turn waits for the beat to end (above). A timed turn
that has waited `max_wait_sec` (default 180) is spoken anyway. `--script` can point at another YAML.

Replay scripts (the listening test, `eval/README.md`) add a saved show state to the first turn:

```yaml
prefix_file: eval/data/minimal_pairs/prefix/tipping.json   # history + mood to swap in before turn 1
persona_swap: true                                         # optional: panelists swap personality traits
flags: {TALKSHOW_ADDRESSEE_FLOOR: "1"}                     # optional: dialogue-quality flags for this session
```

The worker applies them only with `TALKSHOW_GHOST_REPLAY=1`, and then also logs a `history_snapshot` event before each
ghost turn. With the Think & Action floor the prefix is applied when the turn is committed (after the barge-in
rollback, between show beats), and the floor counts saved in the snapshot (`floor`) are restored with it. With the flag
off the prefix is ignored. The client re-sends a turn that got no reply within 30 s; the
worker drops a ghost turn whose text equals the previous ghost turn. `--stop-after-panel N` leaves after N panel lines
following the last turn.

## Record the frontend (no tab)

Laptop only — `talkshow-web` is not synced to Babel. Playwright opens `/custom` as `ghost-viewer` (hidden on the panel), records the stage + transcript, and muxes the show audio plus the guest's spoken turns into an MP4.

```bash
pip install livekit playwright pyyaml
python -m playwright install chromium   # venv Playwright, not the Node CLI

# talkshow-web already on :3000, or let the harness start it:
python -m eval.ghost_session --record
python -m eval.ghost_session --record --start-frontend
python -m eval.ghost_session --record --opening-only --headed   # watch Chromium
python -m eval.ghost_session --record --text-turns   # no voice reference / sidecar needed
```

Output: `logs/ghost-recordings/ghost-<room>-<ts>.mp4` (printed as `RECORDING=...`). Needs `ffmpeg` on PATH to mux audio; otherwise you get video-only `.webm` plus a `.wav` sidecar.

The wav is LiveKit PCM from the ghost client, not Chromium tab audio. Mux uses the Playwright page-start vs first-audio timestamps so the two clocks line up (ghost joins before Chromium, so leading wav is trimmed).

`--record` is not LiveKit Cloud egress (that path needs a public URL + S3). This is a local headless viewer of the real talkshow UI.
