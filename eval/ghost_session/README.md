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
able to read the reference path, so on Babel keep the folder where compute nodes see it. Turns are synthesized before
joining and cached in `logs/ghost-voice-cache/`. A turn can name a recording instead with `audio: path/to.wav`.

### Shared voice on Babel

Howard's references are on Babel at `/data/user_data/hsuanhal/ghost_voice` (readable by everyone, compute nodes
only), so nobody has to record their own to run a ghost session:

```bash
GHOST_VOICE_DIR=/data/user_data/hsuanhal/ghost_voice
```

```
ghost_voice/
├── Neutral.wav, Anger.wav, Happiness.wav, Sadness.wav, Surprise.wav   # references
└── demo/
    ├── pad_swing_demo.yaml   # pad_swing with pre-synthesized turns (audio: → the two wavs)
    └── b105bc….wav, ed37ed….wav
```

`demo/pad_swing_demo.yaml` is `scripts/pad_swing.yaml` with each turn's audio already synthesized: an angry guest
cuts in after the host's opening (`line_end`), then a relieved one cuts into the 4th agent line 3 s in
(`mid_line`). It needs no CosyVoice sidecar and plays the same audio every run, so it is the demo script:

```bash
python -m eval.ghost_session --record --stop-after-panel 6 \
  --script /data/user_data/hsuanhal/ghost_voice/demo/pad_swing_demo.yaml
```

From a laptop, copy `demo/` over and point the two `audio:` paths at the copies.

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

Without a tunnel the laptop cannot reach the CosyVoice sidecar, so spoken turns need their audio already in the cache:
run the script once on Babel (it fills `logs/ghost-voice-cache/`) and copy that folder over, or name recordings with
`audio:`, or use `--text-turns`. With an SSH tunnel to the sidecar, set `COSYVOICE_SIDECAR_URL` and
`GHOST_VOICE_REMOTE_DIR` (see Live turns).

## Script format

See `eval/ghost_session/scripts/smoke.yaml`:

```yaml
locale: en
identity: ghost-guest
name: Howard        # the guest's seat + transcript name (default Guest; --name overrides)
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

### Live turns (LLM-written)

A turn can leave out `text` and set `generate: true`: its words are written during the show by an LLM from what the
panel actually said, then spoken in the guest's voice like any other turn. Timing fields work as above.

```yaml
topic: Is AI Making Us Smarter or More Dependent?   # the show topic (prompt context only)
persona: >                                          # who the guest is and how he talks
  Howard, a grad student who uses AI tools daily ...
turns:
  - generate: true
    goal: push back on whoever is talking right now  # optional hint for this turn
    after_lines: 4
    when: mid_line
    after_sec: 3
```

The client keeps the show transcript from the UI events (`transcript`, trimmed to `heard_text` on an `interrupt`,
dropped on a `line_end` with `played: false`) plus the guest's own turns. As soon as the trigger line is announced
(its transcript arrives before its audio) it sends persona + topic + the last 12 lines + `goal` (+ for `mid_line`,
only the first words of the line being cut into) to the LLM, which answers with an emotion
(Neutral/Happiness/Anger/Sadness/Surprise) and a 25–70-word spoken line; it retries once on an unparseable reply and
falls back to Neutral. The line is synthesized right away, so it is usually ready by the cut-in point; if not, it is
spoken as soon as it is. The log shows `generated in …s`, `TTS …s` and `ready …s before/after the cut-in point`.
With `--text-turns` the generated text is sent as text.

| Env | Default | |
|-----|---------|---|
| `GHOST_LLM_BASE_URL` | `http://127.0.0.1:18000/v1` | OpenAI-compatible chat endpoint |
| `GHOST_LLM_MODEL` | first id from `/models` | |
| `GHOST_LLM_API_KEY` | `EMPTY` | |
| `GHOST_VOICE_REMOTE_DIR` | unset | the sidecar's copy of the references (see below) |

The sidecar opens the reference by path. When it runs on another machine (a laptop with an SSH tunnel to Babel),
set `GHOST_VOICE_REMOTE_DIR` to the folder with the same `<Emotion>.wav` files on that machine: the reference is still
chosen from the local `GHOST_VOICE_DIR`, and the sidecar gets `GHOST_VOICE_REMOTE_DIR/<same name>`.

```bash
COSYVOICE_SIDECAR_URL=http://127.0.0.1:18767 GHOST_VOICE_REMOTE_DIR=/path/on/sidecar/node/voice \
  python -m eval.ghost_session --record --stop-after-panel 6 --script path/to/live.yaml
```

### Reactive guest

With `reactive: true` the guest takes part like a panelist after the turns above (which become seeds, e.g. one
`generate: true` turn after the host's opening). On every agent line announced (its transcript arrives before its
audio):

- **Floor handed to him** (`addressing.py`: a vocative of his name, or "our guest", tied to a question or an
  invitation like "the floor is yours" / "what do you think" in the same or next sentence, at the end of the line): he
  always answers, 0.4–0.8 s after the line ends, even if the next line has started.
- **Floor handed to someone else** ("…Amy, how do you see it?"; names from `panel_roster`, the transcript speakers,
  or a vocative-looking name): he stays silent, with no LLM call, and leaves the gap to them.
- **Asked in passing** (a question tied to his name mid-line): the LLM decides, outside the cooldown.
- **Otherwise** (a bare "Howard, …", a mention, nothing): nothing within `min_lines_between_turns` lines of his last
  turn (no LLM call); after that, one LLM call decides `silent` (default) / `respond` (after the line) / `cut_in`
  (rare, ~2.5 s into the line; at most two) and writes the words. A reply is spoken in the gap after the line; if the
  next line is already playing by then, it is dropped and the newest line is considered instead. A reply is also
  dropped as stale when more than one newer line was announced, when the next line has played 2 s without this
  line's end being seen, or 15 s after its line ended.

The words are written and synthesized as soon as the line is announced. He never speaks twice without an agent line
in between. After `max_turns` (seeds included) or `max_minutes` (from the first agent line) he stops reacting and
leaves after `--stop-after-panel N` panel lines. Every line considered is logged as `react line N <speaker>
addressed=… → <decision> gen=… tts=… latency=…` (latency: line end → his audio).

```yaml
reactive: true
max_turns: 12               # a cap; time normally ends the show
min_lines_between_turns: 3
max_minutes: 9.5
```

**Ending.** When the budget is spent he stops reacting and, at the next natural gap (a line has ended, nothing is
playing, and the floor was not just handed to someone else), says a short LLM-written goodbye (thanks, a one-sentence
takeaway, "I have to head out"; a script `goodbye:` gives fixed words). Right after his audio the ghost sends
`{"type": "wrap_up"}` on `talkshow/control`; the worker moves to the host's closing line (step `close_round`). The ghost
leaves (and the recording stops) once that line has ended and no agent audio has played for 3 s; without a
`close_round` within 90 s, once nothing has played or been announced for 5 s. It never leaves while a line plays.
In reactive mode this replaces `--stop-after-panel`; scripted runs get it with `wrap_up: true` or `--wrap-up` (after the
last turn's beat, `--stop-after-panel` lines or quiet).

Scripted `line_end` turns also hold back when their trigger line hands the floor to someone else: they wait for that
person's line to end (or 20 s without them speaking) and cut in after it.

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
