# Avatar (DyStream)

Live playout uses Piper PCM for audio and LiveKit video frames from DyStream
(**online stream** by default: generate while playing after a short preroll).

See [DYSTREAM_INTEGRATION.md](../docs/DYSTREAM_INTEGRATION.md).

## Storage

| Path | Content |
|------|---------|
| `avatar/assets/` | Fixed portraits + idle loops |
| `avatar/runtime/clips/` | Temp speech wav / optional legacy `.mp4` |

```bash
bash deploy/init-avatar-dirs.sh
bash deploy/bake-avatar-assets.sh
```

## Babel runtime (recommended: sidecar on GPU 1)

```bash
# .env
TALKSHOW_AVATAR_ENABLED=1
TALKSHOW_AVATAR_LK_VIDEO=1
TALKSHOW_AVATAR_SYNTH=stream          # stream | mp4
TALKSHOW_AVATAR_PREROLL_FRAMES=8      # min floor; dynamic target from gen_fps
TALKSHOW_AVATAR_PREROLL_PROBE_FRAMES=2
TALKSHOW_AVATAR_PREROLL_MARGIN_FRAMES=4
TALKSHOW_AVATAR_PREROLL_MARGIN_FACTOR=1.0
DYSTREAM_SIDECAR_URL=http://127.0.0.1:8766
DYSTREAM_PYTHON=.../envs/dystream/bin/python
# Full list: repo-root .env.example (DyStream avatar section)

# tmux 3 — restart after syncing avatar/
bash deploy/run-dystream-sidecar.sh
curl -s http://127.0.0.1:8766/health
```

## DyStream env

```bash
bash deploy/install-dystream-env.sh
```

Full startup: [deploy/RUN-BABEL.md](../deploy/RUN-BABEL.md).

## Laptop

`NEXT_PUBLIC_DYSTREAM_ENABLED=1`

## Latency logs

| Event | Fields |
|-------|--------|
| `tts_synthesize` | `tts_ms` |
| `avatar_bake` | `synth`, `ttff_ms`, `preroll_ms`, `gen_fps`, … (wall clock; may include lock) |
| `avatar_chunk_play` | `perceived_wait_ms` (1st frame play wait), `preroll_frames`, `gen_fps` |
| `avatar_sync_playout` | `first_chunk_sync_wait_ms`, `perceived_wait_ms_mean` |

Latency plot: **1st frame play** (perceived wait), **preroll frames**, **gen /frame**.
