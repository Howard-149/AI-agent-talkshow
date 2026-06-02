# AI-Agent-Talkshow

Voice-in / voice-out talk show: **3 AI roles** (Host / Guest / Commentator), **Gemma 4 audio-in**, **Piper TTS**, **LiveKit Agents**, custom frontend in **`talkshow-web/`** (fork of [LiveKit Meet](https://github.com/livekit-examples/meet)).

CMU capstone — multi-participant dialogue with orchestrated multi-persona agents (single RTC participant + handoff + virtual panel UI).

**Roadmap & collaboration:** [ROADMAP.md](ROADMAP.md) · [CONTRIBUTING.md](CONTRIBUTING.md)

---

## End-to-end flow

```
Laptop browser (talkshow-web)
    │  WebRTC + mic + token from api/tokens.py
    ▼
LiveKit Cloud  (LIVEKIT_URL)
    │  dispatch job → room e.g. talkshow-dev
    ▼
Babel GPU: agent worker  ←→  vLLM (Gemma) + Piper (per role)
```

- **Frontend:** `talkshow-web/` — virtual panel (Lessac / Ryan / Amy), transcript via data channel.
- **Agent / vLLM:** Same GPU node; one worker joins the room; roles hand off in-process.

---

## Architecture (current)

| Piece | Role |
|-------|------|
| **Single agent participant** | One voice track; `session.update_agent()` + Piper swap per role |
| **TurnController** | Scenario-driven order (`panel_round_robin` in `config/scenarios/default.yaml`) |
| **GemmaAudioSTT** | One vLLM multimodal call per human utterance → `[heard]` / `[reply]` |
| **StoredReplyLLM** | Replays cached host reply to TTS (no second LLM call) |
| **Panel round** | Ryan → Amy text generation via `complete_text` + `speak_panel_line` |
| **Shared history** | `TalkShowData.show_history` (trimmed by `TALKSHOW_HISTORY_MAX_LINES`) |
| **UI events** | `talkshow/ui`: `panel_roster`, `role_active`, `transcript` |

```
User audio → VAD/turn → GemmaAudioSTT → StoredReplyLLM → Piper TTS → room
          → panel round (Ryan/Amy) → UI events → talkshow-web
```

---

## Quick start (laptop)

```bash
cp .env.example .env          # LiveKit keys + paths
python3 -m venv .venv && source .venv/bin/activate
pip install livekit-api python-dotenv

python api/tokens.py --room talkshow-dev --identity human-host

cd talkshow-web && pnpm install && pnpm dev
# → http://localhost:3000 — paste URL + TOKEN
```

Agent runs on Babel (see below). After agent changes: `./scripts/sync-to-babel.sh` and restart worker.

---

## Babel (GPU node)

```bash
conda activate talkshow
cd ~/AI-agent-talkshow
pip install -r requirements.txt
bash deploy/download-piper-voices.sh lessac amy ryan
bash deploy/download-livekit-agent-models.sh

cp .env.example .env    # keys, PIPER_MODEL_PATH, TALKSHOW_TURN_DETECTOR_CACHE

# tmux 1 — vLLM with audio
bash deploy/vllm-gemma4-audio.sh

# tmux 2 — agent worker (same node as vLLM)
source .env && python -m agent.main dev
```

### Common cluster logs

| Message | Cause | Fix |
|---------|--------|-----|
| `pthread_setaffinity_np failed` (ONNX) | Slurm CPU cgroup vs ORT affinity | `TALKSHOW_ORT_NUM_THREADS=1` in `.env` |
| Turn detector missing | ONNX not in private cache | `source .env && bash deploy/download-livekit-agent-models.sh` |
| `memory usage is high` | vLLM + agent + turn-detector on one node | OK to ignore; or `TALKSHOW_TURN_DETECTOR=vad` |

Verify ORT threads on agent startup:

```bash
source .env && python -m agent.main dev 2>&1 | head -20
# Expect: ONNX/thread env ORT_NUM_THREADS=1 (Slurm-safe)
```

---

## Configuration

| File | Purpose |
|------|---------|
| `.env` | LiveKit, vLLM, Piper paths, `TALKSHOW_*` tuning |
| `config/scenarios/default.yaml` | `turn_control.mode`, speaker `order`, `listen_role` |
| `config/personas/*.yaml` | Role names, instructions, Piper voice |
| `SCENARIO_PATH` | Override scenario file |
| `TALKSHOW_HISTORY_MAX_LINES` | Shared transcript trim (default 48 lines) |

Turn modes:

- **`panel_round_robin`** (default): human → host → Ryan → Amy → host close → human
- **`rotate_after_user`**: legacy rotation after each user turn
- **`manual_only`**: `[handoff:role]` tags only

---

## talkshow-web

Fork of LiveKit Meet with talkshow panel + transcript. Env: **repo root `.env`** (via `next.config.js`).

See [talkshow-web/README.md](talkshow-web/README.md).

---

## Repo layout (git)

```
agent/           LiveKit worker, supervisor/, ui_events, panel speech
talkshow-web/    Next.js frontend
config/          multimodal.yaml, personas/, scenarios/
deploy/          vLLM, Piper, model download scripts
api/             token helper (laptop)
ROADMAP.md       Planned work
CONTRIBUTING.md  Collaboration guide
```

Not in git (local): `.cursor/*` except `.cursor/rules/`, `docs/`, `scripts/sync-to-babel.sh`

---

## Phase status

| Phase | Status |
|-------|--------|
| **0** | Voice E2E: Gemma audio-in → Piper |
| **1** | Multi-persona handoff + scenario YAML |
| **2** | Scene/floor awareness + shared history |
| **3** | Custom frontend + UI data channel |
| **Next** | Debate scenario, host-moderated turns, avatars — see [ROADMAP.md](ROADMAP.md) |

---

## Smoke-test vLLM audio (GPU node)

```bash
python -c "
import base64, httpx, sys
wav = open(sys.argv[1],'rb').read()
b64 = base64.standard_b64encode(wav).decode()
r = httpx.post('http://127.0.0.1:8000/v1/chat/completions', json={
  'model': 'google/gemma-4-E4B-it',
  'messages': [{'role':'user','content':[
    {'type':'audio_url','audio_url':{'url': f'data:audio/wav;base64,{b64}'}},
    {'type':'text','text': 'Respond with [heard]: ... and [reply]: ...'}
  ]}],
  'max_tokens': 512
}, headers={'Authorization':'Bearer EMPTY'}, timeout=120)
print(r.json()['choices'][0]['message']['content'])
" sample.wav
```
