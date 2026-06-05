# AI-Agent-Talkshow

Voice-in / voice-out talk show: **3 AI roles** (Host / Guest / Commentator), **Gemma 4 audio-in**, **Piper TTS**, **LiveKit Agents**, custom frontend in **`talkshow-web/`**.

CMU capstone — single RTC agent + role handoff + virtual panel UI (Lessac / Ryan / Amy).

**Collaboration:** [ROADMAP.md](ROADMAP.md) · [CONTRIBUTING.md](CONTRIBUTING.md)

---

## Who runs what

| Machine | You install | You run |
|---------|-------------|---------|
| **Laptop** | `requirements-laptop.txt` + Node/pnpm | `api/tokens.py`, `talkshow-web` (`pnpm dev`) |
| **Babel GPU node** | `requirements.txt` (conda `talkshow`, **Python 3.11**) | vLLM + `python -m agent.main dev` |
| **LiveKit Cloud** | (project keys in `.env`) | WebRTC room — no self-hosted server |

You do **not** need vLLM, Piper, or the full agent stack on your laptop.

```
Laptop (talkshow-web + token)
    │  WebRTC
    ▼
LiveKit Cloud  (LIVEKIT_URL, room e.g. talkshow-dev)
    │
    ▼
Babel GPU  —  vLLM (Gemma) + agent worker + Piper (3 voices)
```

---

## Teammate setup — laptop (one-time)

### 1. Clone and Python venv

```bash
git clone <repo-url> AI-agent-talkshow
cd AI-agent-talkshow

python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

pip install -r requirements-laptop.txt
```

`requirements-laptop.txt` is only:

- `livekit-api` — mint room tokens (`api/tokens.py`)
- `python-dotenv` — read repo root `.env`

Do **not** run `pip install -r requirements.txt` on a laptop unless you are debugging the agent locally with a GPU.

### 2. Environment file (repo root)

```bash
cp .env.example .env
```

Edit **at minimum** on the laptop:

| Variable | Where to get it |
|----------|-----------------|
| `LIVEKIT_URL` | [LiveKit Cloud](https://cloud.livekit.io) → Project → Settings |
| `LIVEKIT_API_KEY` | same |
| `LIVEKIT_API_SECRET` | same |

Laptop `.env` does **not** need Piper paths or `VLLM_*` unless you are also developing on Babel. The frontend reads this same file via `talkshow-web/next.config.js` — **no** `talkshow-web/.env.local`.

### 3. Frontend

```bash
cd talkshow-web
pnpm install    # or: npm install
pnpm dev
# → http://localhost:3000
```

### 4. Token + connect

In **another terminal** (repo root, venv active):

```bash
source .venv/bin/activate
python api/tokens.py --room talkshow-dev --identity your-name
```

Copy `LIVEKIT_URL` + `TOKEN` into the **Talkshow (token)** tab, or use the printed values. **Room name must match** what the Babel agent uses (default `talkshow-dev`).

### 5. Laptop checklist

- [ ] `pip install -r requirements-laptop.txt` (not `requirements.txt`)
- [ ] Root `.env` has LiveKit URL + key + secret
- [ ] `pnpm dev` in `talkshow-web/`
- [ ] Agent worker running on Babel (see below)
- [ ] Same room name on token and cluster

---

## Teammate setup — Babel GPU node (one-time)

Only needed if you run or debug the **agent / vLLM**. Coordinate so one person’s interactive GPU session runs the shared dev stack, or use your own `$USER` paths below.

### 1. Get code on Babel

Clone the GitHub repo on the GPU node (or `git pull` after you push from your laptop):

```bash
git clone <repo-url> ~/AI-agent-talkshow
cd ~/AI-agent-talkshow
```

`.env` is not in git — copy your filled `.env` to Babel separately (scp, or recreate from `.env.example`).

### 2. Conda env + Python deps

On the GPU node, use **Python 3.11** (matches vLLM cu129 wheel and LiveKit Agents):

```bash
conda create -n talkshow python=3.11 -y   # skip if env already exists
conda activate talkshow
cd ~/AI-agent-talkshow
pip install -r requirements.txt
python --version   # should print 3.11.x
```

`requirements.txt` includes LiveKit Agents, Piper, vLLM (cu129 wheel), etc. **Laptop uses `requirements-laptop.txt` instead** (any recent Python 3.10+ for venv is fine).

### 3. Download models (private paths under `/data/user_data/$USER/`)

Replace `$USER` with your Babel username everywhere.

```bash
# Piper — three voices for Lessac / Amy / Ryan
bash deploy/download-piper-voices.sh --dest /data/user_data/$USER/piper lessac amy ryan

# Turn-detector + Silero ONNX (agent VAD / end-of-turn)
source .env   # after step 4, or export TALKSHOW_TURN_DETECTOR_CACHE first
bash deploy/download-livekit-agent-models.sh
```

Gemma weights: use your team’s existing vLLM model path, or set `VLLM_MODEL_PATH` in `.env` (see deploy script `deploy/vllm-gemma4-audio.sh`).

### 4. Babel `.env`

```bash
cp .env.example .env
```

Edit with **your** `$USER` (not a teammate’s). Critical paths:

#### LiveKit (same project as laptops)

```
LIVEKIT_URL=wss://….
LIVEKIT_API_KEY=…
LIVEKIT_API_SECRET=…
```

#### Piper — one ONNX file per role

| Role | Persona | `config/personas/*.yaml` | `.env` variable | Example path |
|------|---------|---------------------------|-----------------|--------------|
| Host | Lessac | `host.yaml` → `en_US-lessac-medium` | `PIPER_MODEL_PATH` | `/data/user_data/$USER/piper/en_US-lessac-medium.onnx` |
| Guest | Amy | `guest.yaml` → `en_US-amy-medium` | `PIPER_MODEL_PATH_GUEST` | `/data/user_data/$USER/piper/en_US-amy-medium.onnx` |
| Commentator | Ryan | `commentator.yaml` → `en_US-ryan-medium` | `PIPER_MODEL_PATH_COMMENTATOR` | `/data/user_data/$USER/piper/en_US-ryan-medium.onnx` |

Each `.onnx` file needs a sibling config JSON, e.g. `en_US-lessac-medium.onnx.json` (the download script creates these).

If `PIPER_MODEL_PATH_GUEST` / `_COMMENTATOR` are unset, those roles fall back to the host voice.

#### vLLM (same GPU node as agent)

```
VLLM_BASE_URL=http://127.0.0.1:8000/v1
VLLM_MODEL=google/gemma-4-E4B-it
# optional local weights:
# VLLM_MODEL_PATH=/data/user_data/$USER/models/gemma-4-E4B-it
```

#### Turn-detector cache (per-user, not shared `/data/hf_cache`)

```
TALKSHOW_TURN_DETECTOR_CACHE=/data/user_data/$USER/livekit-turn-detector/hub
TALKSHOW_ORT_NUM_THREADS=1
```

#### Optional tuning

| Variable | Default | Meaning |
|----------|---------|---------|
| `SCENARIO_PATH` | `config/scenarios/default.yaml` | `host_moderated` (default) or `panel_fixed.yaml` for fixed Ryan→Amy |
| `TALKSHOW_HISTORY_MAX_LINES` | `48` | Shared transcript length sent to Gemma |
| `TALKSHOW_PANEL_MAX_TURNS` | `12` | Max panel speeches per human turn |
| `TALKSHOW_IDLE_TOPIC_SEC` | `60` | Host opens topic if room is quiet |

### 5. Run each session (two tmux panes, same GPU node)

```bash
# tmux 1 — vLLM
bash deploy/vllm-gemma4-audio.sh

# tmux 2 — agent worker
cd ~/AI-agent-talkshow
source .env
python -m agent.main dev
```

Smoke-test vLLM: `curl -s http://127.0.0.1:8000/v1/models`

After **agent code** changes: `git push` from laptop → `git pull` on Babel → restart agent in tmux 2. Frontend-only changes stay on the laptop (`pnpm dev`); Babel only needs `agent/`, `config/`, `deploy/`, etc.

### 6. Babel checklist

- [ ] Conda env `talkshow` with **Python 3.11**; `pip install -r requirements.txt`
- [ ] Piper ONNX + JSON for lessac, amy, ryan under `/data/user_data/$USER/piper/`
- [ ] `.env` `PIPER_MODEL_PATH*` points at those three files
- [ ] `download-livekit-agent-models.sh` run; `TALKSHOW_TURN_DETECTOR_CACHE` set
- [ ] vLLM up on `:8000`, then agent `dev`
- [ ] Log shows `ONNX/thread env ORT_NUM_THREADS=1` and no `pthread_setaffinity_np` errors

---

## Daily dev workflow

| Task | Where | Command |
|------|--------|---------|
| UI / panel / transcript | Laptop | `cd talkshow-web && pnpm dev` |
| Mint token | Laptop | `python api/tokens.py --room talkshow-dev --identity <you>` |
| Edit agent logic | Git + Babel | edit `agent/`, `git push` → on Babel `git pull`, restart agent |
| Run voice pipeline | Babel | vLLM + `python -m agent.main dev` |
| Change persona text | Git | `config/personas/*.yaml` |
| Change turn order / mode | Git | `config/scenarios/*.yaml` |

---

## Configuration reference

| Path | Purpose |
|------|---------|
| `.env` (repo root) | Secrets + Babel paths; shared by token script and `talkshow-web` |
| `config/multimodal.yaml` | Default vLLM URL, fallback Piper path template (`${USER}`) |
| `config/personas/host.yaml` | Lessac instructions + `piper_voice` |
| `config/personas/guest.yaml` | Amy |
| `config/personas/commentator.yaml` | Ryan |
| `config/scenarios/default.yaml` | `host_moderated` (default) |
| `config/scenarios/panel_fixed.yaml` | Legacy fixed Ryan → Amy |

Turn modes: **`host_moderated`** (hand-raise + host picks floor), **`panel_round_robin`** (fixed order), **`rotate_after_user`**, **`manual_only`**.

---

## UI data channels

Agent → browser (`talkshow/ui`): `panel_roster`, `role_active`, `transcript`, `hand_raise`, `floor_grant`

Browser → agent (`talkshow/control`): `hand_raise` (Raise hand button)

See [talkshow-web/README.md](talkshow-web/README.md).

---

## Architecture (short)

Single LiveKit participant; roles hand off in-process with different Piper models. Gemma audio-in per human utterance; panel lines via text API. Shared `show_history` for all roles.

```
User audio → GemmaAudioSTT → StoredReplyLLM → Piper (active role) → room
          → host-moderated panel → UI events → talkshow-web
```

---

## Repo layout

```
agent/              LiveKit worker, host_floor, ui_events
talkshow-web/       Next.js frontend
config/             personas/, scenarios/, multimodal.yaml
deploy/             vLLM launch, Piper / turn-detector download
api/tokens.py       Laptop token helper
requirements-laptop.txt   ← laptop pip install
requirements.txt          ← Babel GPU pip install
```

Not in git: `.env`, `.cursor/*` (except `.cursor/rules/`), `docs/`

---

## Troubleshooting (Babel)

| Log / symptom | Fix |
|---------------|-----|
| `pthread_setaffinity_np failed` | `TALKSHOW_ORT_NUM_THREADS=1` in `.env` |
| Turn detector / ONNX missing | `bash deploy/download-livekit-agent-models.sh` |
| Wrong voice for Ryan/Amy | Check `PIPER_MODEL_PATH_GUEST` / `_COMMENTATOR` paths |
| Agent never joins room | LiveKit keys match laptop; room name matches |
| `memory usage is high` | Normal with vLLM + agent; or `TALKSHOW_TURN_DETECTOR=vad` |

---

## Smoke-test vLLM audio (Babel only)

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
