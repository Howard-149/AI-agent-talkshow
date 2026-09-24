# AI-Agent-Talkshow

Voice-in / voice-out talk show: **3 AI roles** (Host / Guest / Commentator), **Gemma 4 audio-in**, **CosyVoice TTS** (default), optional **DyStream** talking-head avatars, **LiveKit Agents**, custom frontend in **`talkshow-web/`**.

CMU capstone — single RTC agent + role handoff + virtual panel UI (Lessac / Ryan / Amy).

**Collaboration:** [CONTRIBUTING.md](CONTRIBUTING.md)

---

## Who runs what

| Machine | You install | You run |
|---------|-------------|---------|
| **Laptop** | `requirements-laptop.txt` + Node/pnpm | `api/tokens.py`, `talkshow-web` (`pnpm dev`) |
| **Babel (login node)** | — | `sbatch deploy/slurm-talkshow-3gpu.sh` |
| **Babel GPU (the SLURM job)** | `requirements.txt` (conda `talkshow`, **Python 3.11**) + CosyVoice / DyStream sidecars | vLLM + CosyVoice + DyStream + `python -m agent.main dev` |
| **LiveKit Cloud** | (project keys in `.env`) | WebRTC room — no self-hosted server |

You do **not** need vLLM, CosyVoice, or the full agent stack on your laptop.

```
Laptop (talkshow-web + token)
    │  WebRTC
    ▼
LiveKit Cloud  (LIVEKIT_URL, room e.g. talkshow-dev)
    │
    ▼
Babel 3-GPU job  —  vLLM (Gemma) + CosyVoice TTS + DyStream + agent worker
```

---

## How to run a session

### 1. Babel — start the 3-GPU stack

From a Babel **login** node (not an interactive GPU shell). The job starts vLLM, the DyStream sidecar, the **CosyVoice** sidecar, then the agent worker.

| CUDA slot (`.env` / default) | Conda env | Process |
|------------------------------|-----------|---------|
| `VLLM_CUDA_DEVICE` (0) | `talkshow` | vLLM Gemma |
| `DYSTREAM_CUDA_DEVICE` (1) | `dystream` | DyStream sidecar |
| `COSYVOICE_CUDA_DEVICE` (2) | `cosyvoice_vllm` | CosyVoice sidecar |
| all slots visible | `talkshow` | `python -m agent.main dev` |

```bash
ssh babel
cd ~/AI-agent-talkshow
mkdir -p slurm-logs
git pull    # or sync from the laptop: ./scripts/sync-to-babel.sh

sbatch deploy/slurm-talkshow-3gpu.sh
# Override partition if needed:  sbatch -p <name> deploy/slurm-talkshow-3gpu.sh
```

Logs:

- `slurm-logs/slurm-talkshow-<jobid>.{out,err}`
- `logs/slurm-<jobid>/{vllm,dystream,cosyvoice,agent}.log`
- GPU bind check: `cat logs/slurm-<jobid>/gpu-bind.txt`

Wait until vLLM (`:8000/v1/models`), DyStream (`:8766/health`), and CosyVoice (`:8767/health`) are up — the script blocks on those, then starts the agent. Room name defaults to **`talkshow-dev`**.

If host RAM OOMs while models load: `TALKSHOW_STAGGER_START=1 sbatch --export=ALL deploy/slurm-talkshow-3gpu.sh`.

After **agent code** changes: push / sync → `scancel` the old job (or let it finish) → `sbatch` again. Frontend-only changes stay on the laptop (`pnpm dev`).

Sidecar install and `.env` keys: [deploy/RUN-BABEL.md](deploy/RUN-BABEL.md).

### 2. Laptop — frontend + token

```bash
cd talkshow-web && pnpm dev          # http://localhost:3000
# another terminal, repo root, venv active:
python api/tokens.py --room talkshow-dev --identity your-name
```

Paste `LIVEKIT_URL` + `TOKEN` into the **Talkshow (token)** tab. **Room name must match** the Babel agent (default `talkshow-dev`).

### 3. Optional — ghost session (no mic)

Same running 3-GPU job. See [eval/ghost_session/README.md](eval/ghost_session/README.md).

```bash
# Babel (no frontend recording):
python -m eval.ghost_session

# Laptop (record the real UI):
python -m eval.ghost_session --record --start-frontend
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

Laptop `.env` does **not** need CosyVoice / vLLM paths unless you are also developing on Babel. The frontend reads this same file via `talkshow-web/next.config.js` — **no** `talkshow-web/.env.local`.

For avatars in the panel: `NEXT_PUBLIC_DYSTREAM_ENABLED=1`.

### 3. Frontend

```bash
cd talkshow-web
pnpm install    # or: npm install
pnpm dev
# → http://localhost:3000
```

### 4. Laptop checklist

- [ ] `pip install -r requirements-laptop.txt` (not `requirements.txt`)
- [ ] Root `.env` has LiveKit URL + key + secret
- [ ] `pnpm dev` in `talkshow-web/`
- [ ] 3-GPU job running on Babel (`sbatch deploy/slurm-talkshow-3gpu.sh`)
- [ ] Same room name on token and cluster

---

## Teammate setup — Babel (one-time)

Only if you submit or debug the **agent / vLLM / sidecars**. Replace `$USER` with your Babel username. Conda may live in `~/miniconda3` or `/data/user_data/$USER/miniconda3` — use whichever you actually have (`which python` after `conda activate`).

The 3-GPU job needs **three separate conda envs** (three Python binaries). Do not pip-install CosyVoice or DyStream into `talkshow`.

| Env | Python | Used by |
|-----|--------|---------|
| `talkshow` | **3.11** | vLLM + agent worker |
| `dystream` | **3.11** | DyStream sidecar |
| `cosyvoice_vllm` | **3.10** | CosyVoice sidecar |

### 1. Code + `.env` stub

```bash
git clone <repo-url> ~/AI-agent-talkshow
cd ~/AI-agent-talkshow
cp .env.example .env
```

Fill LiveKit keys and the path variables in step 6 **before** you download models (several scripts `source .env`). `.env` is not in git — scp a filled copy if you already have one on the laptop.

### 2. `talkshow` env (agent + vLLM)

```bash
conda create -n talkshow python=3.11 -y
conda activate talkshow
cd ~/AI-agent-talkshow
pip install -r requirements.txt
python --version   # 3.11.x
```

Login-node pip is often slow or blocked. Prefer:

```bash
mkdir -p slurm-logs
sbatch deploy/slurm-install-deps.sh
```

Gemma: first `vllm serve` can pull `google/gemma-4-E4B-it` from Hugging Face, or set `VLLM_MODEL_PATH=/data/user_data/$USER/models/gemma-4-E4B-it` if the team already has weights. Keep `VLLM_MODEL=google/gemma-4-E4B-it` as the API id either way.

### 3. LiveKit agent ONNX (VAD / end-of-turn)

In `.env`:

```
TALKSHOW_TURN_DETECTOR_CACHE=/data/user_data/$USER/livekit-turn-detector/hub
TALKSHOW_ORT_NUM_THREADS=1
```

```bash
set -a && source .env && set +a
bash deploy/download-livekit-agent-models.sh
```

### 4. CosyVoice (default TTS)

Do **not** `pip install` CosyVoice into `talkshow`. The 3-GPU job looks for conda env **`cosyvoice_vllm`** (Python 3.10). Login-node pip is usually too slow for torch / vLLM — create the empty env on login, then install on a GPU node.

**4a. Clone + Fun-CosyVoice3 weights** (login is fine):

```bash
export COSYVOICE_ROOT=/data/user_data/$USER/CosyVoice
git clone --recursive https://github.com/FunAudioLLM/CosyVoice.git "${COSYVOICE_ROOT}"
# if you cloned without --recursive:
#   cd "${COSYVOICE_ROOT}" && git submodule update --init --recursive

export HF_TOKEN=hf_...
huggingface-cli download FunAudioLLM/Fun-CosyVoice3-0.5B-2512 \
  --local-dir "${COSYVOICE_ROOT}/pretrained_models/Fun-CosyVoice3-0.5B"
```

**4b. Create + install env `cosyvoice_vllm`:**

```bash
# login: empty env only
conda create -n cosyvoice_vllm python=3.10 -y

# GPU node: inference deps + vLLM 0.11 (needed by the sidecar)
srun -p preempt --gres=gpu:1 --mem=32G --time=02:00:00 --pty bash
cd ~/AI-agent-talkshow
export COSYVOICE_ROOT=/data/user_data/$USER/CosyVoice
export COSYVOICE_CONDA_ENV=cosyvoice_vllm   # must match slurm-talkshow-3gpu.sh
bash deploy/install-cosyvoice-env.sh
```

That script `conda create`s the env if missing, then `pip install`s `deploy/cosyvoice-requirements-inference.txt` plus vLLM/TensorRT. Do **not** `pip install -r "${COSYVOICE_ROOT}/requirements.txt"`.

```bash
conda activate cosyvoice_vllm
python --version                                          # 3.10.x
python -c "import whisper; print('whisper ok')"
python -c "import vllm; print('vllm', vllm.__version__)"  # 0.11.x
which python
# → COSYVOICE_PYTHON=.../envs/cosyvoice_vllm/bin/python
```

**Prompt wavs** (`cosyvoice-prompts/`) are Howard’s local test clips. Teammates skip bake / copy.

### 5. DyStream (talking heads)

Do **not** `pip install` DyStream into `talkshow` or `pip install -r $DYSTREAM_ROOT/requirements.txt` (upstream pins conflict). The 3-GPU job looks for conda env **`dystream`** (Python 3.11).

**5a. Clone + weights** (login is fine; HF token required on Babel shared IPs):

```bash
export DYSTREAM_ROOT=/data/user_data/$USER/dystream
export HF_TOKEN=hf_...
huggingface-cli whoami          # must print your HF user, not "Not logged in"
git clone https://github.com/RobinWitch/DyStream.git "${DYSTREAM_ROOT}"
cd ~/AI-agent-talkshow
bash deploy/download-dystream-weights.sh   # checkpoints/ + tools/ into DYSTREAM_ROOT
```

**5b. Create + install env `dystream`:**

```bash
conda create -n dystream python=3.11 -y

srun -p preempt --gres=gpu:1 --mem=32G --time=02:00:00 --pty bash
cd ~/AI-agent-talkshow
export DYSTREAM_ROOT=/data/user_data/$USER/dystream
export DYSTREAM_CONDA_ENV=dystream
bash deploy/install-dystream-env.sh        # curated deps, not upstream requirements.txt
```

```bash
conda activate dystream
python --version                           # 3.11.x
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"
which python
# → DYSTREAM_PYTHON=.../envs/dystream/bin/python
```

### 6. Babel `.env` (minimum for `slurm-talkshow-3gpu.sh`)

Point `*_PYTHON` at the interpreters you just created (`conda activate … && which python`).

```
LIVEKIT_URL=wss://….
LIVEKIT_API_KEY=…
LIVEKIT_API_SECRET=…

VLLM_BASE_URL=http://127.0.0.1:8000/v1
VLLM_MODEL=google/gemma-4-E4B-it
# VLLM_MODEL_PATH=/data/user_data/$USER/models/gemma-4-E4B-it

TALKSHOW_TTS_ENGINE=cosyvoice
COSYVOICE_MODE=instruct2
COSYVOICE_SIDECAR_URL=http://127.0.0.1:8767
COSYVOICE_ROOT=/data/user_data/$USER/CosyVoice
COSYVOICE_MODEL_DIR=/data/user_data/$USER/CosyVoice/pretrained_models/Fun-CosyVoice3-0.5B
COSYVOICE_PYTHON=/data/user_data/$USER/miniconda3/envs/cosyvoice_vllm/bin/python
COSYVOICE_CUDA_DEVICE=2
COSYVOICE_CONDA_ENV=cosyvoice_vllm

DYSTREAM_ROOT=/data/user_data/$USER/dystream
DYSTREAM_PYTHON=/data/user_data/$USER/miniconda3/envs/dystream/bin/python
DYSTREAM_CUDA_DEVICE=1
DYSTREAM_SIDECAR_URL=http://127.0.0.1:8766
DYSTREAM_CONDA_ENV=dystream
TALKSHOW_AVATAR_ENABLED=1
TALKSHOW_AVATAR_LK_VIDEO=1
NEXT_PUBLIC_DYSTREAM_ENABLED=1

TALKSHOW_TURN_DETECTOR_CACHE=/data/user_data/$USER/livekit-turn-detector/hub
TALKSHOW_ORT_NUM_THREADS=1
```

The job fails if the three GPU slots or three Python paths collide. Extra sidecar knobs: [deploy/RUN-BABEL.md](deploy/RUN-BABEL.md).

### 7. Babel checklist

- [ ] `talkshow` **Python 3.11** + `pip install -r requirements.txt` (or `slurm-install-deps.sh`)
- [ ] LiveKit ONNX in `TALKSHOW_TURN_DETECTOR_CACHE`
- [ ] CosyVoice clone + Fun-CosyVoice3 weights + env `cosyvoice_vllm`
- [ ] DyStream clone + weights + env `dystream`
- [ ] `.env` LiveKit keys match the laptop; `TALKSHOW_TTS_ENGINE=cosyvoice`
- [ ] `sbatch deploy/slurm-talkshow-3gpu.sh` — health on `:8000`, `:8766`, `:8767`, then agent log

---

## Daily dev workflow

| Task | Where | Command |
|------|--------|---------|
| Start the show stack | Babel login | `sbatch deploy/slurm-talkshow-3gpu.sh` |
| UI / panel / transcript | Laptop | `cd talkshow-web && pnpm dev` |
| Mint token | Laptop | `python api/tokens.py --room talkshow-dev --identity <you>` |
| Edit agent logic | Git + Babel | edit `agent/`, push / `sync-to-babel.sh`, resubmit the 3-GPU job |
| Ghost session (no mic) | Babel | `python -m eval.ghost_session` |
| Record ghost frontend | Laptop | `python -m eval.ghost_session --record` |
| Change persona text | Git | `config/personas/*.yaml` |
| Change turn order / mode | Git | `config/scenarios/*.yaml` |

---

## Configuration reference

| Path | Purpose |
|------|---------|
| `.env` (repo root) | Secrets + Babel paths; shared by token script and `talkshow-web` |
| `config/multimodal.yaml` | Default vLLM URL |
| `config/personas/host.yaml` | Lessac — CosyVoice prompt wav under `tts.cosyvoice` |
| `config/personas/guest.yaml` | Amy |
| `config/personas/commentator.yaml` | Ryan |
| `config/scenarios/default.yaml` | `host_moderated` (default) |
| `config/scenarios/panel_fixed.yaml` | Legacy fixed Ryan → Amy |

Turn modes: **`host_moderated`** (hand-raise + FIFO; orchestrated by `agent/show_graph` LangGraph, actuators in `agent/floor/host_floor.py`), **`panel_round_robin`** (fixed order), **`rotate_after_user`**, **`manual_only`**.

---

## UI data channels

Agent → browser (`talkshow/ui`): `panel_roster`, `role_active`, `transcript`, `hand_raise`, `floor_grant`

Browser → agent (`talkshow/control`): `hand_raise` (Raise hand button)

See [talkshow-web/README.md](talkshow-web/README.md).

---

## Architecture (short)

Single LiveKit participant; roles hand off in-process with per-role TTS. Gemma audio-in per human utterance; panel lines via text API. Shared `show_history` for all roles. DyStream streams lip-synced video on the agent LiveKit track.

```
User audio → GemmaAudioSTT → speak_panel_line → CosyVoice sidecar
          → optional DyStream RGBA stream → LiveKit audio + video
          → host-moderated panel → UI events → talkshow-web
```

Full sidecar / SLURM notes: [deploy/RUN-BABEL.md](deploy/RUN-BABEL.md).

---

## Repo layout

```
agent/              LiveKit worker, host_floor, ui_events
talkshow-web/       Next.js frontend
config/             personas/, scenarios/, multimodal.yaml
deploy/             SLURM 3-GPU job, vLLM, CosyVoice / DyStream sidecars
api/tokens.py       Laptop token helper
requirements-laptop.txt   ← laptop pip install
requirements.txt          ← Babel GPU pip install (talkshow env)
```

Not in git: `.env`, `.cursor/`, `docs/` (local notes, knowledge base, Cursor rules/skills)

---

## Troubleshooting (Babel)

| Log / symptom | Fix |
|---------------|-----|
| Job wants 3 GPUs / slots collide | Check `VLLM_CUDA_DEVICE` / `DYSTREAM_CUDA_DEVICE` / `COSYVOICE_CUDA_DEVICE` are 0/1/2 and distinct |
| CosyVoice / DyStream / talkshow share one Python | Set `COSYVOICE_PYTHON` and `DYSTREAM_PYTHON` to their conda envs |
| `pthread_setaffinity_np failed` | `TALKSHOW_ORT_NUM_THREADS=1` in `.env` |
| Turn detector / ONNX missing | `bash deploy/download-livekit-agent-models.sh` |
| Agent never joins room | LiveKit keys match laptop; room name matches |
| CosyVoice health never comes up | `logs/slurm-<jobid>/cosyvoice.log`; `TALKSHOW_TTS_ENGINE=cosyvoice` + sidecar URL |
| `memory usage is high` | Normal with vLLM + sidecars; or `TALKSHOW_STAGGER_START=1` |
| `Invalid qos specification` | Script uses `--qos=preempt_qos` with `partition=preempt` |

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
