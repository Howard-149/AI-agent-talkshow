# AI-Agent-Talkshow (Phase 0)

Voice-in / voice-out talk-show agent: **Gemma 4 E4B audio-in** (vLLM) + **Piper TTS** + **LiveKit Agents**.

## Repo layout (tracked in git)

```
agent/          # LiveKit worker
config/         # multimodal.yaml, personas/
deploy/         # vLLM launch script
api/            # token helper (laptop)
```

Not in git (local only): `.cursor/`, `docs/`, `scripts/sync-to-babel.sh`

## Laptop

1. Copy `.env.example` → `.env` (LiveKit Cloud keys).
2. Sync code to cluster: `./scripts/sync-to-babel.sh` (script lives locally under `scripts/`).
3. Optional token: `python api/tokens.py --room talkshow-dev`

## Babel (GPU node)

```bash
conda activate talkshow
cd ~/AI-agent-talkshow
pip install -r requirements.txt
pip install piper-tts   # or install `piper` CLI

cp .env.example .env    # fill keys; set PIPER_MODEL_PATH

# tmux 1 — vLLM with audio
bash deploy/vllm-gemma4-audio.sh

# tmux 2 — agent worker (same node as vLLM)
python -m agent.main dev
```

## Phase 0 pipeline

```
User audio → VAD/turn → GemmaAudioSTT (vLLM audio_url, [heard]/[reply])
          → StoredReplyLLM ([reply] only) → Piper TTS → room
```

Smoke-test vLLM audio (on GPU node):

```bash
# Use a short .wav file
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
