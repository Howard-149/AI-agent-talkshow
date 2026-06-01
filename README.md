# AI-Agent-Talkshow (Phase 1)

Voice-in / voice-out talk-show: **3 AI roles** (Host / Guest / Commentator), **Gemma 4 audio-in**, **Piper TTS**, **LiveKit Agents**, frontend via **[LiveKit Meet](https://meet.livekit.io/?tab=custom)**.

## 端到端怎麼連（一圖）

```
筆電瀏覽器 (LiveKit Meet Custom)
    │  WebRTC + mic + token from api/tokens.py
    ▼
LiveKit Cloud  (LIVEKIT_URL)
    │  dispatch job → room e.g. talkshow-dev
    ▼
Babel GPU: agent worker (multi-agent)  ←→  vLLM + Piper (per role)
```

- **前端（Phase 1 開發）**：建議 [Agents Playground](https://agents-playground.livekit.io)（即時字幕）；[LiveKit Meet](https://meet.livekit.io/) 為一般會議 UI，**不**顯示 agent 逐字稿。
- **Agent / vLLM**：同一 GPU compute 節點；worker 註冊後 Meet Connect 即 dispatch 進同房。

---

## Phase 0 完整清單

### A. 一次性準備（做一次）

| # | 在哪 | 做什麼 |
|---|------|--------|
| A1 | [LiveKit Cloud](https://cloud.livekit.io) | 建 project → Settings → 複製 `URL` / `API Key` / `API Secret` |
| A2 | 本機 | `~/.ssh/config` 設 `Host babel` → `login.babel.cs.cmu.edu` |
| A3 | 本機 | `python3 -m venv .venv && pip install -r requirements-laptop.txt` |
| A4 | 本機 | `cp .env.example .env`，填入 LiveKit 三項 |
| A5 | Cluster | `mkdir -p ~/AI-agent-talkshow`，本機 `./scripts/sync-to-babel.sh` |
| A6 | Cluster | `cp .env.example .env`（**同 project** 的 LiveKit 三項 + `PIPER_MODEL_PATH`） |
| A7 | Cluster | `conda activate talkshow` → `pip install -r requirements.txt`（含 vLLM cu129 + piper-tts） |
| A8 | Cluster `/data` | `bash deploy/download-piper-voices.sh lessac amy ryan`（或 `--dest` / `--list`）→ 設 `.env` 各 `PIPER_MODEL_PATH_*` |
| A9 | Cluster | `bash deploy/download-livekit-agent-models.sh`（turn detector 等；讀 `~/.bashrc` 的 HF cache） |

### B. 每次開發 session（Cluster）

在 **GPU interactive 節點**（`echo $SLURM_JOB_ID` 非空）：

| # | tmux | 指令 |
|---|------|------|
| B1 | 1 | `bash deploy/vllm-gemma4-audio.sh`（需已裝 `vllm[audio]`） |
| B2 | 1 | 煙霧測：`curl -s http://127.0.0.1:8000/v1/models` |
| B3 | 2 | `cd ~/AI-agent-talkshow && source .env && python -m agent.main dev`（需已完成 **A9**） |
| B4 | 2 | 日誌應顯示 worker 已連上 LiveKit Cloud |

改 code 後：本機 `./scripts/sync-to-babel.sh` → cluster 重啟 B3（`dev` 多數會 hot reload）。

### C. 每次測試（本機筆電）

| # | 做什麼 |
|---|--------|
| C1 | `source .venv/bin/activate` |
| C2 | 選 **房間名**（預設 `talkshow-dev`，前後端要一致） |
| C3 | 進前端（見下節） |
| C4 | 瀏覽器允許麥克風 → Connect → 應聽到 Host 開場 → 說話 → 等 AI 回覆 |

### D. 進 LiveKit 前端（三選一）

**方式 1 — 託管 Playground（最快，推薦）**

1. 打開 [https://agents-playground.livekit.io](https://agents-playground.livekit.io)
2. 設定裡填 **與 `.env` 相同** 的 `LIVEKIT_URL`、`API Key`、`API Secret`
3. Room name 填 `talkshow-dev`（或你自訂的名稱）
4. Connect → 開麥克風

**方式 2 — 本機跑 Playground**

```bash
git clone https://github.com/livekit/agents-playground.git
cd agents-playground && pnpm install
# .env.local: LIVEKIT_API_KEY, LIVEKIT_API_SECRET, NEXT_PUBLIC_LIVEKIT_URL
pnpm dev   # → http://localhost:3000
```

Agent 仍在 cluster 跑；瀏覽器在本機。

**方式 3 — 只用 token（進 Cloud Meet / 自建頁）**

```bash
cd AI-agent-talkshow && source .venv/bin/activate
python api/tokens.py --room talkshow-dev --identity human-host
```

輸出 `LIVEKIT_URL`、`TOKEN` → 貼到支援 LiveKit 的客戶端（進階；Phase 0 建議用方式 1）。

### E. 驗收（Phase 0）

- [ ] Playground 裡看到 agent participant 進房
- [ ] 你說英文一句 → 聽到 Piper 語音回覆
- [ ] Cluster log 有 Gemma / Piper 相關輸出
- [ ] `logs/session-*.jsonl`（cluster）有 `user_heard` / `assistant_reply`

---

## Phase 2 — Scene / floor awareness

| 重點 | 說明 |
|------|------|
| **場景感** | 每段 Gemma 帶 `SCENE`：誰剛說、**誰下一個**、人類 floor 是 FROZEN 還是 OPEN（見 `agent/show_context.py`） |
| **自然對話** | 下一位不是你時，不對你開口（不是單純禁問號，而是整段話的對象錯了就不自然） |
| **主持** | Lessac 對「全場 / panel」點題；只有收尾時 floor 才 OPEN 給你 |
| **開場** | 固定 `session.say`，說明你說完會交給 Ryan、Amy |
| **Session history** | `TalkShowData.show_history` → 每輪 Gemma 帶完整對話（`TALKSHOW_HISTORY_MAX_LINES`） |

---

## Phase 1 — Multi-agent + turn control + LiveKit Meet

### 架構

| 元件 | 說明 |
|------|------|
| **Host / Guest / Commentator** | `config/personas/*.yaml`；handoff 與輪替 |
| **Persona ↔ Piper** | Lessac / Amy / Ryan ↔ `lessac` / `amy` / `ryan` ONNX（見各 yaml `piper_voice`） |
| **Turn control** | `config/scenarios/default.yaml`：`rotate_after_user` 或 `manual_only` |
| **Gemma handoff 標籤** | Host 可在 `[reply]` 末加 `[handoff:guest]` / `[handoff:commentator]` |
| **輪替（預設）** | `panel_round_robin`：你說 → Lessac 回 → **Lessac 報幕** → Ryan → **Lessac 報幕** → Amy → Lessac 收尾 → 再輪到你 |

Debug：見 `docs/DEBUG_PANEL.md`（本機；cluster 可看 log 關鍵字 `PANEL speak` / `Gemma text request`）。

```
User audio → Gemma (active_role persona) → [heard]/[reply] → Piper (active agent TTS)
         → optional [handoff:role] or rotate_after_user → 下一輪由誰聽/說
```

### F. 前端怎麼選

| 前端 | 即時字幕 | 房內幾個 bot 頭像 | 適合 |
|------|----------|-------------------|------|
| **[Agents Playground](https://agents-playground.livekit.io)** | ✅ 有 | **1 個**（handoff 時名稱可變，如 `Lessac (Host)`） | **日常開發、Phase 1 驗收** |
| **[LiveKit Meet](https://meet.livekit.io/?tab=custom)** | ❌ 無 agent transcript | **1 個** | 像一般視訊會議的 demo |

#### 為什麼不是 3 個參與者？

Phase 1 用 LiveKit **handoff**：**一條語音軌、一個 agent participant**，Host / Guest / Commentator **輪流上場**（聲線可不同）。  
要房內 **3 個獨立 bot 頭像** = 要 **3 個 agent worker 各自進房**（3 倍資源、搶話與協調），屬 **Phase 2+ / 自建前端**，目前未做。

#### 推薦：Agents Playground（有字幕）

1. Cluster：`python -m agent.main dev`
2. 筆電打開 [Agents Playground](https://agents-playground.livekit.io)
3. 填與 `.env` 相同的 **URL / API Key / Secret**，Room = `talkshow-dev`
4. Connect → 開麥克風 → 左側應有 **user / agent 逐字稿**

#### 選用：LiveKit Meet（無字幕）

```bash
source .venv/bin/activate && source .env
python api/tokens.py --room talkshow-dev --identity human-host
```

[Meet → Custom](https://meet.livekit.io/?tab=custom) 貼 **URL + TOKEN**。Agent 仍由 worker 自動進同房。

### G. 設定

| 檔案 | 用途 |
|------|------|
| `config/scenarios/default.yaml` | `turn_control.mode`、`order`、`meet.default_room` |
| `SCENARIO_PATH` | 覆寫 scenario（可選） |
| `PIPER_MODEL_PATH_GUEST` / `_COMMENTATOR` | 各角色不同聲線（可選；預設同 Host） |

`manual_only`：不自動輪替，只靠 `[handoff:…]` 或之後擴充。

### H. Phase 1 驗收

- [ ] Meet Connect 後 agent 進 `talkshow-dev`（或你的 room）
- [ ] 連續 3 輪對話，log 中 `active_role` 在 host / guest / commentator 間變化
- [ ] `logs/session-*.jsonl` 有 `handoff` 事件（`rotate_after_user` 或 `gemma_handoff_tag`）
- [ ] （可選）下載第二、第三個 Piper 聲模並設 `PIPER_MODEL_PATH_*`，聽到不同聲音

---

## Repo layout (tracked in git)

```
agent/          # LiveKit worker, multi-agent, supervisor/
config/         # multimodal.yaml, personas/, scenarios/
deploy/         # vLLM launch script
api/            # token helper (laptop, Meet)
```

Not in git (local only): `.cursor/`, `docs/`, `scripts/sync-to-babel.sh`

## Laptop

1. Copy `.env.example` → `.env` (LiveKit Cloud keys).
2. Python deps (token helper only):
   ```bash
   python3 -m venv .venv && source .venv/bin/activate
   pip install livekit-api python-dotenv
   ```
3. Sync code to cluster: `./scripts/sync-to-babel.sh` (script lives locally under `scripts/`).
4. Mint token: `python api/tokens.py --room talkshow-dev`

## Babel (GPU node)

```bash
conda activate talkshow
cd ~/AI-agent-talkshow
pip install -r requirements.txt
bash deploy/download-piper-voices.sh lessac amy ryan   # 多聲線；見 --help / --list
bash deploy/download-livekit-agent-models.sh  # turn detector + Silero 等 ONNX

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
