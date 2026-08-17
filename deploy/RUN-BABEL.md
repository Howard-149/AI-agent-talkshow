# Babel 全栈启动（vLLM + Agent + DyStream avatar）

2× L40S 典型分工：

| GPU | 进程 | 设定 |
|-----|------|------|
| **GPU 0** | vLLM only | `CUDA_VISIBLE_DEVICES=0`（仅 tmux 1） |
| **GPU 1** | DyStream sidecar + **TTS** | DyStream: `DYSTREAM_CUDA_DEVICE=1`；Piper: `TALKSHOW_PIPER_CUDA_DEVICE=1`；可选 CosyVoice sidecar: `COSYVOICE_CUDA_DEVICE=1`（与 DyStream 共卡，注意显存） |

推荐 **sidecar**：模型只 load 一次；未设 `DYSTREAM_SIDECAR_URL` 时退回每句 subprocess（每次重 load，慢）。

**TTS：** 默认 **Piper**。情绪可控路径用 **CosyVoice3** sidecar（`TALKSHOW_TTS_ENGINE=cosyvoice`）— 见下方「CosyVoice」与 `.cursor/cosyvoice-plan.md`。

---

## 一次性准备

### 1. Talkshow conda env（agent / vLLM / Piper）

```bash
conda create -n talkshow python=3.11 -y
conda activate talkshow
cd ~/AI-agent-talkshow
pip install -r requirements.txt

bash deploy/download-piper-voices.sh en zh   # Lessac/Amy/Ryan + 中文 huayan
# 自选：bash deploy/download-piper-voices.sh --list
#       https://huggingface.co/rhasspy/piper-voices
bash deploy/download-livekit-agent-models.sh
cp .env.example .env   # 填 LiveKit、PIPER_*、PIPER_MODEL_PATH_ZH*、VLLM_*
```

### 2. DyStream 代码 + 权重（repo 外）

```bash
export DYSTREAM_ROOT=/data/user_data/$USER/dystream
bash deploy/clone-dystream.sh
export HF_TOKEN=hf_...
bash deploy/download-dystream-weights.sh
bash deploy/verify-hf-token.sh
```

### 3. DyStream **独立** conda env（推荐）

**不要** `pip install -r $DYSTREAM_ROOT/requirements.txt`（上游 numpy/opencv/mediapipe 版本冲突 + `mediapipee` 拼字错误）。

```bash
cd ~/AI-agent-talkshow
bash deploy/install-dystream-env.sh    # 创建 dystream env + 精简 inference 依赖
```

在 `.env`：

```bash
DYSTREAM_PYTHON=/data/user_data/$USER/miniconda3/envs/dystream/bin/python
DYSTREAM_ROOT=/data/user_data/$USER/dystream
DYSTREAM_CUDA_DEVICE=1
DYSTREAM_SIDECAR_URL=http://127.0.0.1:8766
TALKSHOW_AVATAR_ENABLED=1
TALKSHOW_AVATAR_LK_VIDEO=1
```

Portrait（Linux 大小写敏感）→ `avatar/assets/portraits/`，与 persona yaml 一致。

```bash
bash deploy/init-avatar-dirs.sh
```

可选 smoke test：

```bash
conda activate talkshow
export DYSTREAM_ROOT=... DYSTREAM_PYTHON=... DYSTREAM_CUDA_DEVICE=1
python -m avatar.dystream_once --portrait avatar/assets/portraits/lessac.png \
  --audio /path/to/test.wav --output /tmp/test.mp4 --steps 5
```

### 4. 共用 env（仅实验，不推荐）

若坚持单 env：在 **全新** `talkshow` 里先 `pip install -r requirements.txt`，再 **手动** 补 DyStream 缺的包（`lightning`、`einops` 等），**不要** `pip install -r $DYSTREAM_ROOT/requirements.txt` 全覆盖。冲突时改回独立 `dystream` env + `DYSTREAM_PYTHON`。

---

## 每次 session（Babel 上 3 个 tmux）

```bash
conda activate talkshow
cd ~/AI-agent-talkshow
set -a && source .env && set +a
```

**tmux 1 — vLLM（默认 GPU 0；也可 `VLLM_CUDA_DEVICE=0`）**

```bash
bash deploy/vllm-gemma4-audio.sh
# 等 curl -s http://127.0.0.1:8000/v1/models 有输出
```

**tmux 2 — agent worker（不要锁单卡：Piper 默认用 slot 1）**

```bash
conda activate talkshow
cd ~/AI-agent-talkshow
set -a && source .env && set +a
# Do NOT export CUDA_VISIBLE_DEVICES=0 here — that hides GPU 1 from Piper.
unset CUDA_VISIBLE_DEVICES
python -m agent.main dev
```

**tmux 3 — DyStream sidecar（默认 GPU 1）**

```bash
cd ~/AI-agent-talkshow
set -a && source .env && set +a
bash deploy/run-dystream-sidecar.sh
```

**tmux 4 — CosyVoice（默认 GPU 2；只有两卡时自动用 1）**

```bash
bash deploy/run-cosyvoice-sidecar.sh
```

`.env` 需有 `DYSTREAM_SIDECAR_URL=http://127.0.0.1:8766`；agent 会对 sidecar POST `/bake`（传 portrait/audio/output 路径），不再每句 spawn subprocess。

Agent 改代码：`git pull` 或 `./scripts/sync-to-babel.sh` → 重启 tmux 2。Sidecar 改 avatar 代码 → 重启 tmux 3。

---

## CosyVoice TTS（可选，情绪可控）

Upstream: [FunAudioLLM/CosyVoice](https://github.com/FunAudioLLM/CosyVoice)

**推荐 clone 路径（Fun-CosyVoice3）：** 用现有 Piper 音色 bake **每个角色一个**参考音（跨 zh/en 共用），再 `COSYVOICE_MODE=instruct2`。

```bash
# Bake Piper → <repo>/cosyvoice-prompts/{role}.wav (paths in config/personas)
bash deploy/bake-cosyvoice-prompts-from-piper.sh

# Fun-CosyVoice3 + cosyvoice conda, then .env:
#   TALKSHOW_TTS_ENGINE=cosyvoice
#   COSYVOICE_MODE=instruct2
#   COSYVOICE_MODEL_DIR=.../Fun-CosyVoice3-0.5B
#   COSYVOICE_SIDECAR_URL=http://127.0.0.1:8767
# (no COSYVOICE_PROMPT_WAV* — see config/personas/*.yaml)
bash deploy/run-cosyvoice-sidecar.sh
```

旧的 300M-Instruct `spk_id` 路径仍可用（`COSYVOICE_MODE=instruct`），音色选择有限。

### CosyVoice 加速（vLLM + TensorRT）

官方两层：`load_vllm`（LLM）+ `load_trt`（Flow）。可装进 **同一个** `cosyvoice` conda（不必再 clone env）：

```bash
# 默认会装 accel；只要 base：COSYVOICE_INSTALL_ACCEL=0 bash deploy/install-cosyvoice-env.sh
bash deploy/install-cosyvoice-env.sh
# 等价手动：
#   pip install vllm==0.11.0 transformers==4.57.1 numpy==1.26.4 -i https://mirrors.aliyun.com/pypi/simple/ --trusted-host=mirrors.aliyun.com
#   python -m pip install tensorrt
```

`.env`（推荐先只开 vLLM；TRT10 与 CosyVoice `EXPLICIT_BATCH` 不兼容时会在 convert onnx→trt 崩）：

```bash
COSYVOICE_PYTHON=.../envs/cosyvoice/bin/python
COSYVOICE_LOAD_VLLM=1
COSYVOICE_LOAD_TRT=0
COSYVOICE_FP16=0

bash deploy/run-cosyvoice-sidecar.sh
# 日志: loading CosyVoice ... vllm=True trt=False
```

| 开关 | 加速 | 注意 |
|------|------|------|
| `COSYVOICE_LOAD_VLLM=1` | LLM token | 需模型目录下 `vllm/`；启动约几十秒 compile |
| `COSYVOICE_LOAD_TRT=1` | Flow DiT | 需 `flow.decoder.estimator.fp32.onnx`；TRT10 可能报 `EXPLICIT_BATCH` — 先关 |

更重路径：`CosyVoice/runtime/triton_trtllm` Docker（另一套服务）。共 GPU1 时注意与 DyStream 抢显存。

---

## Laptop（前端 + token）

```bash
# repo root
pip install -r requirements-laptop.txt
cp .env.example .env   # LiveKit 三件套

cd talkshow-web && pnpm install && pnpm dev
# 另开 terminal
python api/tokens.py --room talkshow-dev --identity your-name
```

`.env` 加 `NEXT_PUBLIC_DYSTREAM_ENABLED=1`（订阅 agent 的 `talkshow-avatar` video track）。

Room 名须与 Babel agent 一致（默认 `talkshow-dev`）。

---

## GPU 行为说明

- **tmux 1 vLLM：** `CUDA_VISIBLE_DEVICES=0` — 只用物理 GPU 0。
- **tmux 2 agent：** **不要** `export CUDA_VISIBLE_DEVICES=0`（否则 Piper 看不到 GPU 1）。Piper 默认 `TALKSHOW_PIPER_CUDA_DEVICE` = `DYSTREAM_CUDA_DEVICE`（**1**）。
- **tmux 3 sidecar：** `CUDA_VISIBLE_DEVICES=$DYSTREAM_CUDA_DEVICE`（默认 1）。
- **tmux 4 CosyVoice（可选）：** `COSYVOICE_CUDA_DEVICE=1`，与 DyStream 共卡；显存不够就只开其一或回退 Piper。
- **SLURM 3-GPU：** `sbatch deploy/slurm-talkshow-3gpu.sh` — 卡依次绑 Gemma / DyStream / CosyVoice；conda 分别为 `talkshow` / `dystream` / `cosyvoice_vllm`（可用 `TALKSHOW_CONDA_ENV` / `DYSTREAM_CONDA_ENV` / `COSYVOICE_CONDA_ENV` 或 `.env` 的 `*_PYTHON` 覆盖）。日志在 `logs/slurm-<jobid>/`。
- 单卡机器：`DYSTREAM_CUDA_DEVICE=0` 且 `TALKSHOW_PIPER_CUDA_DEVICE=0`，接受与 vLLM 抢同一张卡。

---

## 检查清单

- [ ] vLLM `:8000` 正常
- [ ] agent log 有 `ONNX/thread env ORT_NUM_THREADS=1`
- [ ] agent log 有 `AgentServer idle_processes=1` 与 `Piper prewarm ready...`（**registered 后、job 前**即应出现）
- [ ] `TALKSHOW_AVATAR_ENABLED=1` 且 `DYSTREAM_ROOT` 存在 `app.py` + `checkpoints/`
- [ ] `DYSTREAM_PYTHON` 指向已装 DyStream deps 的 interpreter
- [ ] laptop token + `pnpm dev`，`NEXT_PUBLIC_DYSTREAM_ENABLED=1`
- [ ] panel 发言后 log 有 `dystream bake` / `avatar_sync_playout`

---

## 相关文件

| 文件 | 用途 |
|------|------|
| [README.md](../README.md) | 团队 onboarding |
| [avatar/README.md](../avatar/README.md) | Avatar 资产与 env |
| [DYSTREAM_INTEGRATION.md](../DYSTREAM_INTEGRATION.md) | 架构与 latency 事件 |
