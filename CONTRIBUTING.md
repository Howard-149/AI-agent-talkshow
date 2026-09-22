# Contributing

Thanks for collaborating on AI-Agent-Talkshow.

## Language

**All files tracked in git must be in English**, including:

- README, CONTRIBUTING, code comments (new/changed)
- YAML scenario/persona comments
- `.env.example` comments

Local-only notes (e.g. `.cursor/`, `docs/`, personal `ROADMAP.md`) may use any language and stay out of git.

## Before you start

1. Read [README.md](README.md) for setup and architecture.
2. For Babel sidecars (DyStream, CosyVoice): [deploy/RUN-BABEL.md](deploy/RUN-BABEL.md).
3. Copy `.env.example` → `.env` (never commit secrets).

## Development split

| Work | Where | Install |
|------|--------|---------|
| Token helper | Laptop | `pip install -r requirements-laptop.txt` |
| Frontend (`talkshow-web/`) | Laptop or Vercel | `pnpm install` in `talkshow-web/` |
| Agent, vLLM, CosyVoice / DyStream sidecars | Babel | conda `talkshow` **Python 3.11**, `pip install -r requirements.txt`; submit `deploy/slurm-talkshow-3gpu.sh` |
| LiveKit media | LiveKit Cloud | Keys in repo root `.env` |

- **Laptop:** do not use `requirements.txt` unless running the full agent on a GPU machine.
- **Babel:** default TTS is CosyVoice (`TALKSHOW_TTS_ENGINE=cosyvoice` + sidecar). How to run: [README.md](README.md). Sidecar install: [deploy/RUN-BABEL.md](deploy/RUN-BABEL.md).
- Agent on Babel: `git pull` or `./scripts/sync-to-babel.sh`, then resubmit `sbatch deploy/slurm-talkshow-3gpu.sh`. Copy `.env` to Babel separately (never commit it).

## Pull requests

- One logical change per PR when possible.
- Describe **why**, not only what.
- Note if Babel restart or frontend rebuild is required.

## Key paths

```
agent/           LiveKit worker, handoff, panel speech, ui_events
avatar/          DyStream sidecar integration, streaming synth
talkshow-web/    Next.js frontend (virtual panel + transcript)
config/          personas, scenarios, multimodal.yaml
deploy/          vLLM, sidecars, SLURM
```
