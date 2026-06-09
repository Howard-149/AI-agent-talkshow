# Contributing

Thanks for collaborating on AI-Agent-Talkshow.

## Language

**All files tracked in git must be in English**, including:

- README, ROADMAP, CONTRIBUTING, code comments (new/changed)
- YAML scenario/persona comments
- `.env.example` comments

Local-only notes (e.g. `.cursor/soul.md`, personal `docs/`) may use any language.

**Shared Cursor rules** (`.cursor/rules/`) are tracked in git so collaborators get the same agent guidelines.

## Before you start

1. Read [README.md](README.md) for setup and architecture.
2. Read [ROADMAP.md](ROADMAP.md) for current priorities.
3. Copy `.env.example` → `.env` (never commit secrets).

## Development split

| Work | Where | Install |
|------|--------|---------|
| Token helper | Laptop | `pip install -r requirements-laptop.txt` |
| Frontend (`talkshow-web/`) | Laptop or Vercel | `pnpm install` in `talkshow-web/` |
| Agent, vLLM, Piper | Babel GPU node | conda `talkshow` **Python 3.11**, `pip install -r requirements.txt` |
| LiveKit media | LiveKit Cloud | Keys in repo root `.env` |

- **Laptop:** do not use `requirements.txt` unless running the full agent on a GPU machine.
- **Babel:** set per-role Piper paths in `.env` (`PIPER_MODEL_PATH`, `PIPER_MODEL_PATH_GUEST`, `PIPER_MODEL_PATH_COMMENTATOR`). See README “Piper — one ONNX file per role”.
- Agent on Babel: `git pull` after merge, then restart `python -m agent.main dev`. Copy `.env` to Babel separately (never commit it).

## Pull requests

- One logical change per PR when possible.
- Describe **why**, not only what.
- Note if Babel restart or frontend rebuild is required.
- Link roadmap item if applicable.

## Key paths

```
agent/           LiveKit worker, handoff, panel speech, ui_events
talkshow-web/    Next.js frontend (virtual panel + transcript)
config/          personas, scenarios, multimodal.yaml
ROADMAP.md       Planned features
```
