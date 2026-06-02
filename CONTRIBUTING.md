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

| Work | Where |
|------|--------|
| Agent, vLLM, Piper | Babel GPU node |
| Frontend (`talkshow-web/`) | Laptop or Vercel |
| Token helper | Laptop (`api/tokens.py`) |
| LiveKit media | LiveKit Cloud |

Agent code sync: `./scripts/sync-to-babel.sh` (local script, not in git).

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
