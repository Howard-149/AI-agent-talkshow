# Talkshow Web

Fork of [livekit-examples/meet](https://github.com/livekit-examples/meet) with talkshow-specific UI:

- **Virtual panel** — Lessac / Ryan / Amy avatars (right sidebar)
- **Transcript** — from agent `talkshow/ui` data channel
- **Meet UI** — full LiveKit Components conference (mic, chat, layout)

LiveKit media still runs on **LiveKit Cloud**; token via **`api/tokens.py`**.  
**Env:** reads **repo root `.env`** (same as agent) — no `talkshow-web/.env.local`.

## Quick start

```bash
# Repo root — one .env for agent, tokens, and frontend
source .venv/bin/activate
python api/tokens.py --room talkshow-dev --identity howard

cd talkshow-web
pnpm install   # or: npm install
pnpm dev       # or: npm run dev
```

Open http://localhost:3000 → **Talkshow (token)** tab → paste URL + TOKEN → Connect.

Same flow as [meet.livekit.io Custom tab](https://meet.livekit.io/?tab=custom).

## Agent sync

Panel/transcript events require latest agent on Babel (`agent/ui_events.py`). Sync repo and restart:

```bash
python -m agent.main dev
```

## Two humans

Each runs `tokens.py` with different `--identity`, same `--room talkshow-dev`. Both open the **same deployed or LAN URL** and paste **their own token**.

## Optional: server-signed tokens

Meet includes `/api/connection-details` — uses root `.env` `LIVEKIT_*` and **Meet demo** tab + room name `talkshow-dev`. Default is **token paste** tab.

## Talkshow additions (this fork)

| Path | Purpose |
|------|---------|
| `lib/talkshow/TalkshowMediaControls.tsx` | LiveKit mic/camera toggles + device menus + settings |
| `lib/talkshow/TalkshowView.tsx` | Virtual panel stage + composable controls |
| `lib/talkshow/roles.ts` | Event types; roster from agent (not hardcoded) |
| `styles/TalkshowPanel.module.css` | Sidebar styling |

**Agent contract (`talkshow/ui`):**

- `panel_roster` — on connect, from `config/scenarios/*.yaml` + `config/personas/*.yaml`
- `role_active` — who is speaking now
- `transcript` — final lines

Late joiners can also read roster from agent participant **metadata** (`talkshowAgent` + `panelRoster`).

Upstream: Apache-2.0 — see `LICENSE`.

## Deploy

Vercel root directory: `talkshow-web`. Set `LIVEKIT_*` in Vercel env (or use token tab only).

Connect form default URL comes from `LIVEKIT_URL` in root `.env` locally; on Vercel set `LIVEKIT_URL` or `NEXT_PUBLIC_LIVEKIT_URL`.

## Troubleshooting

**`command not found: pnpm`** — install once, then retry:

```bash
npm install -g pnpm
# or: brew install pnpm
```

**Build: background image "not a valid image file"** — JPEGs are Git LFS in upstream Meet. If you only got 130-byte pointer files:

```bash
bash scripts/fetch-background-images.sh
pnpm build
```
