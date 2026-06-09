# Roadmap

Tracked priorities for collaboration. Update this file when scope changes; link from PRs and issues.

**Last updated:** 2026-06-02

---

## Phase 2+ — Planned work

### 1. Debate scenario + few-shot examples

**Goal:** A new scenario mode where panelists discuss a topic from different angles, closer to structured debate than sequential panel comments.

**Tasks:**

- [ ] Add `config/scenarios/debate.yaml` (turn order, topic seed, floor rules).
- [ ] Add `config/debate_examples/` (or inline in scenario) with 2–3 short debate transcripts as few-shot style references.
- [ ] Extend `panel_prompts.py` / `show_context.py` with debate-specific scene cards (claim, rebuttal, cross-examination beats).
- [ ] Wire scenario selection via `SCENARIO_PATH` and document in README.
- [ ] Smoke-test: human poses a topic → host frames sides → Ryan/Amy argue different perspectives without conflating human vs Amy.

**Acceptance:** Model outputs feel multi-sided (not three agreeable summaries); examples measurably reduce role confusion and “everyone agrees” drift.

---

### 2. Smarter turn-taking (replace naive rotation)

**Goal:** Replace fixed `panel_round_robin` order with host-mediated floor control.

**Status:** Core shipped — `host_moderated` is default in `config/scenarios/default.yaml`.

- [x] **`host_moderated` mode** — parallel hand-raise polls (Ryan/Amy), FIFO queue resolve
- [x] Data-channel events: `hand_raise`, `floor_grant`, `queue_state`, `floor_pending`
- [x] Frontend: ✋ badge, queue bar, hand-raise button
- [x] Strict FIFO including human; no `skip_human`
- [x] After human turn → open floor + poll (not direct panelist grant from tee-up)
- [x] `TALKSHOW_HAND_RAISE_GRANT_PAUSE_SEC` — 1s after poll UI before grant
- [ ] Interrupt / yield rules
- [ ] Logging: `handoff_reason` includes host decision text in JSONL

Legacy fixed order: `SCENARIO_PATH=config/scenarios/panel_fixed.yaml`

**Architecture:** Single LiveKit agent worker + `asyncio.gather` for parallel Gemma polls — **not** separate Python processes per panelist.

---

### 3. Avatar expressions and motion in UI

**Goal:** Visual feedback beyond static initials — **VRM head/bust** avatars tied to floor and speech lifecycle.

**Decision:** Head-only VRM (not full body). Client-side rendering in `talkshow-web`; still single RTC agent. See local plan `.cursor/avatar-plan.md`.

**Tasks:**

- [ ] Define agent → UI event schema: `avatar_state` (idle, listening, thinking, speaking, react; optional emotion/intensity).
- [ ] Agent hooks: poll start → thinking; floor_pending → listening; role_active → speaking.
- [ ] Frontend: Three.js + `@pixiv/three-vrm` head component per panel card.
- [ ] Persona yaml + `panel_roster`: `ui.avatar.vrm` path per role.
- [ ] Lip sync via VRM blend shapes (intensity/RMS; viseme optional later).
- [ ] Optional: Gemma mood tag → expression preset.

**Acceptance:** When Ryan speaks, Ryan VRM head animates (mouth + expression); poll shows thinking state; demo feels alive without extra RTC participants.

---

## Completed (reference)

- Single RTC agent + role handoff + virtual panel UI
- Gemma 4 audio-in + Piper per-role TTS
- `talkshow/ui` data channel (`panel_roster`, `role_active`, `transcript`, hand-raise queue events)
- Host-moderated FIFO floor control + human hand-raise
- Scene/floor awareness (`show_context.py`, shared `show_history`)
- Transcript sync with `speech_created`; Piper voice cache
- Human vs Amy disambiguation in panel prompts

---

## How to pick up a task

1. Comment on the checklist item in an issue or PR.
2. Branch naming: `feat/debate-scenario`, `feat/host-moderated-turns`, `feat/avatar-ui`.
3. Keep **all tracked repo text in English** (README, configs comments, ROADMAP, code comments for new work).
4. Agent changes: `git pull` on Babel and restart worker; frontend: `talkshow-web` locally or Vercel.
