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

**Options (may combine):**

- [ ] **Hand-raise signal:** Panelists emit `raise_hand` UI/agent events; host sees queue before calling next speaker.
- [ ] **Host picks next speaker:** After each beat, host LLM call reads `show_history` and outputs `[next:commentator]` or `[next:guest]` (or tool call).
- [ ] **Interrupt / yield rules:** Define when a panelist can cut in vs must wait.

**Tasks:**

- [ ] Extend `TurnController` with `host_moderated` mode (new `turn_control.mode`).
- [ ] Add data-channel events: `hand_raise`, `floor_grant`, `floor_revoke`.
- [ ] Frontend: optional hand-raise UI for human; visual queue on virtual panel.
- [ ] Logging: `handoff_reason` includes host decision text.

**Acceptance:** Order adapts to conversation (e.g. Amy responds when Ryan made a claim she should rebut); no silent assumption of fixed Ryan → Amy every round unless host chooses it.

---

### 3. Avatar expressions and motion in UI

**Goal:** Visual feedback beyond static initials — expressions and simple motion tied to `role_active` and speech.

**Tasks:**

- [ ] Define agent → UI event schema: `avatar_state` (emotion, gesture, speaking intensity).
- [ ] Map TTS/speech lifecycle to states (idle, speaking, listening, react).
- [ ] Frontend: avatar component (2D sprites, Lottie, or lightweight 3D — TBD).
- [ ] Optional Phase 3 tie-in: sentiment tag from Gemma → drives expression.

**Acceptance:** When Ryan speaks, Ryan avatar animates; panel feels alive in demo without extra RTC participants.

---

## Completed (reference)

- Single RTC agent + role handoff + virtual panel UI
- Gemma 4 audio-in + Piper per-role TTS
- `talkshow/ui` data channel (`panel_roster`, `role_active`, `transcript`)
- Scene/floor awareness (`show_context.py`, shared `show_history`)
- Transcript sync with `speech_created`; Piper voice cache
- Human vs Amy disambiguation in panel prompts

---

## How to pick up a task

1. Comment on the checklist item in an issue or PR.
2. Branch naming: `feat/debate-scenario`, `feat/host-moderated-turns`, `feat/avatar-ui`.
3. Keep **all tracked repo text in English** (README, configs comments, ROADMAP, code comments for new work).
4. Agent changes: sync to Babel and restart worker; frontend: `talkshow-web` locally or Vercel.
