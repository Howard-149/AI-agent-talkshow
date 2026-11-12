# Dialogue quality for the AI talk show

This document describes how we measure and improve what the AI panelists say. It covers four layers that sit on top of the show:

| Layer | Problem it addresses | Status |
| --- | --- | --- |
| Dialogue fixes | Panelists ignore what the audience said, nobody answers when a panelist is named, lines are vague | In `main` (#1), behind flags that are off by default, ported to Think & Action |
| Measurement tools | No way to tell whether a change helped | In `main` (#2) |
| Fact pack | Asking for concrete lines makes panelists invent facts | In `main`, behind a flag that is off by default |
| Benchmark | One fixed yardstick across systems and models | In `main`; pilot results below |

The document is kept up to date; see the changelog at the end.

## System architecture

The line a panelist speaks is the `content` it drafts during its Think & Action call. So our prompt additions and the fact pack are added to the think prompt, and addressee priority is added to free-floor speaker selection. Floor rules, emotion, speech and the avatar belong to the rest of the team's system.

```mermaid
flowchart TB
    A["Audience member<br/>mic, ghost or text"] --> S["Speech to text<br/>transcribe only"] --> G["Show graph<br/>opening, turns, closing"]
    G --> TA
    P["Our prompt additions<br/>uptake plan, concrete style, fact pack"]:::ours --> L
    E["Emotion (PAD mood)<br/>from listeners' appraisal"] --> L
    subgraph TA["Think & Action floor"]
        L["Listeners think every interval<br/>host + panelists: decide, draft content"] --> R["Floor rules<br/>quote check, min speech, cooldown, host first<br/>(interrupts only)"] --> C["Floor controller<br/>an LLM picks who speaks or may cut in"]
        C --- AP["Addressee priority"]:::ours
    end
    TA --> SL["Spoken line<br/>speech queue"] --> T["Text to speech<br/>CosyVoice"] --> V["Talking avatar<br/>DyStream"] --> W["Web stage<br/>LiveKit, browser"]
    WD["Worker draft (fallback)<br/>used when no line was drafted; same fixes apply"] --> SL
    classDef ours fill:#e3eefc,stroke:#2f6fdf,stroke-width:2px;
```

Blue boxes are our additions.

## Components

Every change to the show is behind a flag that is off by default. With the flags off, the panel prompts are byte-identical to `main`.

| Component | What it does | Where it hooks in | Flag |
| --- | --- | --- | --- |
| Addressee priority | If the audience names a panelist and that panelist chose to speak, they get the floor | Free-floor selection, `ThinkActionInteraction.arbitrate_free` | `TALKSHOW_ADDRESSEE_FLOOR=1` |
| Addressee priority, forced | Also gives the floor to a named panelist who chose silent; their line is drafted by the worker, as for any picked panelist with no line of their own (reason code `directed_response_forced`) | Same | `TALKSHOW_ADDRESSEE_FLOOR=force` |
| Answer the audience | Lists the audience member's figures, names and stance, and asks the panelist to answer that point | Think prompt; extraction starts when the audience turn is committed | `TALKSHOW_UPTAKE_PLANNER=lite`, `TALKSHOW_PANEL_SCAFFOLD_FIX=1` |
| Concrete lines | Style examples that carry facts, and a one-concrete-fact rule | Think prompt and worker fallback | `TALKSHOW_DIALOGUE_LIBRARY=panel_concrete`, `TALKSHOW_PANEL_CONCRETE=1` |
| Fact pack | The 6 most relevant checked facts for the topic; specifics may come only from them or from what was said in the show | Think prompt and worker fallback | `TALKSHOW_FACT_PACK=<json>` |
| Lean think prompt | On a free floor: no repeat of the line the transcript ends with, no interrupt or backchannel guidance, no canned host lines in the transcript | Panelist think prompt, `agent/panel/think_prompt.py` | `TALKSHOW_THINK_LEAN=1` |
| Persona near the line | "Write the content as Ryan: ..." from the persona card, right before the line is written | Panelist think prompt | `TALKSHOW_THINK_PERSONA_NEAR=1` |
| Respond to now | Who spoke last, whom they named, the planner's points, and a rule to stay silent when someone else was named | Panelist think prompt | `TALKSHOW_THINK_FOCUS=1` |
| Persona library | Structured personas (role, background, leaning, style, moves, avoid) any panel seat can take; a short card in the system message and a style block next to the line | `config/persona_library/`, `agent/persona_library.py`; system message and panelist think prompt | `TALKSHOW_PERSONA_LIBRARY=1`, `TALKSHOW_PERSONA_<SEAT>=<id>` |
| Listening test | Edit one element of the audience turn, replay, and check whether the reply changes | `eval/responsiveness.py`, `eval/judge_context.py`, replay hooks | `TALKSHOW_GHOST_REPLAY=1` |
| Quality ruler | Yes/no checklist scored by two judges plus counted metrics; every judge measure is validated before use | `eval/quality_ruler/` | none |
| Unsupported specifics | Counts numbers and names that have no source in the fact pack or the show so far | `eval/quality_ruler/specifics.py` | none |
| Benchmark | Fixed items, a common system interface, judges, leaderboard | `eval/bench/` | none |
| T2f floor items | Both panelists bid on one audience line and the show's own free-floor selection picks; plain systems answer a floor question instead | `eval/bench/floor.py` | none |

The recommended set ("fast") turns on addressee priority, answer the audience and concrete lines. "fast_v2" adds the lean think prompt and the persona reminder; it is the recommended set for Think & Action. Both are listed in `deploy/arms/dialogue-quality.arms` (FAST, FAST2); `fast` stays unchanged so earlier results remain comparable.

## Fact pack pipeline

72 topics, 1,482 facts (4 to 31 per topic), built without a paid search API.

1. **Search queries.** An open LLM rewrites each motion into 5 concrete Wikipedia queries. Debate titles are often slogans, so searching the title directly finds off-topic pages.
2. **Wikipedia.** The free API, with a pause between requests. Up to 10 articles and 160 candidate sentences per topic.
3. **Select and rewrite.** An open LLM picks up to 30 sentences that bear on the motion and rewrites each as one short fact.
4. **Re-link and verify.** The model often cites the wrong sentence number, and sometimes invents a fact. A program matches every fact back to its closest source sentence and keeps it only if at least 60% of its content words and all of its numbers appear there.

Known limits:
- Side labels (for/against) are unreliable, so both panelists get the same facts.
- The check only proves that a fact is in its Wikipedia sentence, not that Wikipedia is right.

Facts come from Wikipedia (CC BY-SA 4.0); every fact carries its page and URL.

## Benchmark

Any talk show system runs the same fixed items and is scored the same way. T1 to T4 are counterfactual pairs: run the original and an edited input, and check whether the output changes as it should.

```mermaid
flowchart LR
    I["Items<br/>5 tasks<br/>dev 574, test 811"] --> P1["Plain model"] & P2["Show: worker"] & P3["Show: Think & Action"]
    P1 & P2 & P3 --> O["Outputs<br/>3 reruns per item<br/>reply, speak or yield"]
    O --> R1["Rules"] & R2["Rubric judges"] & R3["Identification judge"]:::ours
    R1 & R2 & R3 --> S["Task scores 0 to 100"] --> B["Leaderboard"]
    classDef ours fill:#e3eefc,stroke:#2f6fdf,stroke-width:2px;
```

| Task | What is edited | Good behaviour | Items (dev / test) |
| --- | --- | --- | --- |
| T1 Listening | A number, the stance, a named entity or a cause in the audience turn; a paraphrase as control | The reply changes with each edit; the paraphrase moves it no more than a rerun does | 150 / 243 |
| T2 Addressing | The audience names panelist A or B | Speak when named, yield when not | 120 / 168 |
| T2f Floor | Same lines plus a control naming nobody; both panelists bid and the floor picks | The named panelist gets the floor | 90 / 126 |
| T3a Grounding, unprompted | The number in one fact of the pack; the audience asks about that fact | The reply uses the new value and adds no unsupported specifics | 92 / 116 |
| T3b Grounding, instructed | Same as T3a, plus "Use these facts when they are relevant to your reply." | Same | 92 / 116 |
| T4 Persona | The two persona cards are swapped | The style follows the card | 120 / 168 |
| T5 Hosting, T6 Full show | Not built yet | | |

**Splits.**
- Dev is 30 hand-written topics, used for tuning.
- Test is 42 motions from the [Intelligence Squared debates corpus](https://convokit.cornell.edu/documentation/iq2.html) ([Zhang et al., NAACL 2016](https://aclanthology.org/N16-1017/)), reported once. The test text is rebuilt locally from ConvoKit and never committed; only IDs and edits are.

**Scoring.**
- **Rules** cover what a program can check: who spoke, whether a number or name has a source, whether the reply states the edited value.
- **Rubric judges** answer yes/no questions that need reading. There are two judges from different model families (Qwen2.5-72B, Llama-3.3-70B), read from token probabilities.
- **Identification judge** handles every counterfactual pair. It sees one reply and both input versions and guesses which one the reply answered; 0.5 is chance.
- Every judge measure must pass three gates before use:
  - its score drops on lines we damage on purpose;
  - real human panels score above the system;
  - the two judges agree.

**Task scores.**
- Identification accuracy J maps to 200 × (J − 0.5).
- T2 uses balanced accuracy over "speak when named" and "yield when not named", so a system that always speaks or always yields scores 0.
- The overall score is the mean over T1, T2, T3a, T3b and T4.
- Each version runs 3 times. Rubric items are scored on the first rerun only, to save compute.

## Current results

These are pilot numbers on the dev split, averaged over both judges, on a 0 to 100 scale.

| System | Overall | T1 | T2 | T3a | T3b | T4 |
| --- | --- | --- | --- | --- | --- | --- |
| Plain Gemma-4 E4B | 39.4 | 42 | 2 | 35 | 37 | 81 |
| Show, Think & Action, fast | 38.2 | 36 | 2 | 43 | 44 | 66 |
| Show, Think & Action, fast + fact pack | 37.2 | 30 | 2 | 44 | 44 | 67 |
| Show, Think & Action, production (current `main`) | 34.8 | 20 | 2 | 43 | 40 | 69 |
| Plain Qwen2.5-7B | 30.1 | 31 | 6 | 36 | 35 | 43 |

All show arms use Gemma-4 E4B as the panel model. Unsupported specifics count the show's own fact pack as a source.

What these show:
- **The benchmark itself checks out.** The paraphrase control sits at J = 0.53, near chance. Persona separates the two models clearly (J 0.94 vs 0.63, intervals far apart). Listening also differs (J 0.71 vs 0.66), though those intervals touch.
- **Addressing.** Gemma never yields. Qwen yields 88% of the time, even when it is the one named.
- **Grounding.** Plain models rarely use the facts they are given: about 2% of replies use the edited value, with or without the instruction. Think & Action panelists do so in 15 to 17% of replies, so they score higher on T3.
- **Our fixes help listening on Think & Action.** The fast arm raises T1 from 20 to 36 (J 0.60 to 0.68); the paraphrase control stays at 0.52 for every arm.
- **Think & Action is weaker on persona** than a plain prompt (66 to 69 vs 81), with or without our flags. The line is drafted inside a long think-and-JSON prompt, which may dilute the persona.
- **The fact pack arm does not stand out here.** In T3 every arm is given the same facts by design, and it scores a little lower than fast on T1.
- **Limit of T2 as built:** each item tests one panelist's own speak-or-yield decision. Addressee priority acts in the floor controller when both panelists bid, so T2 does not exercise it. T2f does (results below).

**Think prompt context variants** (Think & Action, fast arm plus one variant each; dev split, both judges):

| Arm | T1 Listening | T4 Persona |
| --- | --- | --- |
| fast | 36 (J 0.68) | 66 |
| fast + lean think prompt | 50 (J 0.75) | 71 |
| fast + persona near the line | about the same as fast | 68 (+2) |
| fast + respond to now | no change | no change |
| fast + all three | 48 | 72 |

- The lean prompt helps most: the panelist sees the audience line once, as the last transcript line, without 2,500 characters of interrupt and backchannel rules after it.
- The persona reminder adds a little on persona.
- **Prompting does not make the model yield.** With the "respond to now" block and its rule to stay silent when someone else was named, Gemma still chose to speak on 100% of those items. Yielding has to come from the floor (addressee priority), not from the prompt.
- So the recommended set for Think & Action is fast_v2 = fast + lean + persona near (FAST2), without the focus block.

**v2 arms** (Think & Action on Gemma-4 E4B, dev split, both judges; the plain model ran T2f only):

| System | T1 | T2 | T2f | T4 |
| --- | --- | --- | --- | --- |
| Show, fast | 37 | 2 | 100 | 68 |
| Show, fast_v2 | 49 | 2 | 100 | 75 |
| Show, fast_v2 without addressee priority | 49 | 2 | 9 | 73 |
| Show, fast_v2 + persona library | 56 (J 0.78) | 0 | 84 | 79 |
| Plain Gemma-4 E4B, floor question | | | 0 (picks Ryan 90% in every version) | |

- **Addressee priority does the floor work.** With it, the named panelist gets the floor on every T2f item (100); without it, 9. The Think & Action floor controller ignores who was named.
- **A plain model asked "who should speak next" mostly says Ryan** (90%), whoever was named.
- **The persona library is the best Think & Action arm so far:** persona 79, close to the plain prompt's 81, and the best listening score (56, J 0.78).
- **Named but silent.** With the library, Amy chooses silent on 16% of the items where she is named, so the controller hands the floor to Ryan and T2f drops to 84. `TALKSHOW_ADDRESSEE_FLOOR=force` gives a named panelist the floor even after a silent decision; the line then comes from the worker draft, the path the show already uses when a picked panelist has no line. Arm `fast_v2_lib_force` measures it.

The fact pack was also tested offline with the earlier worker prompt path: 72 topics, 1,152 lines per arm.

| Measure | Current show | Fast | Fast + fact pack |
| --- | --- | --- | --- |
| Specificity (judged) | 0.49 | 0.65 | 0.77 |
| Argument quality (judged) | 0.59 | 0.71 | 0.79 |
| Lines with an unsupported specific | 0.5% | 4.6% | 0.6% |
| Persona fidelity (judged) | 0.73 | 0.74 | 0.68 |

With the fact pack, panelists say about 5 times more specifics than with the fast arm alone, and almost all of them have a source. The cost is weaker persona fidelity.

## Next steps

- [x] Add a two-panelist floor item (T2f) so addressee priority is measured; run fast_v2 with and without it.
- [ ] Look into the persona drop under Think & Action.
- [ ] Run the test split: production, fast, fast_v2 and fast_v2_lib_force against plain Gemma-4 E4B, then the 11 open models.
- [ ] Open the fact pack and benchmark PRs, with results attached.
- [ ] Recover the persona fidelity lost with the fact pack; combine it with varied openers.
- [ ] Validate the judge prompts that have not passed the gates yet: fact-edit identification, and interrupt and backchannel scoring.
- [ ] Build T5 (hosting, with a fixed reference panel) and T6 (full show).

## Changelog

| Date | Change |
| --- | --- |
| 2026-11-12 | v2 dev results (fast_v2, without addressee priority, with the persona library); `TALKSHOW_ADDRESSEE_FLOOR=force` for named panelists who chose silent, arm `fast_v2_lib_force` |
| 2026-11-12 | T2f floor items (both panelists bid, the show's floor picks); persona library behind a flag with five personas (Ryan, Amy, Lessac, Dana, Theo) |
| 2026-11-12 | Think prompt context variants measured on dev; fast_v2 (FAST2) = fast + lean prompt + persona near the line becomes the recommended Think & Action set |
| 2026-11-12 | Three Think & Action arms on the leaderboard (dev); fact pack and benchmark merged into `main`; scoring counts the show's fact pack as a source |
| 2026-11-11 | Merged `main` into the benchmark branch with no conflicts; the judge now exits cleanly when there is nothing to score; models resolve from the local cache or fail fast |
| 2026-11-11 | Benchmark pilot done: 2 models, 5 tasks, 2 judges |
| 2026-11-10 | Benchmark v1. Three design fixes from the pilot: drop implausible fact edits, ask about the edited fact and split T3 into a/b, let systems yield in T2 |
| 2026-11-10 | Environment spec and novelty check |
| 2026-11-10 | #1 and #2 ported to Think & Action and merged; #5 cleanup merged |
| 2026-11-09 | Fact pack for 72 topics; offline fact pack experiment; Think & Action pilot runs end to end |
| 2026-11-09 | #1 (dialogue fixes) and #2 (measurement tools) opened |
