# Floor Control v2: results (2026-10-29)

Gefei Gu, interruption and floor control. Local laptop stack: OpenAI listener models,
Fish Audio voices, LiveKit Cloud, no avatars. Not the team's Babel Gemma stack.

## What changed

1. **Continuous listening.** Listeners re-decide after every fully played sentence,
   one poll in flight at a time (previously one poll, after sentence 1 only).
2. **Moderator cut.** The host listens too and may cut in when a panelist leaves the
   topic or attacks someone, quoting played speech. Priority 80, above AI rebuttals
   (50); bypasses the AI cooldown and chain cap; still waits 3 s; returns the floor
   to the speaker it cut.
3. **Quote matching fix** found by this evaluation (below).

## Offline planted-stimulus evaluation

**Question.** Can the floor controller catch a problem anywhere in a turn, route it to
the right listener, and stay silent on clean speech? What do the rules add?

**Data.** 24 hand-written motions (6 domains). `gpt-4.1` wrote a measured
5-sentence opening for Amy plus three replacement sentences per motion: an overclaim,
a clear factual error, an off-topic drift. `gpt-5.5` checked every sentence;
23/24 motions passed (one overclaim was not extreme enough). The planted sentence
sits at position 1-4 (rotating). Clean openings are the false-alarm control.
File: `eval/floor_eval/data/stimuli.json`.

**Procedure.** Each line is revealed one played sentence at a time. At each prefix
(1-4; production never polls after the last sentence) Ryan and Lessac are asked
with the production prompts and client, 3 repetitions: 23 x 4 variants x 4 prefixes
x 2 listeners x 3 = 2,208 calls per model, 0 errors. Arms are replayed from the
same cached answers, so they differ only in floor-control logic. Elapsed speech is
estimated at 2.6 words/s (Fish).

**Scoring.** *Caught* = cut by the expected listener (Ryan for overclaim and factual
error, Lessac for off-topic), after the planted sentence was played, quoting it.
Otherwise premature / wrong target / caught by the other listener / missed. Any cut
on a clean line is a false cut. 95% CIs: bootstrap over motions.

### Planted problems caught

| Arm | gpt-6-luna (production) | gpt-5.4-nano (weak-model stand-in) |
|---|---|---|
| Before: first sentence only, rules | 17% (6-29) | 12% (4-21) |
| Every sentence, rules | 65% (61-67) | 47% (39-54) |
| **Every sentence, rules + moderator (new)** | **94% (89-98)** | **66% (58-73)** |
| Every sentence + moderator, LLM only | 94% (89-98) | 67% (59-74) |

By type (new arm, luna): overclaim 100%, factual error 88% (another 6% were cut by
the moderator, not Ryan), off-topic 93% (0% without the moderator). Caught cuts
landed right after the planted sentence (mean lag 0.00 sentences for luna).
First-sentence-only catches only position-1 plants by construction.

### Clean lines (no cut is correct)

| Arm | gpt-6-luna | gpt-5.4-nano |
|---|---|---|
| Every arm with rules | 100% | 91% |
| LLM only | 100% | 91% |

### Bug found and fixed

The policy required the quote to be an exact substring. `gpt-5.4-nano` wraps quotes in
curly quotation marks, so **165 of 167 rejected interrupt requests quoted played
words correctly** and failed on punctuation only. Under the old matching, the new arm
caught 31% (22-43) for nano instead of 66%. `quote_heard()` now compares word
sequences; paraphrases and stitched fragments are still rejected. Pre-fix scores:
`results/nano-before-quote-fix/`.

### What the rules do and do not do

- With these single-turn stimuli, rules and LLM-only catch the same problems. The
  rules blocked the 2 nano cuts that quoted unplayed text; luna produced none
  (445/445 interrupt quotes were played words).
- Rules cannot fix judgment: nano's 9% false cuts on clean lines quote real played
  words. The policy checks *grounding*, not *merit*. With a smaller model (e.g. the
  team's Gemma 4B) false alarms are the main risk.
- Cooldown, chain cap, stale epochs and human priority only act across turns. This
  evaluation is single-turn; those are covered by unit tests and live trials.

### Listener behaviour (luna, all prefixes)

Median 1.2 s per intent call (p90 1.7 s), 4 strict parse failures of 2,208.
Ryan on clean speech: none 64%, queue 36%, interrupt 0%. Lessac never interrupted
clean speech or overclaims; interrupted 3% of factual-error prefixes (outside the
moderator's brief).

## Live LiveKit trials (gpt-6-luna + Fish, received audio recorded)

| Mode | Result | Cut | Server stop ack | Intent calls (median) |
|---|---|---|---|---|
| AI interrupt | pass | Ryan, grounded rebuttal | 15.5 ms | 10 (1.35 s) |
| AI interrupt | pass | Ryan, grounded rebuttal | 14.2 ms | 10 (1.28 s) |
| Queue only | pass | none (queued) | - | 14 (1.31 s) |
| Human interrupt | pass | human priority, host replied | 9.5, 5.0 ms | 0 |
| AI-only observer | pass | Ryan, grounded rebuttal | 14.9 ms | 10 (1.19 s) |

All five: non-silent received audio, no unplayed text in shared history, the
7-speech discussion completed where expected. Both listeners were polled every
time; the moderator never cut the homework debate (no drift). In queue-only mode
Amy's opening was re-polled after sentences 1, 2 and 3, i.e. continuous listening ran
live. In the AI-cut runs the demo opening states its claim in sentence 1, so the cut
came at the first poll. The worker ran before the quote-matching fix; luna rarely
adds quotation marks. Stop ack is server speech-handle cleanup, not browser silence.
File: `results/live-acceptance-2026-10-29.json`.

## Limits

- Labels come from two models plus spot checks of the factual errors, not a full
  human review. Hand-written motions; swap in the IQ2 set when available.
- Scores whether a cut was right, not how good the takeover speech was.
- "Clean lines should not be cut" is a show-design choice; a real debater may rebut
  measured claims too.
- Offline polls every boundary (upper bound); live skips boundaries while a poll runs.
- gpt-6-luna only allows temperature 1; per-call temperatures are ignored.

Reproduce: `python -m eval.floor_eval.stimuli`, `python -m eval.floor_eval.run
--reps 3 [--model gpt-5.4-nano --out logs/floor-eval/nano]`,
`python -m eval.floor_eval.analyze --run logs/floor-eval/main`.
