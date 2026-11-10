# Talk show benchmark

Any talk show system (a plain LLM, the show's panel worker, the show's Think & Action floor, or your own) answers the
same fixed items and is scored the same way. An item is one turn of a show: a topic, persona cards, the transcript so
far, an audience member's new line, and sometimes a list of checked facts. Most items come in counterfactual pairs: one
element of the input is edited, and a good reply should change with it (or should not, for a control edit).

## Tasks

| task | what is edited | role tested | a good reply |
|---|---|---|---|
| T1 listening | the audience line: a number, the stance, a named entity, a cause; a paraphrase as control | the panelist the line addresses | follows the edit; ignores the paraphrase |
| T2 addressing | the audience line names panelist A or panelist B | both panelists, each may speak or yield | the named panelist speaks and answers, the other yields |
| T3a grounding | one number in the fact the audience asks about (up to 2 pairs per topic); facts listed, no instruction to use them | the addressed panelist | uses the fact as given, unprompted |
| T3b grounding, instructed | the T3a items plus one instruction to use the facts | the addressed panelist | uses the fact as given |
| T4 persona | the two panelists' persona cards are swapped | both panelists | sounds like the card it was given |
| T5 hosting | (later) fixed reference panel, only the host varies | host | |
| T6 full show | (later) a free multi-beat episode | all | |

T2 template: the audience line with its opening name removed, put to one panelist as `"{name}, {question}"`. Every
generated line is checked with the show's own addressee detector (`agent/floor/addressee.py`). T2 is a floor
decision plus a reply: the tested panelist may speak or yield the floor. Only T2 offers that choice; T1, T3a/T3b and T4
always ask for a line.

T3 pairs (`build.py`): the facts to edit are taken, in the show's order, from the 6 facts it would offer for the
original audience line (at most 2 per topic, from different Wikipedia pages). For each, the audience asks the
panelist about the fact's subject, its page title without parentheses or numbers, with one of 4 fixed templates
chosen by a seeded hash (e.g. "Ryan, what does the record actually say about Nuclear power? I want the specifics.").
The question never contains a number and is the same on both sides of the pair. The panelist is offered 6 facts: that
fact (original or edited) and the 5 facts from other pages the show ranks most relevant to the question (none holding
the old or new value), with the fact at a seeded position. The two sides differ only in that one fact. The show's
adapters show the same 6 facts in the show's own relevance order.

T3a and T3b share every item except the task, ids and one line. T3a measures whether a system uses the evidence it is
given without being asked; T3b adds the instruction "Use these facts when they are relevant to your reply."
(`items.FACT_INSTRUCTION`, in `meta.fact_instruction`). PlainLLM gets that line right after the facts block. The show's
adapters (worker and Think & Action) already put a rule under the facts ("Use specific numbers ... ONLY from these
facts ..."); on arms whose prompt also has the show's own instruction to ground the line in the checked facts (fast,
concrete_fact_pack), T3b is identical to T3a. On arms without it (production), the line is added right after the fact
block's rule. If a prompt had no fact block, T3b would be marked not applicable for that system and left out of its
score. Each output row records which case applied (`fact_instruction`: show_rule, adapter_line, not_applicable).

T3a/T3b edit rule (`fact_edit.py`): the first number of the fact that can be edited is changed: years minus 10, percentages times 2 (divided by 2 if that would reach 100), other numbers
times 2; units, currency and format are kept. Numbers inside names, ranges, ratios, dates and ages are skipped, and
neither the old nor the new value may appear anywhere else the panelist sees. Implausible edits are dropped and the
next editable number is tried: a number compared with another number of the same kind in its sentence ("rose to 91%
from 55%"), a year in a sentence that names another year, a decade or a century, and any value past an obvious bound
(a percentage of 100 or more, a negative value).
Example: "employers must pay minimum wage unless employees customarily receive more than $30 a month in tips" becomes
"... more than $60 a month in tips".

## Splits

| split | topics | source | history before the audience line | items (T1 / T2 / T3a / T3b / T4) |
|---|---|---|---|---|
| dev | 30 | hand-written (`eval/data/minimal_pairs`) | the recorded show (panel already discussed the topic) | 574 (150 / 120 / 92 / 92 / 120) |
| test | 42 | real audience and debater turns from public debates (`eval/data/public_set`) | the host's two opening lines | 811 (243 / 168 / 116 / 116 / 168) |

Test items start from the host's opening by choice: every test item must be rebuildable from committed data plus the
public corpus, and a recorded show state (made by our own system) exists for only some test topics. So on test the
audience member's line opens the discussion; on dev the panel has already discussed the topic.

Dev has no cause edit. Test lacks the number edit on 2 topics and the entity edit on 7 (not applicable to the turn).
T3a (and T3b) has 46 pairs on 28 dev topics and 58 pairs on 37 test topics; a topic is dropped when no fact gives a usable pair
(no editable number, or fewer than 5 facts from other pages).

Reruns: 3 per item. Paired items share their sampling seed (seed = pair group, role, rerun), so the two sides of a pair
differ only in the edit.

## Files

- `eval/data/bench/dev.jsonl`: the dev items (committed).
  Facts are from Wikipedia, CC BY-SA 4.0, see `meta.fact_sources` (page and URL per fact).
- `eval/data/bench/test_manifest.jsonl`: test item ids, the edit applied (source key, fact indices, old and new
  number) and a content hash. No source text: test items are rebuilt locally into `eval/data/bench/build/` (git-ignored),
  like the public test set itself.

Build (CPU, deterministic):
```bash
python -m eval.bench.build --split dev  --pack <fact pack json>   # rewrites eval/data/bench/dev.jsonl
python -m eval.bench.build --split test --pack <fact pack json>   # needs eval/data/public_set/build/pairs_public.yaml;
                                                                  # checks every item against test_manifest.jsonl
```

## Item format (one JSON object per line, `items.py`)

`item_id`, `task`, `split`, `topic_id`, `topic`, `pair_group` (items that differ only in the edited element),
`version` (`orig`, `paraphrase`, `number`, `stance`, `entity`, `cause`, `addr_commentator`, `addr_guest`,
`fact_orig_<n>`, `fact_edit_<n>`, `persona_orig`, `persona_swap`), `is_control`, `role_under_test`, `personas`
(role: persona card text), `history` (`[{speaker, role, text, procedural?}]`, oldest first; role is host, commentator,
guest or human), `audience_turn`, `fact_pack` (list of fact strings or null), `expected` (e.g. the addressee, the edited
fact with its old and new value), `meta` (reference version of the pair, names, source, recorded show state).

## Systems

```python
respond(obs) -> {"text": str, "speaker": role,
                 "floor_action": "speak" | "yield" | "wait" | "interrupt" | "backchannel" | None}
```
`obs` (`items.observation`) is what the role under test sees; `obs["floor_choice"]` is true on T2. `floor_action` is
None when the system was simply asked for a line; yield, wait and backchannel all count as leaving the floor. Adapters in `systems.py`:

- `plain`: one public prompt (persona card as system message; the transcript, optional facts, and "You are X. Reply
  with your next spoken line only: 1-3 sentences, plain text, no speaker label."), temperature 0.7. On T2 only, one
  line is added after the instruction: "If the question was addressed to someone else, reply with exactly PASS
  instead of a line." A reply that is exactly PASS (any case, spaces, trailing . or !) is a yield with no text.
- `worker`: the show's panel worker prompt and sampling (`eval/model_only.py`). It makes no floor decision, so on
  T2 only its reply is scored.
- `think_action`: the show's Think & Action think call (`eval/think_action_prompt.py`); its decision is the floor
  action. If the panelist does not choose to speak, the text is the worker line (as live, when the floor still goes to
  it); `--no-fallback` leaves it empty.

The show's systems take `--arm production | fast | concrete_fact_pack` (flag presets, `eval/quality_ruler/arms.py`).
Items with facts always reach the show's fact block, whatever the arm. The show only accepts its own persona cards or
the swapped pair.

To add a system: subclass `System`, build the prompts in `requests` and turn the model output into a reply in `finish`
(or override `respond_batch` for a remote service), and register it in `SYSTEMS`.

## Run and score

```bash
python -m eval.bench.run --system plain --model Qwen/Qwen2.5-7B-Instruct --tasks T1,T2,T3a,T3b,T4 --split test \
  --reruns 3 --out eval/results/bench/plain-qwen25-7b
python -m eval.bench.run --system think_action --arm fast --model google/gemma-4-E4B-it --split test --out DIR
python -m eval.bench.run ... --dry-run   # no model: DIR/prompts.jsonl and DIR/prompts/<item>.txt
python -m eval.bench.run ... --fake      # no model: placeholder replies in the real output format
python -m eval.bench.score DIR           # CPU scores, R lines and judge requests -> DIR/score/
```
On the cluster: `deploy/slurm-eval-bench.sh` (`BENCH_CMD`, several commands separated by `;`).

`DIR/outputs.jsonl`: one row per (item, rerun): `text`, `floor_action`, `decision` and `text_source` (think_action),
`raw`, `latency_s` (batch time per request: offline batching, not live latency), `prompt_hash`, `seed`, and every model
call made. `DIR/run.json`: system, arm, model, items file hash, commit.

Scoring (`score.py`): rules where checkable, yes/no judges otherwise, no holistic scores.
- T2 right floor (systems that decide the floor), per panelist per rerun: speak when addressed, yield when not.
  Accuracy is reported for each case and balanced (their mean), so always speaking or always yielding gives 0.5.
  The reply (identification and rubric requests) is judged only on replies given when addressed.
- T3a/T3b follow the fact: the reply states the new value, the old value, both, or neither.
- Unsupported specifics, all tasks: numbers and names in the reply that the panelist was not shown.
- Response shift R: `DIR/score/R/<task>/lines.jsonl`; embed with `python -m eval.responsiveness --out <that dir>
  embed` (GPU) and score again.
- Judge requests: `DIR/score/judge/identification.jsonl` (which input version was this reply answering, A/B in both
  orders, chance 0.5; for T4, which persona card) and `rubric.jsonl` (the quality ruler's kept yes/no items; rerun 0
  only by default, `--rubric-reruns all` for every rerun).

Judges (GPU): `python -m eval.bench.judge --requests DIR/score[,DIR2/score] --model <judge> --tag <name> --tp 2`
answers both files from next-token logprobs (as `eval/judge_context.py` and the quality ruler's jury), checkpointing to
a partial file so a requeued job resumes. The next `python -m eval.bench.score DIR` merges every `answers_<tag>.jsonl`:
identification accuracy J per task and pair type with a 95% CI from resampling topics (the paraphrase control
separately), and rubric pass rates; per judge and pooled.

Task scores (0-100, `score.task_scores`), each the mean of the parts that exist for the run, judges pooled:
- T1: 200 x (J - 0.5) on the non-control edits.
- T2: 200 x (balanced right-floor accuracy - 0.5), for systems that decide the floor; 200 x (J - 0.5) on the
  addressee pair, judged on the replies given when addressed.
- T3a and T3b, each: 200 x (J - 0.5) on the fact pair; 100 x share of replies stating the new value only; 100 x share of replies with
  no unsupported specific.
- T4: 200 x (J - 0.5) on persona attribution; 100 x persona item pass rate.
- overall: the mean over T1, T2, T3a, T3b and T4 (the tasks scored). Scores are clipped to 0-100. The report and the
  leaderboard show T3a and T3b as separate columns. `--tasks T3` selects both.

Outputs: `DIR/score/score.json`, `DIR/score/report.md` (one page per system), and
`python -m eval.bench.score --leaderboard DIR1/score DIR2/score ... --out board.md` (one table across systems).
