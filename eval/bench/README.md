# Talk show benchmark

Any talk show system (a plain LLM, the show's panel worker, the show's Think & Action floor, or your own) answers the
same fixed items and is scored the same way. An item is one turn of a show: a topic, persona cards, the transcript so
far, an audience member's new line, and sometimes a list of checked facts. Most items come in counterfactual pairs: one
element of the input is edited, and a good reply should change with it (or should not, for a control edit).

## Tasks

| task | what is edited | role tested | a good reply |
|---|---|---|---|
| T1 listening | the audience line: a number, the stance, a named entity, a cause; a paraphrase as control | the panelist the line addresses | follows the edit; ignores the paraphrase |
| T2 addressing | the audience line names panelist A or panelist B | both panelists | the named panelist answers, the other holds back |
| T3 grounding | one number in one offered fact (up to 2 facts per topic, one at a time) | the addressed panelist | uses the fact as given |
| T4 persona | the two panelists' persona cards are swapped | both panelists | sounds like the card it was given |
| T5 hosting | (later) fixed reference panel, only the host varies | host | |
| T6 full show | (later) a free multi-beat episode | all | |

T2 template: the audience line with its opening name removed, put to one panelist as `"{name}, {question}"`. Every
generated line is checked with the show's own addressee detector (`agent/floor/addressee.py`).

T3 rule (`fact_edit.py`): the show offers its usual 6 facts for that turn; in reading order, the first number of a fact
that can be edited is changed: years minus 10, percentages times 2 (divided by 2 if that would reach 100), other numbers
times 2; units, currency and format are kept. Numbers inside names, ranges, ratios, dates and ages are skipped, and
neither the old nor the new value may appear anywhere else the panelist sees.
Example: "employers must pay minimum wage unless employees customarily receive more than $30 a month in tips" becomes
"... more than $60 a month in tips".

## Splits

| split | topics | source | history before the audience line | items (T1 / T2 / T3 / T4) |
|---|---|---|---|---|
| dev | 30 | hand-written (`eval/data/minimal_pairs`) | the recorded show (panel already discussed the topic) | 476 (150 / 120 / 86 / 120) |
| test | 42 | real audience and debater turns from public debates (`eval/data/public_set`) | the host's two opening lines | 689 (243 / 168 / 110 / 168) |

Dev has no cause edit. Test lacks the number edit on 2 topics and the entity edit on 7 (not applicable to the turn).
T3 drops topics where no offered fact has an editable number (1 dev, 2 test).

Reruns: 3 per item. Paired items share their sampling seed (seed = pair group, role, rerun), so the two sides of a pair
differ only in the edit.

## Files

- `eval/data/bench/dev.jsonl`: the dev items (committed). T3 items contain sentences from Wikipedia
  (CC BY-SA 4.0); each item lists the source page and URL in `meta.fact_sources`.
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
`version` (`orig`, `paraphrase`, `number`, `stance`, `entity`, `cause`, `addr_commentator`, `addr_guest`, `fact_orig`,
`fact_edit_1`, `fact_edit_2`, `persona_orig`, `persona_swap`), `is_control`, `role_under_test`, `personas`
(role: persona card text), `history` (`[{speaker, role, text, procedural?}]`, oldest first; role is host, commentator,
guest or human), `audience_turn`, `fact_pack` (list of fact strings or null), `expected` (e.g. the addressee, the edited
fact with its old and new value), `meta` (reference version of the pair, names, source, recorded show state).

## Systems

```python
respond(obs) -> {"text": str, "speaker": role, "floor_action": "speak" | "wait" | "interrupt" | "backchannel" | None}
```
`obs` (`items.observation`) is what the role under test sees. `floor_action` is None when the system was simply asked for
a line. Adapters in `systems.py`:

- `plain`: one public prompt (persona card as system message; the transcript, optional facts, and "You are X. Reply
  with your next spoken line only: 1-3 sentences, plain text, no speaker label."), temperature 0.7.
- `worker`: the show's panel worker prompt and sampling (`eval/model_only.py`).
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
python -m eval.bench.run --system plain --model Qwen/Qwen2.5-7B-Instruct --tasks T1,T2,T3,T4 --split test \
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
- T2 right responder (systems that decide the floor): the named panelist speaks and the other does not.
- T3 follows the fact: the reply states the new value, the old value, both, or neither.
- Unsupported specifics, all tasks: numbers and names in the reply that the panelist was not shown.
- Response shift R: `DIR/score/R/<task>/lines.jsonl`; embed with `python -m eval.responsiveness --out <that dir>
  embed` (GPU) and score again.
- Judge requests: `DIR/score/judge/identification.jsonl` (which input version was this reply answering, A/B in both
  orders, chance 0.5; for T4, which persona card) and `rubric.jsonl` (the quality ruler's kept yes/no items). The judge
  runner that answers them is not written yet.
