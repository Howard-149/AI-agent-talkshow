# Evaluation toolkit

Two tools tell whether a change to the show helps:

- the **listening test**: does the panel react to what the human actually said?
- the **quality ruler**: are the panel lines specific, in character, and free of templates?

Both run on Babel with the SLURM wrappers below. Every tool also runs as `python -m eval.<tool>` from the repo root and
has `--help`. A single "compare two versions" entry point is planned for a later change.

The benchmark built on them (fixed items, any system, counterfactual pairs, hybrid scoring) is in
`eval/bench/README.md`.

## Listening test

We replay the same show up to one human turn, then change one detail of that turn and watch the reply.
Example: the human says "AI will replace 40% of jobs" in one run and "AI will replace 4% of jobs" in another.
A panel that listens answers the two differently, for example by arguing with the number.

Each topic has the original turn and edited versions (`eval/data/minimal_pairs/pairs.yaml`, 30 hand-written topics):
paraphrase (same meaning, the noise control), negate, number, entity, addressee (the question goes to the other
panelist), persona swap (the panelists swap personalities), plus graded number edits. Every version is replayed K = 5
times from the same saved show state (`prefix/<topic>.json`), so only the edit differs.

Two numbers per edit type:

- **R, response shift** (`eval.responsiveness`): how far the replies move after the edit, divided by how far they move
  when we simply rerun the original. R = 1 means the edit changed nothing beyond chance; the paraphrase should sit near 1.
- **J, judge accuracy** (`eval.judge_context`): an open judge model sees both versions of the human turn and one reply,
  and guesses which version the reply answered. 0.5 is a coin flip: the reply carries no trace of the edit.

`eval.responsiveness` also reports a log-likelihood shift (KL_lp) and "landing" (does the addressed panelist speak
first). `eval.model_only` runs the same prompts on the model alone, without the live show, to separate the model from
the show's machinery. With the Think & Action floor the selected panelist speaks the `content` of its own think call,
so `eval.model_only --path think_action` (and `ruler generate --path think_action`) build that think call instead of
the worker prompt (`eval/think_action_prompt.py`: the production `_think_and_act` with a recording client, so prompt,
history and sampling match the live show, and the `TALKSHOW_*` flags apply as live). The public test set (42 real debate turns, `eval/data/public_set/`) is built locally and
never committed.

## Quality ruler

Judge models answer yes/no questions about each panel line; code counts the rest. Every measure was tested before use,
and only those that passed are kept:

| kept measure | how |
|---|---|
| specific | 3 yes/no items: fits any topic, a concrete detail, on the point (each judge reported too) |
| who is speaking | names hidden, the judge picks the persona card; 0.5 = chance |
| sounds like the persona | 3 yes/no items per persona |
| host follows up | 1 yes/no item, only on real host replies |

Argument quality is a guardrail: it must not drop. Counted by code, no judge: figures and names per 100 words,
similarity to generic filler, new versus repeated lines, repeated openers, words per line, and opener reuse across
episodes ("I hear Ryan's point..."). Judges are Qwen2.5-72B-AWQ and Llama-3.3-70B-AWQ.

A measure had to pass three checks (`ruler.py --v2 validate`):

1. **catches damage**: when we break an episode on purpose (filler lines, swapped speaker names, a copied line,
   reasons stripped), the measure for that damage drops in at least 90% of cases;
2. **humans score higher**: real debaters on the same motions beat our show, also after controlling for line length;
3. **judges agree**: Krippendorff's alpha of at least 0.4 between the judges.

Details: `eval/quality_ruler/README.md`.

## Running on Babel

Once per shell (the wrappers carry no partition or QOS; pick yours):

```bash
cd ~/AI-agent-talkshow && conda activate talkshow
export TALKSHOW_SLURM_PARTITION=<partition> TALKSHOW_SLURM_QOS=<qos>
SB="sbatch --export=ALL ${TALKSHOW_SLURM_PARTITION:+-p $TALKSHOW_SLURM_PARTITION} ${TALKSHOW_SLURM_QOS:+--qos $TALKSHOW_SLURM_QOS}"
```

Models load from `HF_HUB_CACHE` (default `${HF_HOME:-~/.cache/huggingface}/hub`); the ruler wrapper runs offline, so
download the judges first. If vLLM cannot read the Gemma-4 config, see `talkshow_tf_overlay_pythonpath` in
`deploy/slurm_common.sh`.

### Listening test, live show

```bash
# 1. Replay scripts: one per topic x edit x resample, for the version to test.
python -m eval.data.minimal_pairs.make_pairs conditions -k 5
#    Flags for this version only (two versions can share a job): add --arm B --flags TALKSHOW_ADDRESSEE_FLOOR=1,...
# 2. Live sessions (1 GPU each; split the scripts over several jobs).
GHOST_SCRIPTS="$(ls eval/data/minimal_pairs/cond/*.yaml)" GHOST_OUT=logs/listen GHOST_CLIENT_ARGS="--stop-after-panel 4" \
  $SB deploy/slurm-eval-ghost.sh
# 3. Score.
python -m eval.responsiveness --out eval/results/listen collect logs/listen      # add --arm B for one arm
LISTEN_OUT=eval/results/listen $SB deploy/slurm-eval-listen.sh                   # embed, logprob, report
LISTEN_OUT=eval/results/listen LISTEN_STAGES=judge $SB --gpus=2 deploy/slurm-eval-listen.sh
```

New topics need saved show states first: `make_pairs prefix`, run those scripts with the ghost wrapper, then
`make_pairs extract <log dir>`.

### Listening test, model only

```bash
LISTEN_OUT=eval/results/listen-model LISTEN_STAGES="generate embed logprob report" $SB deploy/slurm-eval-listen.sh
# Think & Action think content instead of the worker prompt (flags as in .env or exported):
LISTEN_PATH=think_action LISTEN_OUT=eval/results/listen-ta LISTEN_STAGES="generate embed logprob report" \
  $SB deploy/slurm-eval-listen.sh
LISTEN_OUT=eval/results/listen-model LISTEN_STAGES=judge $SB --gpus=2 deploy/slurm-eval-listen.sh
```

### Quality ruler

```bash
R=eval/results/ruler-x
# Episodes: model-only (needs build/pairs_public.yaml, see eval/data/public_set/README.md) or from live sessions.
RULER_CMD="generate --model google/gemma-4-E4B-it -k 2 --out $R/production.jsonl" RULER_OVERLAY=1 \
  $SB --gpus=1 deploy/slurm-eval-ruler.sh
python -m eval.quality_ruler.ruler live logs/listen --arm B --out $R/live_B.jsonl
# Embeddings (counted metrics), then each judge.
RULER_CMD="embed --episodes $R/production.jsonl,$R/live_B.jsonl" $SB --gpus=1 deploy/slurm-eval-ruler.sh
for J in "Qwen/Qwen2.5-72B-Instruct-AWQ qwen72" "casperhansen/llama-3.3-70b-instruct-awq llama70"; do set -- $J
  RULER_CMD="jury --model $1 --tag $2 --dims specific,argument,fidelity,distinct,host --repeat 0 \
    --episodes $R/production.jsonl,$R/live_B.jsonl" $SB deploy/slurm-eval-ruler.sh
done
python -m eval.quality_ruler.ruler score $R/production.jsonl $R/live_B.jsonl
```

### Fact pack: production vs fast vs fast + pack

Arms `production`, `fast` (the live FAST flags, `deploy/arms/dialogue-quality.arms`) and `concrete_fact_pack` (fast +
`TALKSHOW_FACT_PACK`). The two live-flag arms build each prompt with the live worker's own code, so an offline episode
sees exactly the live prompt. The pack path comes from the environment. Topics: the 30 tuning topics plus the 42 public
ones (a comma list of pairs files).

```bash
export TALKSHOW_FACT_PACK=<path to the fact pack json> TALKSHOW_ROOT=$PWD
R=eval/results/ruler-factpack P=eval/data/minimal_pairs/pairs.yaml,eval/data/public_set/build/pairs_public.yaml
# CPU first: every prompt, no model (read <dir>/*.txt)
python -m eval.quality_ruler.ruler generate --arm concrete_fact_pack -k 1 --topics nuclear-power,iq2-1595 \
  --pairs $P --dry-run $R/dry
for A in production fast concrete_fact_pack; do
  RULER_CMD="generate --model google/gemma-4-E4B-it -k 2 --arm $A --pairs $P --out $R/$A.jsonl" RULER_OVERLAY=1 \
    $SB --gpus=1 deploy/slurm-eval-ruler.sh
done
# then embed + the two juries over the three files as above, and
python -m eval.quality_ruler.ruler score $R/production.jsonl $R/fast.jsonl $R/concrete_fact_pack.jsonl  # adds specifics.*
python -m eval.quality_ruler.specifics $R/production.jsonl $R/fast.jsonl $R/concrete_fact_pack.jsonl --out $R/specifics.json
```

`specifics.py` counts, per arm, specifics per 100 words, the share supported by the pack or by an earlier line of
someone else, and the share of lines with at least one unsupported specific. Every arm is checked against the same pack.

Validate the ruler again (after changing items, judges or perturbations):

```bash
V=eval/results/ruler-validate
# needs build/anchors_v2.jsonl and model-only production episodes in $V/system.jsonl
python -m eval.quality_ruler.ruler --root $V --v2 build2 --live-logs logs/listen --live-arm A
RULER_CMD="--root $V --v2 rewrite --model Qwen/Qwen2.5-72B-Instruct-AWQ" $SB deploy/slurm-eval-ruler.sh
python -m eval.quality_ruler.ruler --root $V --v2 build2 --live-logs logs/listen --live-arm A   # merges rewrites
RULER_CMD="--root $V --v2 embed" $SB --gpus=1 deploy/slurm-eval-ruler.sh
RULER_CMD="--root $V --v2 jury --model Qwen/Qwen2.5-72B-Instruct-AWQ --tag qwen72 \
  --dims specific,argument,fidelity,distinct,interaction2,repetition2,host" $SB deploy/slurm-eval-ruler.sh
# ... the same jury command for Llama (--tag llama70)
python -m eval.quality_ruler.ruler --root $V --v2 validate
```

## Output files

| file | content |
|---|---|
| `<out>/lines.jsonl` | one panel line per row: topic, edit type, resample k, position after the edit, speaker, text |
| `<out>/sessions.json` | every session found, kept or not, and why |
| `<out>/responsiveness.json` | R with 95% CI per edit type and position, KL_lp, landing |
| `<out>/judge.json` | J with 95% CI per edit type (`judge.jsonl`: every judge answer) |
| `$R/*.jsonl` | episodes: lines (role, speaker, text) and the indices to score |
| `$R/jury_<tag>.jsonl` | every judge answer (probability of yes, or of A) |
| `$R/scores.json` | per episode file: kept measures, each judge's specific, counted metrics, opener reuse |
| `$V/VALIDATION.md` | the validation checks, plain summary first |

## Cost (rough, from our runs)

| step | GPUs | time |
|---|---|---|
| live replay sessions | 1 per job | about 55 sessions per GPU-hour; 30 topics x 7 edits x K=5 = 1,050 sessions, about 20 GPU-hours |
| embed + logprob + report | 1 | 5 to 20 min |
| listening judge (72B) | 2 | 5 to 15 min |
| model-only generation (600 replies) | 1 | 10 to 15 min |
| ruler generation (42 motions x K=2) | 1 | about 5 min |
| ruler jury, one judge, about 100 episodes | 2 | 10 to 20 min |
| ruler validation jury, one judge (about 5,700 lines) | 2 | 1.5 hours |
| collect, score, validate | CPU | seconds to a minute |
