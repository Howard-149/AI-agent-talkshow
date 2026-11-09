# Public-data test set: real human turns from Intelligence Squared US debates

The 30 hand-written topics in `eval/data/minimal_pairs/` are the development set. This directory builds the test set:
42 debate motions (7 per domain x 6 domains), each paired with a REAL spoken turn from the same debate, edited minimally
for every edit type. See DATA_CARD.md for sources, licenses, filters and counts.

## Nothing copyrighted is committed
The IQ2 transcripts carry no redistribution license (see DATA_CARD.md), so the repo stores only:
corpus IDs and character offsets (`selection.jsonl`, `order.json`, `anchors*.jsonl`), manual curation decisions
(`curation.yaml`), domain labels (`domains.tsv`), and edits as diffs against the source span (`edits.yaml`: offset,
length, sha1 of the replaced source text, and OUR replacement text). `build.py` rebuilds every text from the ConvoKit
download into `build/` (git-ignored) and refuses to run if any source hash differs. Never commit anything from `build/`.

## Rebuild
Use a separate environment (convokit pins its own dependencies):
```
python -m venv .venv-pubdata && .venv-pubdata/bin/pip install convokit==4.1.2 spacy==3.8.16 pyyaml
.venv-pubdata/bin/python -m spacy download en_core_web_sm-3.8.0 --direct   # or: spacy download en_core_web_sm
P=.venv-pubdata/bin/python
$P -m eval.data.public_set.build select       # domains → filters → candidates → stable walk + curation → selection.jsonl
$P -m eval.data.public_set.build apply        # selection + edits.yaml → checks → build/pairs_public.yaml
$P -m eval.data.public_set.build anchors      # human anchors → build/anchors.jsonl
$P -m eval.data.public_set.build anchors --v2 # the most interactive windows → build/anchors_v2.jsonl (final ruler)
```
`select` downloads `iq2-corpus` (8.7 MB) on first use and takes ~3 min on one CPU (spaCy en_core_web_sm). It also
writes `candidates_log.jsonl` and `selection_stats.json`; `apply` writes `edit_stats.json` (all git-ignored logs).
`apply` writes `build/pairs_public.yaml` in the same schema as the development `pairs.yaml`
(`topics[].turn1/turn2/meta`), so `eval.model_only --pairs`, `make_pairs.py` (via `TALKSHOW_PAIRS`) and the ruler's
`generate --pairs` can use it directly.

Authoring (maintainers only): edits are written as `old → new` ops in `build/edits_src*.yaml` (git-ignored, quotes the
source), then `build.py compile` turns them into the offset/hash form of `edits.yaml`. Every edit is checked by `apply`.

## Pipeline (fixed: SEED 20261006)
1. **Domains** (`domains.tsv`): each of the 108 IQ2 motions labelled with one of 6 domains by an LLM from the title
   alone (logged). A keyword cross-check (`keyword_check` in `order.json`) flags disagreements for spot-checking.
   3 motions that duplicate development topics (minimum wage, college, drug legalisation) are excluded.
2. **Candidates** (`build.py select`): audience turns (speaker "audience") and opening statements (segment 0, for/against),
   normalised (curly quotes, standalone uh/um, repeated dashes). Sentences are dropped if they introduce the speaker
   (also keeps audience members' names out), manage the floor, reference the debaters / the debate / the room, name our
   cast (Ryan, Amy, Lessac), are cut off ("--"), or carry transcript artifacts (page headers, dates, the motion title
   spliced in quotes). Contiguous runs of kept sentences are trimmed from the front to 60–180 words. A span passes if it
   has ≥ 1 figure (digit / % / currency, or a spelled-out spaCy PERCENT/MONEY/QUANTITY/CARDINAL other than "one"),
   ≥ 1 PERSON/ORG/GPE entity, a question or a concrete-example marker, and no profanity. Rank: audience first, ending in
   a question, then (figures + entities) per 100 words.
3. **Walk + curation**: per domain, topics with an audience candidate first, then the rest, each group ordered by a fixed
   per-topic key (sha256 of seed:id) so one topic's eligibility never reorders the others. Each topic reached is curated
   (`curation.yaml`): pick a candidate (pinned by utterance id + offset) and a sentence range, or reject with a reason;
   the walk continues until 7 topics per domain are accepted.
4. **Edits** on the real span (`edits.yaml`), all types, with the review rules enforced by `apply` (length 0.8–1.25,
   token edit distance ≤ 20%, no markdown, per-type sanity checks).
5. **Human anchors** (`build.py anchors`, for the quality ruler): one real IQ2 panel exchange per motion
   (4 side-alternating debater turns + the moderator's question), reformatted to the show's beat format with names
   normalised to Ryan / Amy / Lessac (37/42 motions). `--v2` picks the most interactive window instead. See DATA_CARD.md.

## Show framing
- turn 1 (a template, not source text): `I want to put a motion to the panel: "<motion>." Where do the two of you land?`
  The live prefix records the panel's beat on it.
- turn 2 = the real span with a vocative prepended: `Ryan, <span>` (orig and every edit) / `Amy, <span>` (addressee).
