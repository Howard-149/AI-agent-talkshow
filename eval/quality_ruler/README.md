# Quality ruler

Scores panel dialogue line by line and was validated before use. How to run it: `eval/README.md`.

## Files
| file | what it holds |
|---|---|
| `items.py` | wording of every yes/no question, the persona cards, the generic-filler pool |
| `jury.py` | one judge model per run; answers read from next-token logprobs (no sampling); resumes after a requeue |
| `auto.py` | counted metrics (no judge): figures and names, generic similarity, novelty, repetition, opener reuse |
| `perturb.py` | known-bad edits of an episode, each with the measures it should lower |
| `stats.py` | Krippendorff's alpha, Bradley–Terry with bootstrap, length-controlled logistic fit, Pearson r |
| `episodes.py` | episode format; model-only generation, human anchors, live session logs |
| `arms.py` | generation knobs for model-only episodes (`production` = no knobs) |
| `ruler.py` | the command line: generate, live, embed, jury, score, and the validation stages |

## Measures
An episode is one panel exchange (`lines` = role, speaker, text; `scored` = indices of the panel lines to judge).

Kept after validation (judged, averaged over two judges, Qwen2.5-72B-AWQ and Llama-3.3-70B-AWQ):
- **specific**: three yes/no items (fits any topic, concrete detail, on the point). The two judges agree on averages
  but not line by line, so `score` also reports each judge.
- **distinct** (who is speaking): with names hidden, the judge picks which persona card said the line; 0.5 = chance.
- **fidelity** (sounds like the persona): three items per persona.
- **host.H1** (host follows up): only on real host replies (live episodes); it separates a real reply from a template.
- **argument** is a guardrail only (claim, reason, evidence, other side): it must not drop.

Counted by code (no judge): figures and names per 100 words, 1 − similarity to generic filler, share of novel lines,
distinct-2, 1 − repeated-opener share, words per line, and across episodes the opener reuse (`reuse.*`, rarefied to
100-line samples). Judges could not see word-for-word repeats or tell real engagement from "I hear Ryan's point", so
novelty, repetition and interaction are not judged. Listening is measured by the listening test instead.

## Validation checks (`ruler.py --v2 validate`)
- (a) catches damage: each known-bad perturbation must lower its own measure in ≥ 90% of cases;
- (b) humans score higher: real debate anchors beat the system on content measures, raw, length-controlled and against
  length-matched anchors;
- (c) judges agree: Krippendorff's α ≥ 0.4 per measure.
The report also has judge sanity on real speakers (d), length checks (e), redundancy between measures (f) and
test–retest (g).
