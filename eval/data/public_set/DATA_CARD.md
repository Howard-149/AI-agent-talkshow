# Data card: public-data test set

## Sources and licenses
| source | used for | license / terms | decision |
|---|---|---|---|
| Intelligence Squared US debates corpus, via ConvoKit `iq2-corpus` (108 debates, 2006–2015; 26,562 utterances) — Zhang, J., Kumar, R., Ravi, S., Danescu-Niculescu-Mizil, C. *Conversational Flow in Oxford-Style Debates.* NAACL 2016 | topics (motions) + the real human turns | No dataset license is stated in the ConvoKit documentation or the corpus README (ConvoKit's *code* is MIT). Transcripts are IQ2US debate content. | **Used for research; not redistributed.** We publish IDs, offsets, edit diffs and the rebuild script only; users obtain the text from ConvoKit. Cite the paper and ConvoKit (Chang et al., SIGDIAL 2020). |
| ChangeMyView "Winning Arguments" corpus (ConvoKit `winning-args-corpus`) — Tan, Niculae, Danescu-Niculescu-Mizil, Lee. WWW 2016 | planned backup turns (posts nearest to a motion, embedding cosine ≥ 0.6) | No dataset license stated in the ConvoKit documentation; Reddit content. | **Not used**: IQ2 alone filled every domain. |
| BBC Question Time QT30 (Hautli-Janisz et al., LREC 2022), AIFdb | planned held-out slice | AIFdb: "These corpora are free for academic use (for non-academic use, please contact Prof. Chris Reed)." No redistribution grant; BBC broadcast content. | **Skipped** (rule: skip if not redistributable). |
| spaCy 3.8.16 + en_core_web_sm 3.8.0 (MIT) | sentence splitting, NER, numeric entities | MIT | used (CPU) |

Tool versions: convokit 4.1.2; corpus `utterances.jsonl` sha1 26fa87ded161 (first 12 hex). Seed 20261006.

## Sampling
- 108 motions → 6 domains (LLM label from the title, logged in `domains.tsv`; keyword cross-check disagrees on 1/108:
  "Millennials Don't Stand A Chance", economy vs society — not in the test set). Domain sizes: economy 16,
  foreign_security 32, politics_law 20, religion_ethics 12, science_health 16, society_education 12.
- Excluded as near-duplicates of development topics: 3 (Abolish the Minimum Wage, Too Many Kids Go To College, Legalize Drugs).
- Candidate spans: 2,509 trimmed spans from audience turns and opening statements; 403 pass the filters
  (fail reasons, not exclusive: no question/example 1,616; no figure 1,128; no PERSON/ORG/GPE 629; profanity 15),
  covering 103 motions. Eligible motions per domain: 15 / 31 / 17 / 11 / 15 / 11.
- Walk + manual curation (`curation.yaml`, curator = an LLM agent, every decision with a reason): 50 topics reached,
  8 rejected (no self-contained span ×3, debate/speaker references ×2, moderator announcement ×1, spliced-title
  artifact ×1, no concrete figure ×1), **42 accepted = 7 per domain**. 16 further topics were curated before the walk
  order was made stable and are not in the final walk — kept in curation.yaml as reserves.

## The test set
- 42 topics; turn-2 source: **6 audience questions, 36 opening-statement spans** (per domain, audience:
  economy 0, foreign_security 2, politics_law 1, religion_ethics 0, science_health 1, society_education 2).
  Why so few audience turns: almost every IQ2 audience question addresses a debater ("the comment you made…", "your
  partner…"), and those sentences must go (Ryan never said those things); what remains is rarely 60–180 words with a
  figure and an entity. Opening statements are real spoken turns too, but by public figures arguing a side, so the test
  set is "real debate speech", not "lay audience speech" — a limitation to state in the paper.
- Span length 63–177 words, median 120 (development turn 2: ~25–40 words). 33/42 contain a digit figure; the other 9 a
  spelled-out figure (e.g. "six hundred million dollars").
- 35/42 spans contain a question; 7 were accepted as concrete examples without a question (judged by hand, logged:
  iq2-5703, 6098, 6660, 11767, 11978, 11136, 897) — why_how is not applicable to those.

## Edits
All seven pre-prefix types for all 42 topics; every gate passes (`build.py apply`, which also writes these numbers to `edit_stats.json`):

| type | n | n/a | length ratio (min / median / max) | token edit ratio (median / max) | kinds |
|---|---|---|---|---|---|
| paraphrase | 42 | 0 | 0.95 / 0.99 / 1.03 | 0.10 / 0.16 | — |
| negate | 42 | 0 | 0.90 / 1.00 / 1.03 | 0.06 / 0.19 | — |
| number | 40 | 2 (only "figure" is a label: 9/11, Prop 209) | 0.98 / 1.00 / 1.02 | 0.01 / 0.05 | magnitude 36 · date 4 |
| entity | 35 | 7 (1 no named entity; 6 entity policy) | 0.99 / 1.00 / 1.02 | 0.01 / 0.09 | — |
| anecdote | 42 | 0 | 0.94 / 1.00 / 1.08 | 0.12 / 0.19 | case 22 · story 14 · statistic 6 |
| causal_flip | 42 | 0 | 0.93 / 1.01 / 1.09 | 0.08 / 0.17 | reattribute 35 · invert 7 |
| why_how | 35 | 7 (no question in span) | 0.99 / 1.00 / 1.03 | 0.03 / 0.08 | — |

Process: three LLM agents drafted the edits against a written spec (`build/EDIT_SPEC.md`: the edit-type rules + the
review rules); the curator then read every edit (`build/edits_review.md`, git-ignored) and changed 6:
number edits that only relabel (9/11 → 7/7, Prop 209 → Prop 54) set to n/a; an anachronistic date (since 2003 → 2023,
for a 2009 debate) → 1983; Boeing → Airbus made consistent (aircraft models too); a negation that made a quoted
official contradict his own quote (iq2-4134); two causal flips with a dangling referent / participle (iq2-15726, iq2-0).
Anecdote facts are well-known (e.g. the 2010 Afghan war logs naming informants, the Anglo-Afghan wars, Begin's Nobel,
Nevada's licensed brothels since 1971), hedged ("I read", "by most estimates"), or personal stories; one is
anachronistic for its debate (iq2-8444: the WHO cancer agency's 2015 processed-meat classification, debate 2013).

### Decisions applied as defaults (reversible — see curation/edits history)
1. **Entity policy** (option b): entity swaps of places, products, historical actors and real facts are kept
   (counterfactual by design, as in the development set). Swaps that pinned a finding, ruling, quote or action on a different real
   person or organisation were replaced by a swap of another entity in the same span (11 topics: e.g. the TV show Nixon
   pre-empted, Bonanza → Gunsmoke; "Caesar crossed the Rubicon" → "Hannibal crossed the Alps"; a real official's name
   → a fictional one; CBO war-cost places), or set n/a where every named entity carries a study finding (6 topics:
   iq2-8444, 11767, 11978, 21166, 897, 9739). Entity: 35 edits, 7 n/a.
2. **Implied-cause causal flips** (iq2-16966, 5180, 12797) are kept and tagged `implied_cause` in the built meta, so the
   analysis can report causal_flip with and without them.
3. Source mix, spelled-out-figure rule, turn-1 template, LLM domain labels / curation — as documented above.

- Release note: `edits.yaml` stores no source spans, but replacement strings sometimes keep a few original words
  around the change (e.g. a paraphrase "interrupting the much-loved popular horse opera"). These are short phrase-level
  fragments; trim them further before a public release if counsel prefers.

## Human anchors (`build.py anchors` → anchors.jsonl = IDs + offsets; text in build/anchors.jsonl)
- For each test motion: the first run of 4 debater turns in the IQ2 discussion segment whose SIDES alternate
  (for/against), each trimmed to whole sentences (≥ 25, ≤ 120 words; interrupted fragments dropped), preceded by the
  last 1–2 sentences of the moderator's previous turn as the host line. Sides map to panelists (first side → Ryan,
  other → Amy; a side's two debaters speak as one panelist). Names normalised: opponents → that panelist's name, a
  teammate → "my colleague", the moderator → Lessac.
- 37/42 motions have an anchor (no alternating window: iq2-21535, 7859, 13452, 7204, 20897).
- `build.py anchors --v2` (anchors_v2.jsonl, IDs + offsets): the most interactive window per motion, turns ≤ 80
  words; these are the anchors of the final quality ruler.
- Valid anchors for the content dimensions (specificity, new information, interaction, argument quality, repetition,
  flow); NOT for persona fidelity/distinctiveness (the debaters are not our personas, and a side can be two people).

## Limitations
1. **Expert debate speech, not lay guest speech.** 36/42 turns are opening-statement spans by public figures arguing a
   side; only 6 are audience questions (most audience questions address the debaters and fail the "no reference to
   other speakers" rule). Turns are long (63–177 words, median 120; development set ≈ 25–40), spelled-out figures count as
   figures, and turn 1 is a template built from the motion title, not source text.
2. **LLM-made decisions without a human spot-check.** Domain labels, span curation (picks, trims, rejects) and the
   first drafts of every edit were made by LLMs (one curator agent + three drafting agents); every edit was re-read by
   the curator, but no human has reviewed them. The entity policy is a default (swaps that would attribute a finding,
   quote or action to another real person/organisation were replaced or set n/a).
3. **Anchors and missing types.** Human anchors exist for 37/42 motions; each debate side (sometimes two people) is
   mapped onto one panelist, so anchors are valid for the content dimensions only, not for persona fidelity or
   distinctiveness. The prefix-dependent types (callback, concede, panelist_edit) are not built until the live replay
   prefixes are recorded.

## Other known limitations
- Mostly expert/opening speech (36/42); long turns (median 120 words) — longer than the development turns, which changes
  the difficulty (more for the panel to engage with; a single edit is a smaller share of the turn).
- Domain labels and curation are by one LLM curator; human spot-checks are pending.
- Transcribed speech keeps disfluencies and some transcript quirks (e.g. "totaled16.7 trillion" spacing errors); only
  fillers (uh/um), curly quotes and repeated dashes are normalised.
- Post-prefix types (callback, concede, panelist_edit) are not built yet: they need the live replay prefixes.
