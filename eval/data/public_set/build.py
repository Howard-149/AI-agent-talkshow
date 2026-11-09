#!/usr/bin/env python3
"""
Public-data test set for the listening test (see README.md, DATA_CARD.md).

Topics = Intelligence Squared US (IQ2) debate motions; the edited human turn = a REAL audience question (fallback: an
opening-statement span) from the same debate, trimmed to a self-contained sentence span. Nothing copyrighted is stored
in the repo: selection.jsonl keeps corpus IDs + character offsets, edits.yaml keeps diff operations against the span,
and this script rebuilds the texts from the ConvoKit download.

Stages (fixed seed; CPU only — spaCy en_core_web_sm for sentences/entities):
    python -m eval.data.public_set.build select   # domains → candidates → 7 topics/domain
                                                  # → selection.jsonl (+ candidates_log.jsonl, git-ignored)
    python -m eval.data.public_set.build apply    # selection + edits.yaml → checks →
                                                  # build/pairs_public.yaml (raw text; git-ignored)
    python -m eval.data.public_set.build anchors [--v2]   # human anchors for the quality ruler → build/anchors*.jsonl
(run with the separate environment from README.md: convokit + spaCy)
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
BUILD = HERE / "build"  # raw-text outputs; git-ignored
CORPUS = "iq2-corpus"
SEED = 20261006
LEGACY = False  # --legacy: reproduce the ranking the first curation pass was reviewed on (no spliced-title filter)
PER_DOMAIN = 7
MIN_W, MAX_W = 60, 180
PROFANITY = {"fuck", "fucking", "shit", "bullshit", "damn", "goddamn", "ass", "asshole", "bitch", "crap", "hell"}
# Sentences dropped before span selection: self-introductions (also keeps audience members' names out), greetings and
# floor management, and references to the debaters / the panel / the room.
DROP_SENT = re.compile(
    r"\b(?:my name is|i'?m an? (?:\w+ )?(?:student|professor|teacher|lawyer|doctor|member)|i am an? (?:\w+ )?student|"
    r"i work (?:at|for)|i'?m from|i come from|thank you|thanks|good evening|hi\b|hello|sorry|excuse me|"
    r"my question is|question for|this question|i have a question|i'?d like to ask|i want to ask|first question|"
    r"the gentleman|the lady|the panel|panelists?|your side|the other side|your opponents?|your partner|your colleague|"
    r"your argument|your point|your (?:co-?)?debater|"
    r"you(?:'ve| have)? (?:just )?(?:said|mentioned|talked|admitted|argued|suggested|suggesting|described|saying|raised|"
    r"brought up|pointed out|spoke)|"
    r"(?:he|she|they) (?:was |were )?(?:said|saying|mentioned|argued|talked)|as (?:he|she|you) said|somebody asked|"
    r"(?:mr|ms|mrs|dr)\.|professor|the motion|this motion|tonight|this debate|the debate|the audience|applause|"
    r"laughter|booyah|opposition|proponents?|opponents?|our side|my side|the other team|ladies and gentlemen)",
    re.IGNORECASE,
)
PANEL_NAMES = re.compile(r"\b(?:Ryan|Amy|Lessac)\b")  # our show's cast — a span naming them would corrupt addressee tests
SELF_INTRO = re.compile(r"\bI'?m [A-Z][a-z]+\b[,.]|\bI am [A-Z][a-z]+\b[,.]")
CUT = re.compile(r"(?:--|—|-)\s*$")  # interrupted sentence
ARTIFACT = re.compile(r"Rosenkranz Foundation|Intelligence Squared|\(\d{1,2}/\d{1,2}/\d{2,4}\)|\bPage \d+\b|- \d+ -")
# the transcript sometimes splices the motion title, in quotes, into a sentence (checked per debate in spans_for;
# an unquoted mention of the motion's words is normal speech and is kept)
NUMBER = re.compile(r"\d|%|\$|£|€|\bpercent\b", re.IGNORECASE)
EXAMPLE = re.compile(r"\b(for example|for instance|such as|e\.g\.|take the case|look at)\b", re.IGNORECASE)
FILLER = re.compile(r"(?<![\w'])(uh|um|er|ah)(?![\w'])[,.]?\s*", re.IGNORECASE)


def normalize(text: str) -> str:
    """Deterministic clean-up applied before span selection (part of the rebuild, not an edit): curly quotes,
    standalone fillers (uh/um), doubled spaces."""
    text = text.replace("’", "'").replace("‘", "'").replace("“", '"').replace("”", '"').replace("—", " — ")
    text = FILLER.sub("", text)
    text = re.sub(r"(?:—\s*){2,}", "— ", text)  # transcript "— —" restarts
    return re.sub(r"\s+", " ", text).strip()


def load_corpus():
    from convokit import Corpus, download

    return Corpus(filename=download(CORPUS))


def load_domains() -> dict:
    rows = {}
    for line in (HERE / "domains.tsv").read_text().splitlines():
        if line.startswith("#") or line.startswith("conversation_id") or not line.strip():
            continue
        parts = line.split("\t")
        rows[parts[0]] = {"domain": parts[1], "dev_overlap": parts[2] if len(parts) > 2 else ""}
    return rows


KEYWORDS = {
    "economy": r"econom|tax|bank|wage|market|capitalis|dollar|job|inequality|wall street|amazon|benefits",
    "foreign_security": r"iran|iraq|afghan|china|russia|syria|isis|terror|war\b|drone|pentagon|middle east|policeman|"
                        r"spy|surveil|snowden|cyber|profiling|hamas|palestin|israel|islamists|mexico",
    "science_health": r"science(?! refutes)|tech|google|internet|online|carbon|warming|energy|fracking|organic|"
                      r"genetic|fda|health|obamacare|obesity|end-of-life",
    "religion_ethics": r"relig|islam|god|death is not|sex\b|organs|assisted suicide|death penalty|face\b|babies",
    "society_education": r"college|school|campus|teacher|media|hollywood|millennials|men are|common core|lecture|"
                         r"affirmative|sports|football",
    "politics_law": r"constitution|congress|president|gop|party|amendment|guns|arms|immigra|government|press|pacs|"
                    r"state\b|same-sex|courts|bush",
}


def keyword_check(title: str, domain: str) -> str:
    hits = [d for d, pat in KEYWORDS.items() if re.search(pat, title, re.IGNORECASE)]
    return "" if not hits or domain in hits else "keywords suggest " + "/".join(hits)


# ----------------------------------------------------------------------------------------------- select

def name_pattern(n: str) -> str:
    """Word-boundary pattern for a debater's name; single name parts also match with the last letter dropped
    ("Hannah" also matches the transcript's "Hanna")."""
    stem = n[:-1] if len(n) >= 5 and " " not in n else n
    return rf"\b{re.escape(stem)}\w*\b"


def spans_for(doc, speaker_names: set[str], title: str = ""):
    """Contiguous runs of kept sentences, longest first, each 60–180 words after trimming."""
    sents = list(doc.sents)
    keep = []
    for s in sents:
        t = s.text
        bad = (DROP_SENT.search(t) or ARTIFACT.search(t) or CUT.search(t.strip()) or "--" in t
               or (title and not LEGACY and re.search('"' + re.escape(title[:20]), t, re.IGNORECASE))
               or PANEL_NAMES.search(t) or SELF_INTRO.search(t)
               or len(t.split()) < 3 or not re.search(r"[.?!]\"?$", t.strip())
               or any(n and re.search(name_pattern(n), t) for n in speaker_names))
        keep.append(not bad)
    runs, cur = [], []
    for s, k in zip(sents, keep):
        if k:
            cur.append(s)
        elif cur:
            runs.append(cur)
            cur = []
    if cur:
        runs.append(cur)
    out = []
    for run in runs:
        # shrink from the front until the span fits (the question usually comes last)
        for i in range(len(run)):
            sub = run[i:]
            w = sum(len(x.text.split()) for x in sub)
            if w > MAX_W:
                continue
            if w >= MIN_W:
                out.append((sub[0].start_char, sub[-1].end_char))
            break
    return out


def score_span(doc, text: str) -> dict:
    words = len(text.split())
    ents = [e for e in doc.ents if e.label_ in ("PERSON", "ORG", "GPE", "NORP", "LOC")]
    core = [e for e in ents if e.label_ in ("PERSON", "ORG", "GPE")]
    nums = NUMBER.findall(text)
    # IQ2 transcripts often spell figures out ("twenty percent"): count spaCy numeric entities too (not "one")
    spelled = [e for e in doc.ents if (e.label_ in ("PERCENT", "MONEY", "QUANTITY")
                                       or (e.label_ == "CARDINAL" and e.text.lower() not in ("one", "1")))
               and not NUMBER.search(e.text)]
    return {
        "words": words, "n_numbers": len(nums) + len(spelled), "n_digit_numbers": len(nums),
        "n_spelled_numbers": len(spelled), "n_entities": len(ents), "n_core_entities": len(core),
        "question": "?" in text, "ends_with_question": text.rstrip().rstrip('"').endswith("?"),
        "example": bool(EXAMPLE.search(text)),
        "profanity": any(w.strip(".,!?;:'\"").lower() in PROFANITY for w in text.split()),
        "density": round(100 * (len(nums) + len(ents)) / max(words, 1), 2),
    }


def load_curation() -> dict:
    import yaml

    p = HERE / "curation.yaml"
    return (yaml.safe_load(p.read_text()) or {}).get("topics", {}) if p.is_file() else {}


CURATION: dict = {}


def finalize(order: dict) -> list[dict]:
    """Walk each domain's seeded order; take a topic only if curation.yaml picks a candidate for it, until 7 per domain.
    Un-curated topics reached before the domain is full are listed in build/review.md and the selection is incomplete."""
    global CURATION
    CURATION = load_curation()
    selection, todo = [], []
    for d, topics in order.items():
        n = 0
        for t in topics:
            if n == PER_DOMAIN:
                break
            cur = CURATION.get(t["topic_id"])
            if cur is None:
                todo.append((d, t))
                n += 1  # count it as a slot pending review
                continue
            if "reject" in cur:
                continue
            if "utt" in cur:  # pinned: identity, not rank
                hit = [x for x in t["candidates"] if x["utterance_id"] == cur["utt"] and x["start"] == cur["start"]]
                if not hit:
                    sys.exit(f"{t['topic_id']}: pinned candidate {cur['utt']}@{cur['start']} no longer passes the filters")
                c = dict(hit[0])
            else:
                c = dict(t["candidates"][cur.get("pick", 0)])
            selection.append({"topic_id": t["topic_id"], **c, "trim": cur.get("trim"), "curation_note": cur.get("note", ""),
                              "keyword_check": t["keyword_check"]})
            n += 1
    with (HERE / "selection.jsonl").open("w") as f:
        for r in selection:
            f.write(json.dumps(r) + "\n")
    (HERE / "review_todo.json").write_text(json.dumps([t["topic_id"] for _, t in todo], indent=1))
    print(f"selected {len(selection)}; topics awaiting curation: {len(todo)}", file=sys.stderr)
    return selection


def cmd_select(args) -> None:
    import spacy

    global LEGACY
    LEGACY = bool(getattr(args, "legacy", False))

    nlp = spacy.load("en_core_web_sm")
    corpus = load_corpus()
    domains = load_domains()
    log, per_topic = [], defaultdict(list)
    for cv in corpus.iter_conversations():
        meta = domains.get(cv.id)
        if meta is None:
            sys.exit(f"conversation {cv.id} has no domain label")
        speakers = {u.speaker.id for u in cv.iter_utterances()} - {"audience", "unknown"}
        names = set()
        for sp in speakers:
            names.add(sp)
            names.update(p for p in sp.split() if len(p) > 3)
        for u in cv.iter_utterances():
            st, seg = u.meta.get("speakertype"), u.meta.get("segment")
            if st == "unknown" and u.speaker.id == "audience":
                source = "audience"
            elif st in ("for", "against") and int(seg) == 0:
                source = "opening"
            else:
                continue
            text = normalize(u.text)
            if len(text.split()) < MIN_W:
                continue
            doc = nlp(text)
            for a, b in spans_for(doc, names, cv.meta["title"]):
                span = text[a:b]
                sdoc = nlp(span)
                sc = score_span(sdoc, span)
                ok = (sc["n_numbers"] >= 1 and sc["n_core_entities"] >= 1 and (sc["question"] or sc["example"])
                      and not sc["profanity"])
                row = {"conversation_id": cv.id, "utterance_id": u.id, "source": source, "start": a, "end": b,
                       "title": cv.meta["title"], "domain": meta["domain"], "ok": ok, **sc}
                log.append(row)
                if ok:
                    per_topic[cv.id].append(row)
    BUILD.mkdir(exist_ok=True)
    with (HERE / "candidates_log.jsonl").open("w") as f:  # IDs + offsets + features only (no text)
        for r in log:
            f.write(json.dumps(r) + "\n")

    # rank: audience before opening, ending in a question, then density; keep the top 3 for curation
    ranked = {}
    for cid, rows in per_topic.items():
        rows.sort(key=lambda r: (r["source"] != "audience", not r["ends_with_question"], -r["density"], -r["n_numbers"]))
        ranked[cid] = rows
    by_domain = defaultdict(list)
    for cid, meta in domains.items():
        if meta["dev_overlap"] or cid not in ranked:
            continue
        by_domain[meta["domain"]].append(cid)
    # Walk order per domain: topics with a real audience question first, then the rest; within each group a fixed
    # per-topic key (sha256 of seed:id), so adding or removing one topic never reorders the others.
    order = {}
    for d in sorted(by_domain):
        key = lambda c: hashlib.sha256(f"{SEED}:{c}".encode()).hexdigest()  # noqa: E731
        aud = sorted((c for c in by_domain[d] if ranked[c][0]["source"] == "audience"), key=key)
        rest = sorted((c for c in by_domain[d] if ranked[c][0]["source"] != "audience"), key=key)
        order[d] = [{"topic_id": f"iq2-{c}", "title": ranked[c][0]["title"], "candidates": ranked[c],
                     "keyword_check": keyword_check(ranked[c][0]["title"], d)} for c in aud + rest]
    (HERE / "order.json").write_text(json.dumps(order, indent=1))  # IDs + offsets + features (no text)
    selection = finalize(order)
    stats = {
        "candidate_spans": len(log), "passing_spans": sum(r["ok"] for r in log),
        "topics_with_candidates": len(ranked),
        "eligible_per_domain": {d: len(v) for d, v in sorted(by_domain.items())},
        "selected_per_domain": dict(Counter(r["domain"] for r in selection)),
        "selected_source": dict(Counter(r["source"] for r in selection)),
        "selected_source_per_domain": {d: dict(Counter(r["source"] for r in selection if r["domain"] == d))
                                       for d in sorted(by_domain)},
        "selected_with_digit_number": sum(r["n_digit_numbers"] >= 1 for r in selection),
        "curation_rejects": sum(1 for d in order.values() for t in d if t["topic_id"] in CURATION
                                and "reject" in CURATION[t["topic_id"]]),
        "fail_reasons": dict(Counter(
            k for r in log if not r["ok"] for k, bad in (
                ("no_number", r["n_numbers"] < 1), ("no_entity", r["n_core_entities"] < 1),
                ("no_question_or_example", not (r["question"] or r["example"])), ("profanity", r["profanity"])) if bad)),
        "dev_overlap_excluded": [cid for cid, m in domains.items() if m["dev_overlap"]],
    }
    (HERE / "selection_stats.json").write_text(json.dumps(stats, indent=1))
    print(json.dumps(stats, indent=1))


def span_text(corpus, row: dict) -> str:
    return normalize(corpus.get_utterance(row["utterance_id"]).text)[row["start"]:row["end"]]


def trimmed(nlp, text: str, trim: dict | None) -> str:
    """Apply a curation trim: keep sentences [first, last] (0-based, inclusive) of the candidate span."""
    if not trim:
        return text
    sents = [x.text for x in nlp(text).sents]
    return " ".join(sents[trim.get("first", 0): trim.get("last", len(sents) - 1) + 1])


def cmd_review(_args) -> None:
    """build/review.md (raw text, git-ignored): every candidate of every topic still awaiting curation, sentences
    numbered, so curation.yaml can pick a candidate and a sentence range."""
    import spacy

    nlp = spacy.load("en_core_web_sm")
    corpus = load_corpus()
    order = json.loads((HERE / "order.json").read_text())
    todo = set(json.loads((HERE / "review_todo.json").read_text()))
    BUILD.mkdir(exist_ok=True)
    out = []
    for d, topics in order.items():
        for t in topics:
            if t["topic_id"] not in todo:
                continue
            out.append(f"## {t['topic_id']} [{d}] {t['title']}  {t['keyword_check']}")
            for i, c in enumerate(t["candidates"][:3]):
                sents = [x.text for x in nlp(span_text(corpus, c)).sents]
                out.append(f"### cand {i} ({c['source']}, {c['words']} w, nums {c['n_numbers']}, ents {c['n_core_entities']}, "
                           f"endsQ {c['ends_with_question']})")
                out += [f"[{j}] {x}" for j, x in enumerate(sents)]
            out.append("")
    (BUILD / "review.md").write_text("\n".join(out))
    print(f"{len(todo)} topics → {BUILD / 'review.md'}")


# ----------------------------------------------------------------------------------------------- edits
# Authoring: build/edits_src.yaml (git-ignored; ops quote the source) → `compile` → edits.yaml (committed; char offsets +
# sha1 of the replaced source text + our new text — no source text). `apply` rebuilds every condition from the corpus.

TYPES = ("paraphrase", "negate", "number", "entity", "anecdote", "causal_flip", "why_how")
KINDS = {"anecdote": ("story", "case", "statistic"), "causal_flip": ("invert", "reattribute"),
         "number": ("magnitude", "date")}  # number: magnitude = the analysed edit; date = a year/period shift
LEN_GATE = (0.8, 1.25)
EDIT_CAP = 0.20  # token edit distance / orig tokens; callback / concede (post-prefix) are exempt by design
TURN1 = 'I want to put a motion to the panel: "{title}." Where do the two of you land?'


def sha(text: str) -> str:
    import hashlib as _h

    return _h.sha1(text.encode()).hexdigest()[:12]


def final_spans() -> dict:
    """topic_id → (selection row, trimmed span text), rebuilt from the corpus."""
    import spacy

    nlp = spacy.load("en_core_web_sm")
    corpus = load_corpus()
    out = {}
    for line in (HERE / "selection.jsonl").read_text().splitlines():
        r = json.loads(line)
        out[r["topic_id"]] = (r, trimmed(nlp, span_text(corpus, r), r.get("trim")))
    return out


_NLP = None


def vocative(name: str, span: str) -> str:
    """'Ryan, <span>': the span's first word is lowercased unless it is I/I'm/… or a proper noun / named entity."""
    global _NLP
    if _NLP is None:
        import spacy

        _NLP = spacy.load("en_core_web_sm")
    tok = _NLP(span[:80])[0]
    names = ("PERSON", "ORG", "GPE", "NORP", "LOC", "FAC", "PRODUCT", "EVENT", "WORK_OF_ART", "LAW")
    if not (tok.text in ("I", "I'm", "I've", "I'd", "I'll") or tok.ent_type_ in names
            or (tok.pos_ == "PROPN" and not tok.ent_type_)):
        span = span[0].lower() + span[1:]
    return f"{name}, {span}"


def tok_edit_ratio(a: str, b: str) -> float:
    x, y = a.split(), b.split()
    prev = list(range(len(y) + 1))
    for i in range(1, len(x) + 1):
        cur = [i] + [0] * len(y)
        for j in range(1, len(y) + 1):
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (x[i - 1] != y[j - 1]))
        prev = cur
    return prev[-1] / max(len(x), 1)


def ops_of(entry):
    return entry["ops"] if isinstance(entry, dict) else entry


def cmd_compile(_args) -> None:
    import yaml

    src = {}
    for f in sorted(BUILD.glob("edits_src*.yaml")):  # one file, or parts written in parallel
        src.update(yaml.safe_load(f.read_text())["topics"])
    spans = final_spans()
    out, errors = {}, []
    for tid, types in src.items():
        if tid not in spans:
            errors.append(f"{tid}: not in selection")
            continue
        text = spans[tid][1]
        out[tid] = {"span_sha": sha(text)}
        for t, entry in types.items():
            if entry is None or (isinstance(entry, dict) and entry.get("na")):
                out[tid][t] = {"na": (entry or {}).get("na", "not applicable")}
                continue
            comp = []
            for op in ops_of(entry):
                n = text.count(op["old"])
                if n != 1:
                    errors.append(f"{tid}/{t}: old text occurs {n}× (must be exactly once): {op['old'][:40]!r}")
                    continue
                a = text.index(op["old"])
                comp.append({"at": a, "len": len(op["old"]), "old_sha": sha(op["old"]), "new": op["new"]})
            rec = {"ops": comp}
            if isinstance(entry, dict):
                rec.update({k: v for k, v in entry.items() if k not in ("ops",)})
            out[tid][t] = rec
    if errors:
        sys.exit("COMPILE FAILED\n" + "\n".join(errors))
    (HERE / "edits.yaml").write_text(
        "# GENERATED by build.py compile from build/edits_src.yaml — char offsets into the trimmed span, sha1 of the\n"
        "# replaced source text, and our replacement text. No source text. Rebuild with build.py apply.\n"
        + yaml.safe_dump({"topics": out}, sort_keys=False, allow_unicode=True, width=120))
    print(f"compiled edits for {len(out)} topics → {HERE / 'edits.yaml'}")


def apply_ops(text: str, ops: list[dict]) -> str:
    for op in sorted(ops, key=lambda o: -o["at"]):  # right to left keeps offsets valid
        old = text[op["at"]: op["at"] + op["len"]]
        if sha(old) != op["old_sha"]:
            raise ValueError(f"source mismatch at {op['at']} (corpus version changed?)")
        text = text[: op["at"]] + op["new"] + text[op["at"] + op["len"]:]
    return text


def cmd_apply(args) -> None:
    import yaml

    edits = yaml.safe_load((HERE / "edits.yaml").read_text())["topics"]
    spans = final_spans()
    topics, errors, stats = [], [], defaultdict(list)
    only = set(args.topics.split(",")) if getattr(args, "topics", "") else None
    for tid, (row, span) in spans.items():
        if only and tid not in only:
            continue
        e = edits.get(tid)
        if e is None:
            errors.append(f"{tid}: no edits")
            continue
        if e["span_sha"] != sha(span):
            errors.append(f"{tid}: span changed since the edits were compiled")
            continue
        turn2 = {"orig": vocative("Ryan", span), "addressee": vocative("Amy", span)}
        meta = {"source": row["source"], "conversation_id": row["conversation_id"], "utterance_id": row["utterance_id"],
                "domain": row["domain"], "words": len(span.split()), "na": {}, "kinds": {}, "len_ratio": {},
                "edit_ratio": {}}
        for t in TYPES:
            rec = e.get(t)
            if rec is None:
                errors.append(f"{tid}/{t}: missing")
                continue
            if "na" in rec:
                meta["na"][t] = rec["na"]
                continue
            if rec.get("implied_cause"):
                meta.setdefault("implied_cause", []).append(t)
            if t in KINDS:
                if rec.get("kind") not in KINDS[t]:
                    errors.append(f"{tid}/{t}: kind must be one of {KINDS[t]}")
                meta["kinds"][t] = rec.get("kind")
            try:
                new = apply_ops(span, rec["ops"])
            except ValueError as ex:
                errors.append(f"{tid}/{t}: {ex}")
                continue
            lr, er = len(new) / len(span), tok_edit_ratio(span, new)
            meta["len_ratio"][t], meta["edit_ratio"][t] = round(lr, 2), round(er, 2)
            stats[t].append((lr, er))
            if new == span:
                errors.append(f"{tid}/{t}: no change")
            if not LEN_GATE[0] <= lr <= LEN_GATE[1]:
                errors.append(f"{tid}/{t}: length ratio {lr:.2f} outside {LEN_GATE}")
            if er > EDIT_CAP:
                errors.append(f"{tid}/{t}: token edit distance {er:.2f} > {EDIT_CAP}")
            if re.search(r"[*_#`]", new):
                errors.append(f"{tid}/{t}: markdown characters")
            changed_old = " ".join(span[o["at"]: o["at"] + o["len"]] for o in rec["ops"])
            changed_new = " ".join(o["new"] for o in rec["ops"])
            if t == "number" and not (re.search(r"\d|percent|million|billion|trillion|hundred|thousand|half|third|"
                                                r"quarter|one|two|three|four|five|six|seven|eight|nine|ten|twenty|thirty",
                                                changed_old, re.IGNORECASE)):
                errors.append(f"{tid}/number: the replaced text contains no figure")
            if t == "entity" and not re.search(r"[A-Z]", changed_old + changed_new):
                errors.append(f"{tid}/entity: no capitalized name changed")
            if t == "why_how" and "?" not in changed_new:
                errors.append(f"{tid}/why_how: the new text must carry the changed question")
            turn2[t] = vocative("Ryan", new)
        topics.append({"slug": tid, "topic": row["title"], "turn1": TURN1.format(title=row["title"].rstrip(".")),
                       "turn2": turn2, "meta": meta})
    if errors:
        print("CHECKS FAILED\n" + "\n".join(errors), file=sys.stderr)
        if not args.force:
            sys.exit(1)
    BUILD.mkdir(exist_ok=True)
    (BUILD / "pairs_public.yaml").write_text(yaml.safe_dump({"topics": topics}, sort_keys=False, allow_unicode=True,
                                                            width=120))
    summary = {t: {"n": len(v), "len_ratio_min_med_max": [round(min(x[0] for x in v), 2),
                                                          round(sorted(x[0] for x in v)[len(v) // 2], 2),
                                                          round(max(x[0] for x in v), 2)],
                   "edit_ratio_med_max": [round(sorted(x[1] for x in v)[len(v) // 2], 2), round(max(x[1] for x in v), 2)]}
               for t, v in stats.items() if v}
    summary["na"] = dict(Counter(t for tp in topics for t in tp["meta"]["na"]))
    summary["kinds"] = {t: dict(Counter(tp["meta"]["kinds"].get(t) for tp in topics if t in tp["meta"]["kinds"]))
                        for t in KINDS}
    (HERE / "edit_stats.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary, indent=1))
    print(f"{'(with errors) ' if errors else ''}wrote {BUILD / 'pairs_public.yaml'} ({len(topics)} topics)")


# ----------------------------------------------------------------------------------------------- anchors
# Human anchors for the quality ruler: one real IQ2 panel exchange per test motion, in the show's beat format.
ANCHOR_TURNS, ANCHOR_MIN_W, ANCHOR_MAX_W = 4, 25, 120
ANCHOR_V2 = False      # set by `anchors --v2`: pick the most interactive window instead of the first one
ANCHOR_V2_MAX_W = 100  # v2: prefer shorter, conversational turns (fallback: most interactive window of any length)
_STOP = set("the a an and or but so to of in on at for with by from as is are was were be it this that i you we they "
            "he she not no do does did have has had will would can could should just very really what which who".split())


def interactivity(texts: list[str], speakers: set[str]) -> float:
    """How much each turn engages the previous one: shared content words / figures with the previous turn, a question
    in the previous turn (that this turn answers), and second-person address ("you", "your") or naming the opponent."""
    def cw(t: str) -> set[str]:
        return {w for w in re.findall(r"[a-z0-9']+", t.lower()) if w not in _STOP and len(w) > 3}

    sc = 0.0
    for prev, cur in zip(texts, texts[1:]):
        sc += min(len(cw(prev) & cw(cur)), 6) / 6
        sc += 0.5 if "?" in prev else 0.0
        sc += 0.5 if re.search(r"\byou(r)?\b", cur, re.IGNORECASE) else 0.0
        sc += 0.5 if any(p in cur for sid in speakers for p in sid.split() if len(p) > 3) else 0.0
    return sc / max(len(texts) - 1, 1)


def _sentence_trim(nlp, text: str, max_w: int) -> tuple[int, int]:
    """(start, end) char span: drop a leading/trailing interrupted fragment, keep whole sentences up to max_w words."""
    sents = [s for s in nlp(text).sents]
    keep, w = [], 0
    for s in sents:
        t = s.text.strip()
        if not keep and (t.startswith("--") or t.startswith("—")):
            continue
        if not re.search(r"[.?!][\"']?$", t) or "--" in t:
            if keep:
                break
            continue
        n = len(t.split())
        if keep and w + n > max_w:
            break
        keep.append(s)
        w += n
    return (keep[0].start_char, keep[-1].end_char) if keep else (0, 0)


def cmd_anchors(args) -> None:
    import spacy

    global ANCHOR_V2
    ANCHOR_V2 = bool(getattr(args, "v2", False))

    nlp = spacy.load("en_core_web_sm")
    corpus = load_corpus()
    sel = [json.loads(l) for l in (HERE / "selection.jsonl").read_text().splitlines()]
    out, built, misses = [], [], []
    for r in sel:
        cv = corpus.get_conversation(r["conversation_id"])
        utts = [u for u in cv.iter_utterances() if int(u.meta.get("segment", -1)) == 1]
        all_names = {u.speaker.id for u in cv.iter_utterances()} - {"audience", "unknown"}
        kept = []  # (index, utterance, side, start, end, text)
        for i, u in enumerate(utts):
            st = u.meta.get("speakertype")
            if st not in ("for", "against"):
                continue
            text = normalize(u.text)
            a, b = _sentence_trim(nlp, text, ANCHOR_MAX_W)
            if len(text[a:b].split()) >= ANCHOR_MIN_W:
                kept.append((i, u, st, a, b, text[a:b]))
        found = None
        best = (False, -1.0)
        for j in range(len(kept) - ANCHOR_TURNS + 1):
            win = kept[j: j + ANCHOR_TURNS]
            sides = [w[2] for w in win]
            if any(sides[x] == sides[x + 1] for x in range(len(sides) - 1)):  # sides must alternate
                continue
            mod = next((utts[x] for x in range(win[0][0] - 1, -1, -1) if utts[x].meta.get("speakertype") == "mod"), None)
            if not ANCHOR_V2:
                found = (win, mod)
                break
            sc = interactivity([w[5] for w in win], {w[1].speaker.id for w in win})
            short = all(len(w[5].split()) <= ANCHOR_V2_MAX_W for w in win)
            key = (short, sc)  # short windows first, then interactivity
            if found is None or key > best:
                best, found = key, (win, mod)
        if not found:
            misses.append(r["topic_id"])
            continue
        win, mod = found
        # Sides map to panelists (each side's debaters speak as one panelist): first side → Ryan, the other → Amy.
        side_role = {win[0][2]: "commentator", win[1][2]: "guest"}
        names = {"commentator": "Ryan", "guest": "Amy", "host": "Lessac"}
        debaters = {u.speaker.id: u.meta.get("speakertype") for u in cv.iter_utterances()
                    if u.meta.get("speakertype") in ("for", "against")}
        rec = {"topic_id": r["topic_id"], "conversation_id": r["conversation_id"], "host": None,
               "turns": [{"utterance_id": w[1].id, "role": side_role[w[2]], "side": w[2], "start": w[3], "end": w[4]}
                         for w in win]}

        def rename(t: str, own_side: str | None) -> str:
            """Debater names → the side's panelist name (a teammate of the speaker → "my colleague"); moderator → Lessac."""
            for sid, side in debaters.items():
                new = "my colleague" if (own_side and side == own_side) else names[side_role.get(side, "guest")]
                t = re.sub(rf"\b{re.escape(sid)}\b", new, t)
                for part in sid.split():
                    if len(part) > 3:
                        t = re.sub(name_pattern(part), new, t)
            if mod is not None:
                t = re.sub(rf"\b{re.escape(mod.speaker.id)}\b", "Lessac", t)
                for part in mod.speaker.id.split():
                    if len(part) > 3:
                        t = re.sub(name_pattern(part), "Lessac", t)
            return t

        lines = []
        if mod is not None:
            mt = normalize(mod.text)
            ms = [x for x in nlp(mt).sents][-2:]
            if ms:
                rec["host"] = {"utterance_id": mod.id, "start": ms[0].start_char, "end": ms[-1].end_char}
                lines.append({"role_id": "host", "speaker": "Lessac",
                              "text": rename(mt[ms[0].start_char: ms[-1].end_char], None)})
        for w in win:
            role = side_role[w[2]]
            lines.append({"role_id": role, "speaker": names[role], "text": rename(w[5], w[2]), "side": w[2]})
        if ANCHOR_V2:
            rec["v2"] = {"short": bool(best[0]), "interactivity": round(float(best[1]), 3)}
        out.append(rec)
        built.append({"topic_id": r["topic_id"], "topic": r["title"], "lines": lines, **({"v2": rec["v2"]} if ANCHOR_V2 else {})})
    suffix = "_v2" if ANCHOR_V2 else ""
    (HERE / f"anchors{suffix}.jsonl").write_text("".join(json.dumps(x) + "\n" for x in out))  # IDs + offsets only
    BUILD.mkdir(exist_ok=True)
    (BUILD / f"anchors{suffix}.jsonl").write_text("".join(json.dumps(x, ensure_ascii=False) + "\n" for x in built))
    print(f"anchors for {len(out)}/{len(sel)} topics; no window: {misses}")


# ----------------------------------------------------------------------------------------------- main

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0] + " Stages: select, review, compile, "
                                 "apply, anchors (eval/data/public_set/README.md).")
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("select")
    p.add_argument("--legacy", action="store_true")
    p.set_defaults(fn=cmd_select)
    sub.add_parser("review").set_defaults(fn=cmd_review)
    sub.add_parser("compile").set_defaults(fn=cmd_compile)
    p = sub.add_parser("anchors")
    p.add_argument("--v2", action="store_true", help="most interactive window, turns ≤ 80 words (the final ruler's anchors)")
    p.set_defaults(fn=cmd_anchors)
    p = sub.add_parser("apply")
    p.add_argument("--force", action="store_true", help="write pairs_public.yaml even if checks fail (inspection)")
    p.add_argument("--topics", default="", help="comma list: check only these topics")
    p.set_defaults(fn=cmd_apply)
    args = ap.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
