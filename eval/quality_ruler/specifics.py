"""Unsupported specifics (CPU, no judge): how many of a panel line's specifics have no source.

A specific is a number (digits or spelled out, e.g. "six hundred"), a percentage, an amount of money, a year, or a
capitalised multi-word name ("Fair Labor Standards Act", "New York"). A specific is SUPPORTED when it appears in the
topic's fact pack (the facts and their source sentences) or earlier in the episode, said by someone else (the human, the
host, the other panelist); otherwise it is unsupported, i.e. the panelist brought it in from memory or made it up.

Every arm is checked against the same pack, also arms whose prompts never saw it, so the comparison is fair.

Numbers match by value ("40%" = "forty percent" = "40 percent"; "$1.5 million" = "1,500,000"). Names match when the
name appears in the support text, or all of its words do. Spelled-out numbers below 11 without a scale word or a
percent ("two sides", "one reason") are not counted: in speech they are rarely factual claims.

    python -m eval.quality_ruler.specifics A.jsonl B.jsonl [--pack factpack.json] [--out specifics.json]

`--pack` defaults to TALKSHOW_FACT_PACK. Per episode file: specifics per 100 words, share supported, share of lines with
at least one unsupported specific, and some unsupported examples to read.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

PANEL = ("commentator", "guest")
UNITS = {w: i for i, w in enumerate("zero one two three four five six seven eight nine ten eleven twelve thirteen "
                                     "fourteen fifteen sixteen seventeen eighteen nineteen".split())}
TENS = {w: 10 * (i + 2) for i, w in enumerate("twenty thirty forty fifty sixty seventy eighty ninety".split())}
SCALES = {"hundred": 100, "thousand": 1_000, "million": 1_000_000, "billion": 1_000_000_000,
          "trillion": 1_000_000_000_000}
NUMBER_WORDS = set(UNITS) | set(TENS) | set(SCALES)
SCALE_ABBR = {"k": 1_000, "m": 1_000_000, "bn": 1_000_000_000, "b": 1_000_000_000, "tn": 1_000_000_000_000,
              "trillion": 1_000_000_000_000, "billion": 1_000_000_000, "million": 1_000_000, "thousand": 1_000,
              "hundred": 100}
MONEY_WORDS = ("dollar", "dollars", "euro", "euros", "pound", "pounds", "cent", "cents")
PERCENT_WORDS = ("percent", "per cent", "percentage points", "percentage point")

# Tokens of a number or a name, keeping punctuation as separate tokens so names break at commas and full stops.
TOKEN = re.compile(r"\$|£|€|\d+(?:,\d{3})*(?:\.\d+)?(?:st|nd|rd|th|s)?%?|[A-Za-z][A-Za-z'’\-.]*[A-Za-z]|[A-Za-z]|[^\sA-Za-z\d]")
CONNECT = {"of", "the", "for", "and", "de", "du", "von", "van", "der", "la", "le", "on", "in", "&"}
# Capitalised words that are not names at the start of a run (sentence starters, pronouns, the show's own speakers).
NOT_NAME = set("""i i'm i've i'd i'll a an the and but or so if when while because yes no not look well now then that this
these those there here what why how who which where it it's its we we're our you you're your they their he she his her
my me us them let let's sure okay ok honestly frankly actually still even just maybe perhaps also only both all some
many most more much every each any one to of in on at for with by from as is are was were be do does did can could will
would should may might must have has had think say said take consider imagine remember right exactly true
indeed absolutely sorry thanks thank please again first second finally meanwhile however instead plus yet after before
since ryan amy lessac human guest host""".split())


# ------------------------------------------------------------------------------------------------- extraction

def _num(tok: str) -> float | None:
    m = re.fullmatch(r"(\d+(?:,\d{3})*(?:\.\d+)?)(?:st|nd|rd|th|s)?%?", tok)
    if not m:
        return None
    try:
        return float(m.group(1).replace(",", ""))
    except ValueError:
        return None


def _key(v: float) -> str:
    return f"{v:.6g}"


def _words_value(ws: list[str]) -> float:
    """Value of a run of number words: "six hundred" 600, "two hundred and fifty thousand" 250000."""
    total, cur = 0, 0
    for w in ws:
        if w in UNITS:
            cur += UNITS[w]
        elif w in TENS:
            cur += TENS[w]
        elif w == "hundred":
            cur = max(cur, 1) * 100
        elif w in SCALES:
            total += max(cur, 1) * SCALES[w]
            cur = 0
    return float(total + cur)


def extract(text: str) -> list[dict]:
    """Specifics in text: [{"kind": number|percent|money|year|name, "text": surface, "key": match key}]."""
    toks = []
    for tok in TOKEN.findall(text or ""):
        parts = tok.lower().split("-")
        # "twenty-five" -> "twenty", "five"
        toks += tok.split("-") if len(parts) > 1 and all(p in NUMBER_WORDS for p in parts) else [tok]
    low = [t.lower() for t in toks]
    out: list[dict] = []
    i, n = 0, len(toks)
    while i < n:
        t, lt = toks[i], low[i]
        start = i
        money = False
        if t in ("$", "£", "€") and i + 1 < n and _num(toks[i + 1]) is not None:
            money, i = True, i + 1
            t, lt = toks[i], low[i]
        v = _num(t)
        spelled = False
        if v is None and lt in NUMBER_WORDS:
            j = i
            ws = []
            while j < n and (low[j] in NUMBER_WORDS or (low[j] in ("and", "-") and ws and j + 1 < n
                                                         and low[j + 1] in NUMBER_WORDS)):
                if low[j] in NUMBER_WORDS:
                    ws.append(low[j])
                j += 1
            v, spelled, i = _words_value(ws), True, j - 1
            has_scale = any(w in SCALES for w in ws)
        if v is None:
            i += 1
            continue
        # a scale word or abbreviation right after the figure ("1.5 million", "17bn")
        j = i + 1
        if not spelled and j < n and low[j] in SCALE_ABBR:
            v *= SCALE_ABBR[low[j]]
            j += 1
            has_scale = True
        elif not spelled:
            has_scale = False
        rest = " ".join(low[j:j + 3])
        percent = t.endswith("%") or any(rest.startswith(p) for p in PERCENT_WORDS)
        money = money or any(rest.startswith(w) for w in MONEY_WORDS)
        extra = 1 if (percent and not t.endswith("%")) or (money and toks[start] not in ("$", "£", "€")) else 0
        surface = " ".join(toks[start:j + extra])
        if spelled and v < 11 and not has_scale and not percent:
            i = j
            continue
        raw = toks[i]
        if percent:
            kind = "percent"
        elif money:
            kind = "money"
        elif not spelled and re.fullmatch(r"\d{4}s?", raw) and 1000 <= v <= 2100:
            kind = "year"
        else:
            kind = "number"
        out.append({"kind": kind, "text": surface, "key": "n:" + _key(v)})
        i = j
    out += _names(toks)
    return out


def _is_cap(tok: str) -> bool:
    return bool(re.fullmatch(r"[A-Z][A-Za-z'’\-.]*", tok)) and tok.lower() not in NOT_NAME


def _names(toks: list[str]) -> list[dict]:
    out, i, n = [], 0, len(toks)
    while i < n:
        if not _is_cap(toks[i]):
            i += 1
            continue
        run = [toks[i]]
        j = i + 1
        while j < n:
            if _is_cap(toks[j]):
                run.append(toks[j])
                j += 1
            elif toks[j].lower() in CONNECT and j + 1 < n and _is_cap(toks[j + 1]):
                run += [toks[j], toks[j + 1]]
                j += 2
            else:
                break
        caps = [w for w in run if w.lower() not in CONNECT]
        if len(caps) >= 2:
            name = " ".join(run)
            out.append({"kind": "name", "text": name, "key": "w:" + _norm(name)})
        i = j
    return out


def _norm(text: str) -> str:
    return " ".join(re.findall(r"[a-z0-9]+", text.lower().replace("’", "'").replace("'s", "")))


# ------------------------------------------------------------------------------------------------- support

class Support:
    """Everything a specific may be taken from: number values, normalised text and its word set."""

    def __init__(self, texts: list[str]) -> None:
        self.values: set[str] = set()
        for t in texts:
            self.values |= {s["key"] for s in extract(t) if s["kind"] != "name"}
        self.text = " " + " ".join(_norm(t) for t in texts) + " "
        self.words = set(self.text.split())

    def extended(self, texts: list[str]) -> "Support":
        new = Support(texts)
        new.values |= self.values
        new.text += self.text
        new.words |= self.words
        return new

    def supports(self, spec: dict) -> bool:
        if spec["kind"] != "name":
            return spec["key"] in self.values
        name = spec["key"][2:]
        if f" {name} " in self.text:
            return True
        ws = [w for w in name.split() if w not in CONNECT]
        return bool(ws) and all(w in self.words for w in ws)


def pack_texts(entry: dict | None) -> list[str]:
    if not entry:
        return []
    out = [entry.get("topic", "")]
    for f in entry.get("facts", []):
        out += [f.get("fact", ""), f.get("source_sentence", "")]
    return [t for t in out if t]


def pack_entry(ep: dict, pack: dict) -> dict | None:
    """The episode's pack entry: by topic id, else by topic text (agent/panel/fact_pack.py matching)."""
    if ep.get("topic_id") in pack:
        return pack[ep["topic_id"]]
    want = _norm(ep.get("topic", ""))
    for v in pack.values():
        if want and _norm(v.get("topic", "")) == want:
            return v
    return None


def words(text: str) -> int:
    return len(re.findall(r"[A-Za-z0-9$%']+", text or ""))


def episode_lines(ep: dict, pack: dict) -> list[dict]:
    """Per scored panel line: its specifics with a supported flag."""
    entry = pack_entry(ep, pack)
    base = Support(pack_texts(entry) + [ep.get("topic", "")])
    lines = ep["lines"]
    out = []
    for i in ep.get("scored", [k for k, l in enumerate(lines) if l["role"] in PANEL]):
        line = lines[i]
        if line["role"] not in PANEL:
            continue
        earlier = [l["text"] for l in lines[:i] if l["role"] != line["role"]]
        sup = base.extended(earlier)
        specs = [{**s, "supported": sup.supports(s)} for s in extract(line["text"])]
        out.append({"episode": ep["id"], "index": i, "role": line["role"], "text": line["text"],
                    "words": words(line["text"]), "specifics": specs, "pack_found": entry is not None})
    return out


def summarise(rows: list[dict], n_examples: int = 15) -> dict:
    n_w = sum(r["words"] for r in rows)
    specs = [s for r in rows for s in r["specifics"]]
    n_sup = sum(s["supported"] for s in specs)
    bad = [r for r in rows if any(not s["supported"] for s in r["specifics"])]
    by_kind: dict[str, list[int]] = {}
    for s in specs:
        by_kind.setdefault(s["kind"], [0, 0])
        by_kind[s["kind"]][0] += 1
        by_kind[s["kind"]][1] += int(s["supported"])
    return {
        "lines": len(rows),
        "words": n_w,
        "specifics": len(specs),
        "specifics_per100w": 100 * len(specs) / n_w if n_w else float("nan"),
        "share_supported": n_sup / len(specs) if specs else float("nan"),
        "unsupported_per100w": 100 * (len(specs) - n_sup) / n_w if n_w else float("nan"),
        "share_lines_unsupported": len(bad) / len(rows) if rows else float("nan"),
        "lines_without_pack": sum(not r["pack_found"] for r in rows),
        "by_kind": {k: {"n": v[0], "supported": v[1]} for k, v in sorted(by_kind.items())},
        "examples_unsupported": [
            {"episode": r["episode"], "index": r["index"],
             "unsupported": [s["text"] for s in r["specifics"] if not s["supported"]], "text": r["text"]}
            for r in bad[:n_examples]],
    }


SET_KEYS = ("specifics_per100w", "share_supported", "unsupported_per100w", "share_lines_unsupported")


def score_episodes(eps: list[dict], pack: dict) -> dict:
    return summarise([row for e in eps for row in episode_lines(e, pack)])


def load_pack(path: str) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("episodes", nargs="+", help="episode jsonl files, one per arm")
    ap.add_argument("--pack", default=os.environ.get("TALKSHOW_FACT_PACK", ""),
                    help="fact pack json (default TALKSHOW_FACT_PACK)")
    ap.add_argument("--out", default="", help="json report (default: print only)")
    args = ap.parse_args()
    if not args.pack:
        sys.exit("give --pack or set TALKSHOW_FACT_PACK")
    pack = load_pack(args.pack)
    res = {}
    for f in args.episodes:
        eps = [json.loads(l) for l in Path(f).read_text().splitlines() if l.strip()]
        name = Path(f).stem if Path(f).stem not in res else f"{Path(f).parent.name}/{Path(f).stem}"
        res[name] = score_episodes(eps, pack)
    keys = ("lines", "specifics", "specifics_per100w", "share_supported", "unsupported_per100w",
            "share_lines_unsupported", "lines_without_pack")
    print("| measure | " + " | ".join(res) + " |\n|---|" + "---|" * len(res))
    for k in keys:
        print(f"| {k} | " + " | ".join(f"{v[k]:.3f}" if isinstance(v[k], float) else str(v[k]) for v in res.values())
              + " |")
    if args.out:
        Path(args.out).write_text(json.dumps(res, indent=1, ensure_ascii=False))
        print(f"→ {args.out}")


if __name__ == "__main__":
    main()
