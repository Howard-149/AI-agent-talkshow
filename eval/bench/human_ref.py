"""Human reference for T6: real IQ2 panel exchanges against a system on the same motions and opening question.

    python -m eval.data.public_set.build anchors --v2          # real exchanges, rebuilt locally (needs spaCy, ConvoKit)
    python -m eval.bench.human_ref --out DIR                   # DIR/scripts.yaml (one audience turn per motion) and
                                                               # DIR/human_runs (the exchanges as a T6 run)
    python -m eval.bench.show generate --system think_action --arm demo --model ... \\
        --scripts DIR/scripts.yaml --slots 5 --reruns 1 --out DIR/system
    python -m eval.bench.t6_dims build --runs DIR/human_runs,DIR/system --out DIR/dims --open-header

The real debate text is copyrighted: it is rebuilt from ConvoKit into git-ignored paths and never committed.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

ANCHORS = Path(__file__).resolve().parents[1] / "data" / "public_set" / "build" / "anchors_v2.jsonl"


def opening(topic: str) -> str:
    """The audience's opening question, the same for the real exchange and the system."""
    return f'I want to put a motion to the panel: "{topic.rstrip(".")}." Where do the two of you land?'


def build(anchors: list[dict], out: Path) -> tuple[dict, list[dict]]:
    scripts = {"topics": [{"slug": a["topic_id"], "topic": a["topic"], "audience": [opening(a["topic"])]}
                          for a in anchors]}
    eps = []
    for a in anchors:
        lines = [{"role": "host", "speaker": "host", "text": "Welcome to the show.", "procedural": True},
                 {"role": "human", "speaker": "Human guest", "text": opening(a["topic"])}]
        lines += [{"role": l["role_id"], "speaker": l["speaker"], "text": l["text"]} for l in a["lines"]]
        eps.append({"id": f"human-iq2:{a['topic_id']}:k0", "topic_id": a["topic_id"], "topic": a["topic"], "k": 0,
                    "system": "human-iq2", "audience": [opening(a["topic"])], "lines": lines})
    out.mkdir(parents=True, exist_ok=True)
    (out / "scripts.yaml").write_text(yaml.safe_dump(scripts, sort_keys=False, allow_unicode=True))
    runs = out / "human_runs"
    runs.mkdir(exist_ok=True)
    (runs / "run.json").write_text(json.dumps({"label": "human-iq2"}))
    (runs / "episodes.jsonl").write_text("".join(json.dumps(e) + "\n" for e in eps))
    return scripts, eps


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--anchors", default=str(ANCHORS))
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)
    anchors = [json.loads(l) for l in Path(args.anchors).read_text().splitlines() if l.strip()]
    scripts, eps = build(anchors, Path(args.out))
    print(f"{len(scripts['topics'])} motions, {len(eps)} human exchanges -> {args.out}")


if __name__ == "__main__":
    main()
