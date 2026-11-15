import json
import tempfile
from pathlib import Path

from eval.bench import t6_dims


def _run(d: Path, label: str, text: str) -> None:
    d.mkdir(parents=True)
    (d / "run.json").write_text(json.dumps({"label": label}))
    eps = []
    for k in range(2):
        eps.append({"id": f"{label}:t:k{k}", "topic_id": "t", "topic": "Topic", "k": k, "system": label,
                    "lines": [{"role": "human", "speaker": "Guest", "text": "Is it fair?"},
                              {"role": "commentator", "speaker": "Ryan", "text": text}]})
    (d / "episodes.jsonl").write_text("".join(json.dumps(e) + "\n" for e in eps))


def test_build_and_score_roundtrip():
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        _run(tmp / "a", "sysA", "Short.")
        _run(tmp / "b", "sysB", "A much longer line with many more words in it.")
        out = tmp / "out"
        t6_dims.main(["build", "--runs", f"{tmp / 'a'},{tmp / 'b'}", "--out", str(out), "--ref-pairs", "2"])
        reqs = [json.loads(l) for l in (out / "judge" / "pairwise.jsonl").read_text().splitlines()]
        assert len(reqs) == 2 * 2 * len(t6_dims.DIMS)  # 2 pairs, both orders, every dimension
        assert all(t6_dims.LENGTH_RULE in r["messages"][1]["content"] for r in reqs)
        # open judge always prefers A (position bias): order consistency 0, preferences 0.5
        (out / "judge" / "answers_fake.jsonl").write_text("".join(
            json.dumps({"id": r["id"], "p_a": 0.9}) + "\n" for r in reqs))
        t6_dims.main(["score", "--out", str(out)])
        md = (out / "results.md").read_text()
        assert "| overall | 0.00 | 0.50 | 0.50 |" in md


def test_parse_ref():
    ok = '```json\n{"audience":"A","interplay":"B","novelty":"A","persona":"A","entertainment":"B",' \
         '"naturalness":"A","overall":"B","why":"x"}\n```'
    assert t6_dims.parse_ref(ok)["overall"] == "B"
    assert t6_dims.parse_ref('{"audience":"A"}') is None
    assert t6_dims.parse_ref("no json") is None
