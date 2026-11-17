import json
import tempfile
from pathlib import Path

from eval.bench import human_ref, show


def test_build_human_reference():
    anchors = [{"topic_id": "iq2-1", "topic": "Ban X.", "lines": [
        {"role_id": "host", "speaker": "Lessac", "text": "Why?"},
        {"role_id": "commentator", "speaker": "Ryan", "text": "Because."}]}]
    with tempfile.TemporaryDirectory() as d:
        scripts, eps = human_ref.build(anchors, Path(d))
        assert scripts["topics"][0]["audience"] == [human_ref.opening("Ban X.")]
        assert len(show.load_scripts(Path(d) / "scripts.yaml")) == 1  # one audience turn is a valid script
        ep = json.loads((Path(d) / "human_runs" / "episodes.jsonl").read_text())
        assert [l["role"] for l in ep["lines"]] == ["host", "human", "host", "commentator"]
        assert ep["lines"][1]["text"] == human_ref.opening("Ban X.")
