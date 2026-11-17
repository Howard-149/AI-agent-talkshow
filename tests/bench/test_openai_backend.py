import json

from eval.bench import systems


class _Resp:
    def __init__(self, payload):
        self.payload = payload

    def read(self):
        return json.dumps(self.payload).encode()

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def test_openai_backend_parallel_and_usage(monkeypatch, tmp_path):
    monkeypatch.setenv("OPENAI_API_KEY", "test")
    seen = []

    def fake_urlopen(req, timeout=0):
        body = json.loads(req.data)
        seen.append(body)
        text = body["messages"][-1]["content"].upper()
        return _Resp({"choices": [{"message": {"content": text}}],
                      "usage": {"prompt_tokens": 10, "completion_tokens": 3, "prompt_tokens_details": {"cached_tokens": 4}}})

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    log = tmp_path / "usage.jsonl"
    b = systems.OpenAIBackend("gpt-test", workers=4, usage_path=str(log))
    reqs = [systems.Request("think", [{"role": "user", "content": f"line {i}"}], 0.2, 100) for i in range(5)]
    out = b.generate(reqs)
    assert out == [f"LINE {i}" for i in range(5)]  # order kept
    assert all(s["model"] == "gpt-test" and s["reasoning_effort"] == "none" for s in seen)
    assert b.usage == {"calls": 5, "prompt_tokens": 50, "cached_tokens": 20, "completion_tokens": 15}
    assert len(log.read_text().splitlines()) == 5
