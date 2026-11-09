"""Prompt log-probability scoring with vLLM (helpers for eval/responsiveness.py logprob).

score() gives log P(line | system, user prompt) under an open scorer model, summed over the line's tokens, read from
vLLM prompt logprobs (no sampling). load_persona() reads a panelist's name and personality card.
"""

from __future__ import annotations

from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
PERSONA_DIR = ROOT / "config" / "personas"
HUMAN_LABEL = "Human guest"  # agent/show/show_history.py


def load_persona(role: str) -> tuple[str, str]:
    data = yaml.safe_load((PERSONA_DIR / f"{role}.yaml").read_text())
    return str(data.get("name") or role), str(data.get("personality") or "").strip()


def score(requests: list[tuple], model: str, max_model_len: int) -> list[tuple[float, int]]:
    """requests: (key, condition, system, user, line). Returns (summed logprob of line, number of line tokens)."""
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams
    from vllm.inputs import TokensPrompt

    tok = AutoTokenizer.from_pretrained(model)
    prompts, spans = [], []
    for _, _, system, user, line in requests:
        msgs = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        prefix = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        p_ids = tok(prefix, add_special_tokens=False)["input_ids"]
        y_ids = tok(line, add_special_tokens=False)["input_ids"]
        prompts.append(TokensPrompt(prompt_token_ids=p_ids + y_ids))
        spans.append((len(p_ids), len(y_ids)))

    llm = LLM(model=model, max_model_len=max_model_len, gpu_memory_utilization=0.85)
    params = SamplingParams(max_tokens=1, prompt_logprobs=1)
    outs = llm.generate(prompts, params)

    results = []
    for out, (start, n) in zip(outs, spans):
        ids = out.prompt_token_ids
        lps = out.prompt_logprobs
        total = sum(lps[i][ids[i]].logprob for i in range(start, start + n))
        results.append((total, n))
    return results
