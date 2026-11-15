"""Free-floor speaker selection with a decision model (TALKSHOW_FLOOR_JEV=1, default OFF).

Jev (TypeSafe, https://docs.typesafe.ai) answers a typed choice question with a probability per option and a
confidence, and generates no text, so a pick takes about 0.1-0.2 s instead of a full LLM controller call. It is used
as the first step of a cascade: when the speak candidates are known, Jev picks one of them or "host" (nobody
suitable). A candidate picked with confidence >= TALKSHOW_FLOOR_JEV_MIN_CONF (default 0.7) gets the floor. A "host"
pick, a low confidence, a missing key (TYPESAFE_API_KEY) or any API error falls through to the LLM controller,
which also writes the host instruction. Addressee priority runs before this, as before.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
import urllib.request
from typing import Any

logger = logging.getLogger(__name__)

URL = "https://api.typesafe.ai/v1/systemone"
MODEL = "jev-1.14.0"
TIMEOUT_S = 3.0
INSTRUCTIONS = (
    "The current speech on this talk show has ended and the floor is free. Who should speak next? "
    "Pick the candidate whose reply best advances the discussion: an answer to a question or invitation already "
    "spoken in the transcript comes first, then relevance, new substance and a fair share of the floor. "
    "Pick host only if every candidate is off topic, vacuous or repeating earlier lines."
)


def jev_enabled() -> bool:
    return os.environ.get("TALKSHOW_FLOOR_JEV", "0").strip().lower() in ("1", "true", "yes", "on")


def min_confidence() -> float:
    try:
        return float(os.environ.get("TALKSHOW_FLOOR_JEV_MIN_CONF", "0.7"))
    except ValueError:
        return 0.7


def build_request(transcript: str, candidates: list[dict], counts: dict[str, int], names: dict[str, str],
                  host_role: str) -> dict[str, Any]:
    """The Jev request: the state (transcript, each candidate's draft and floor share) and one choice question."""
    lines = [f"Transcript (latest last):\n{transcript}", "", "Candidates who want the floor now:"]
    for c in candidates:
        lines.append(f'- {c["speaker"]} ({c["role"]}), {counts.get(c["role"], 0)} turns so far, would say: '
                     f'"{c.get("content") or ""}"')
    criteria = {c["role"]: f'{c["speaker"]} speaks next' for c in candidates}
    criteria[host_role] = f"nobody suitable: the host {names.get(host_role, host_role)} steps in"
    return {"model": MODEL, "state": "\n".join(lines),
            "questions": {"next": {"type": "choice", "instructions": INSTRUCTIONS, "criteria": criteria}}}


def _post(body: dict[str, Any], key: str) -> dict[str, Any]:
    req = urllib.request.Request(URL, data=json.dumps(body).encode(), method="POST",
                                 headers={"Content-Type": "application/json", "Authorization": f"Bearer {key}"})
    with urllib.request.urlopen(req, timeout=TIMEOUT_S) as r:  # noqa: S310 (fixed https URL)
        return json.loads(r.read())


async def jev_pick(transcript: str, candidates: list[dict], counts: dict[str, int], names: dict[str, str],
                   host_role: str) -> tuple[str | None, float, dict[str, Any]]:
    """(role, confidence, log fields). role is a candidate role, ``host_role``, or None on any failure."""
    key = os.environ.get("TYPESAFE_API_KEY", "").strip()
    if not key or not candidates:
        return None, 0.0, {"jev": "skipped"}
    body = build_request(transcript, candidates, counts, names, host_role)
    t = time.monotonic()
    try:
        resp = await asyncio.to_thread(_post, body, key)
        ans = resp["answers"]["next"]
        role, conf = ans["choice"], float(ans.get("confidence", 0.0))
    except Exception as e:  # noqa: BLE001  (any failure falls back to the LLM controller)
        logger.warning("jev floor pick failed: %s", type(e).__name__)
        return None, 0.0, {"jev": "error", "jev_error": type(e).__name__}
    info = {"jev": "ok", "jev_choice": role, "jev_conf": round(conf, 3),
            "jev_latency_s": round(time.monotonic() - t, 3), "jev_probs": ans.get("probabilities")}
    if role not in {c["role"] for c in candidates} | {host_role}:
        return None, 0.0, {**info, "jev": "invalid"}
    return role, conf, info
