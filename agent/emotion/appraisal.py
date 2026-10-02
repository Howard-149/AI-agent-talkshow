"""Separate PAD appraisal step (Sentipolis fast update).

After every spoken line, each AI listener appraises it in its own small LLM
call and emits ``[pad]: <feeling> ΔP ΔA ΔD``. Dialogue prompts no longer carry
[pad]; they only see a plain-language mood descriptor.

Split for a future LangGraph node:
- pure: ``appraisal_listeners``, ``build_appraisal_prompt``, ``parse_pad_appraisal``
- effectful: ``run_appraisal`` (LLM), ``apply_appraisal`` (PAD write + log)
- scheduling glue (today's hook from ``show_history.append_*``):
  ``schedule_appraisals`` / ``await_pending_appraisal``
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Sequence

from agent.emotion.state import (
    affect_descriptor,
    emotion_source,
    get_role_emotion,
    parse_pad_appraisal_full,
)

if TYPE_CHECKING:
    from agent.data import TalkShowData

logger = logging.getLogger(__name__)

DEFAULT_APPRAISAL_TEMPERATURE = 0.9
DEFAULT_APPRAISAL_MAX_TOKENS = 48
DEFAULT_APPRAISAL_WAIT_S = 3.0
_CONTEXT_LINES = 6


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name, "").strip()
    if raw:
        try:
            return float(raw)
        except ValueError:
            pass
    return default


def appraisal_enabled() -> bool:
    return emotion_source() != "llm"


@dataclass(frozen=True)
class Utterance:
    speaker: str  # role id or "human"
    text: str
    seq: int = 0
    # Lines spoken before this one, snapshotted when it was scheduled:
    # (role_id, speaker_label, text, procedural). Empty → read live history.
    context: tuple[tuple[str, str, str, bool], ...] = ()


APPRAISAL_PROTOCOLS = ("reply", "exchange", "line")


def appraisal_protocol() -> str:
    """How a listener appraises a line that replies to them.

    - ``reply`` (default): appraise only the reply, flagged as a reply to them; their
      own line stays in context. Mood changes only from what *others* say — the
      speaker's line was generated from their pre-speech mood, so reading it back
      would feed the mood into itself.
    - ``exchange``: appraise own line + reply as one pair (Sentipolis-round-like).
    - ``line``: no reply awareness; every line is appraised on its own.
    """
    raw = (os.environ.get("TALKSHOW_PAD_APPRAISAL_PROTOCOL") or "reply").strip().lower()
    return raw if raw in APPRAISAL_PROTOCOLS else "reply"


def replied_line(
    role: str, utt: Utterance, context: Sequence[tuple]
) -> tuple | None:
    """``role``'s own line that ``utt`` answers: the last substantive line before it.

    Canned procedural host lines ("Amy, the floor is yours.") sit between panelists
    in the transcript; they are skipped so a panelist answering a panelist is not
    mistaken for a reply to the host.
    """
    if utt.speaker == role:
        return None
    for c in reversed(context):
        if len(c) > 3 and c[3]:
            continue
        return c if c[0] == role else None
    return None


@dataclass(frozen=True)
class AppraisalResult:
    role: str
    word: str | None
    delta: tuple[float, float, float] | None
    raw: str
    latency_s: float
    error: str | None = None
    valence: str | None = None
    flipped: bool = False


# --- pure -----------------------------------------------------------------


def ai_roles(data: TalkShowData) -> list[str]:
    from agent.panel.panel_context import panel_speaker_roles

    listen = data.scenario.turn_control.listen_role
    return list(dict.fromkeys([listen, *panel_speaker_roles(data.scenario)]))


def appraisal_listeners(data: TalkShowData, speaker: str) -> list[str]:
    """AI roles that hear ``speaker``. Speaker self-appraisal is opt-in."""
    include_self = os.environ.get("TALKSHOW_PAD_APPRAISAL_SELF", "").lower() in (
        "1",
        "true",
        "yes",
    )
    return [r for r in ai_roles(data) if include_self or r != speaker]


def _speaker_label(role: str) -> str:
    from agent.show.show_history import HUMAN_LABEL

    if role == "human":
        return HUMAN_LABEL
    from agent.config import load_persona_name
    from agent.panel.panel_context import role_label

    return f"{load_persona_name(role)} ({role_label(role)})"


def _persona_summary(role: str) -> str:
    """Appraisal-only profile (``appraisal_profile``) → panel personality → first line."""
    from agent.config import load_persona_personality, load_persona_yaml

    raw = load_persona_yaml(role)
    text = str(raw.get("appraisal_profile") or "").strip() or load_persona_personality(role)
    if not text:
        instr = str(raw.get("instructions") or "").strip()
        text = instr.splitlines()[0] if instr else ""
    return " ".join(text.split())


APPRAISAL_EXAMPLES = (
    "[pad]: stung | unpleasant | -0.30 0.25 -0.20",
    "[pad]: validated | pleasant | 0.25 0.10 0.20",
    "[pad]: saddened on their behalf | unpleasant | -0.25 0.10 -0.05",
    "[pad]: mildly amused | pleasant | 0.10 0.05 0.00",
    "[pad]: settled | neutral | 0.00 -0.10 0.00",
    "[pad]: uneasy | unpleasant | -0.15 0.15 -0.10",
    "[pad]: unmoved | neutral | 0.00 0.00 0.00",
)


def render_appraisal_prompt(
    *,
    name: str,
    persona: str,
    mood: str,
    pad: Sequence[float],
    context_lines: Sequence[str],
    new_lines: Sequence[tuple[str, str]],
    addressed: bool = False,
    reply_to_self: bool = False,
    self_spoken: bool = False,
) -> tuple[str, str]:
    """Pure (system, user) renderer — shared by the live show and offline harnesses.

    ``new_lines`` is one ``(speaker_label, text)`` per utterance being appraised
    (live show: the single latest line; Sentipolis-style rounds: both lines).
    """
    system = (
        f"You track the inner emotional reactions of {name} in a live conversation.\n"
        f"Who {name} is: {persona}\n"
        f"You never speak for {name}; you only report how the latest exchange made them feel."
    )
    context = "\n".join(context_lines) or "(start of the conversation)"
    heading = "New line" if len(new_lines) == 1 else "New exchange"
    if reply_to_self and len(new_lines) > 1:
        heading = f"New exchange — {name}'s own line and the reply to it"
    elif reply_to_self:
        heading = f"New line — a reply to what {name} said above"
    elif self_spoken:
        heading = f"New line — said by {name} themself"
    elif addressed:
        heading += " — addressed to them by name"
    new_block = "\n".join(f'{label}: "{text}"' for label, text in new_lines)
    examples = "\n".join(APPRAISAL_EXAMPLES)
    user = f"""{name}'s current mood: {mood} (PAD P={pad[0]:+.2f} A={pad[1]:+.2f} D={pad[2]:+.2f}).

Recent conversation:
{context}

{heading}:
{new_block}

How does this change {name}'s feelings, given their personality and stake in the topic?
Each change is in [-1, 1]:
- ΔP pleasure: + pleased / validated / amused, − annoyed / hurt / worried / disappointed
- ΔA arousal: + fired up / alert / provoked, − calmer / bored / settled
- ΔD dominance: + more confident or in control, − challenged / overruled / on the back foot
Calibration:
- Ordinary discussion moves each axis by about 0.05 or less; 0 is a normal answer.
- Reserve 0.15–0.4 for things that clearly land on {name}: a direct challenge, praise,
  agreement or attack aimed at them, or news that really matters to them.
- Finding a point interesting is not pleasure; at most it raises A a little.
- Hearing agreement, repetition, or a calm procedural remark can lower A.
- Disagreement with {name} usually lowers P or D, even when it is phrased politely.
- Hearing about someone's suffering, loss, or unfair treatment lowers P (sad or angry on
  their behalf), even when the story is gripping. Their good news raises P.
- The axes often move in different directions.
- First decide whether this feels pleasant, unpleasant, or neutral to {name}; ΔP must have
  the matching sign (unpleasant → negative, pleasant → positive, neutral → about 0).

Illustrative examples (choose values for THIS exchange, do not copy):
{examples}

Answer with exactly one line:
[pad]: <feeling in 1-3 words> | <pleasant|unpleasant|neutral> | <ΔP> <ΔA> <ΔD>"""
    return system, user


def build_appraisal_prompt(
    data: TalkShowData,
    role: str,
    utt: Utterance,
) -> tuple[str, str]:
    """(system, user) for one live-show listener's appraisal of ``utt``."""
    from agent.config import load_persona_name
    from agent.emotion.role_pad import get_role_pad

    name = load_persona_name(role)
    pad = get_role_pad(data, role)
    neighbors = (getattr(data, "role_pad_neighbors", None) or {}).get(role) or [
        get_role_emotion(data, role)
    ]
    context = list(utt.context or snapshot_context(data, utt.text))
    protocol = appraisal_protocol()
    own = replied_line(role, utt, context) if protocol != "line" else None
    new_lines = [(_speaker_label(utt.speaker), utt.text)]
    earlier = context
    if own is not None and protocol == "exchange":
        earlier = [c for c in context if c is not own]
        new_lines.insert(0, (own[1], own[2]))
    context_lines = [
        f"{c[1]}: {c[2]}" + ("   ← the line being answered" if c is own else "")
        for c in earlier[-_CONTEXT_LINES:]
    ]
    return render_appraisal_prompt(
        name=name,
        persona=_persona_summary(role),
        mood=affect_descriptor(pad, neighbors),
        pad=pad,
        context_lines=context_lines,
        new_lines=new_lines,
        addressed=name.lower() in utt.text.lower(),
        reply_to_self=own is not None,
    )


def snapshot_context(
    data: TalkShowData, text: str
) -> tuple[tuple[str, str, str, bool], ...]:
    """Lines before ``text`` in history (history already ends with ``text``)."""
    lines = list(data.show_history.lines)
    if lines and lines[-1].text == text:
        lines = lines[:-1]
    return tuple(
        (ln.role_id, ln.speaker, ln.text, bool(getattr(ln, "procedural", False)))
        for ln in lines[-(_CONTEXT_LINES + 1) :]
    )


# --- effectful ------------------------------------------------------------


async def run_appraisal(
    data: TalkShowData, role: str, utt: Utterance
) -> AppraisalResult:
    system, user = build_appraisal_prompt(data, role, utt)
    t0 = time.monotonic()
    try:
        raw = await data.runtime.gemma_client.complete_text(
            user,
            system_prompt=system,
            history_messages=None,
            raw=True,
            temperature=_env_float(
                "TALKSHOW_PAD_APPRAISAL_TEMPERATURE", DEFAULT_APPRAISAL_TEMPERATURE
            ),
            max_tokens=int(
                _env_float(
                    "TALKSHOW_PAD_APPRAISAL_MAX_TOKENS", DEFAULT_APPRAISAL_MAX_TOKENS
                )
            ),
        )
    except Exception as exc:  # network / vLLM hiccup: keep the show going
        logger.exception("pad appraisal failed role=%s", role)
        return AppraisalResult(role, None, None, "", time.monotonic() - t0, repr(exc))
    parsed = parse_pad_appraisal_full(raw)
    return AppraisalResult(
        role,
        parsed.word,
        parsed.delta,
        raw or "",
        time.monotonic() - t0,
        valence=parsed.valence,
        flipped=parsed.flipped,
    )


def apply_appraisal(data: TalkShowData, utt: Utterance, result: AppraisalResult) -> None:
    from agent.emotion.pad_pipeline import apply_pad_delta_from_parsed

    if result.delta is not None:
        apply_pad_delta_from_parsed(data, result.role, result.delta, reason="appraisal")
    if data.turn_log is not None:
        data.turn_log.log(
            "pad_appraisal",
            role=result.role,
            speaker=utt.speaker,
            seq=utt.seq,
            protocol=appraisal_protocol(),
            reply_to_self=replied_line(result.role, utt, utt.context) is not None,
            word=result.word,
            valence=result.valence,
            sign_flipped=result.flipped,
            delta=list(result.delta) if result.delta is not None else None,
            ok=result.delta is not None,
            error=result.error,
            latency_s=round(result.latency_s, 3),
            text=utt.text[:200],
            raw=result.raw[:300] or None,
            room=getattr(data, "room_name", "") or "",
        )


# --- scheduling glue (replaced by a graph node later) ---------------------


async def _appraise_chained(
    data: TalkShowData,
    role: str,
    utt: Utterance,
    prev: asyncio.Task | None,
) -> None:
    result = await run_appraisal(data, role, utt)
    if prev is not None and not prev.done():
        # Apply in utterance order per role; LLM calls themselves run in parallel.
        await asyncio.wait([prev])
    apply_appraisal(data, utt, result)


DEFAULT_EXPRESSION_AROUSAL_DECAY = 0.12


def expression_arousal_decay() -> float:
    """Fraction of the speaker's arousal released by speaking (0 disables)."""
    return max(0.0, min(1.0, _env_float(
        "TALKSHOW_PAD_EXPRESSION_AROUSAL_DECAY", DEFAULT_EXPRESSION_AROUSAL_DECAY
    )))


def relax_after_expression(data: TalkShowData, speaker: str) -> float | None:
    """Speaking lowers the speaker's own arousal proportionally: A ← A·(1 − γ).

    Ours for LLM agents, after Schweitzer & Garcia 2010 (agent-based model: arousal
    drops proportionally after expression) and Garcia et al. 2016 (arousal decreases
    after participating in a discussion). P and D are unchanged; no LLM call.
    Returns the new arousal, or None when not applied.
    """
    from agent.emotion.role_pad import get_role_pad, set_role_pad

    gamma = expression_arousal_decay()
    if not appraisal_enabled() or gamma <= 0 or speaker in ("", "human"):
        return None
    if speaker not in ai_roles(data):
        return None
    p, a, d = get_role_pad(data, speaker)
    new_a = a * (1.0 - gamma)
    set_role_pad(data, speaker, (p, new_a, d))
    if data.turn_log is not None:
        data.turn_log.log(
            "pad_expression",
            role=speaker,
            gamma=round(gamma, 3),
            arousal_before=round(a, 4),
            arousal_after=round(new_a, 4),
            room=getattr(data, "room_name", "") or "",
        )
    return new_a


def on_utterance(data: TalkShowData, speaker: str, text: str) -> list[str]:
    """Hook for every spoken line: speaker relaxes, listeners appraise in background."""
    if not (text or "").strip():
        return []
    relax_after_expression(data, speaker)
    return schedule_appraisals(data, speaker, text)


def schedule_appraisals(data: TalkShowData, speaker: str, text: str) -> list[str]:
    """Fire-and-forget appraisal for every listener of a new line. Returns listeners."""
    if not appraisal_enabled() or not (text or "").strip():
        return []
    if getattr(data, "runtime", None) is None or data.runtime.gemma_client is None:
        return []
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        return []

    seq = len(data.show_history.lines)
    text = text.strip()
    utt = Utterance(
        speaker=speaker, text=text, seq=seq, context=snapshot_context(data, text)
    )
    listeners = appraisal_listeners(data, speaker)
    pending = data.pending_appraisals
    for role in listeners:
        prev = pending.get(role)
        pending[role] = loop.create_task(
            _appraise_chained(data, role, utt, prev),
            name=f"pad_appraisal:{role}:{seq}",
        )
    logger.debug("pad appraisal scheduled speaker=%s listeners=%s", speaker, listeners)
    return listeners


async def await_pending_appraisal(
    data: TalkShowData,
    role: str,
    *,
    timeout_s: float | None = None,
) -> bool:
    """Wait (bounded) for ``role``'s in-flight appraisals before building its prompt.

    Returns True if nothing was pending or it finished in time.
    """
    task = (getattr(data, "pending_appraisals", None) or {}).get(role)
    if task is None or task.done():
        return True
    wait = (
        timeout_s
        if timeout_s is not None
        else _env_float("TALKSHOW_PAD_APPRAISAL_WAIT_S", DEFAULT_APPRAISAL_WAIT_S)
    )
    t0 = time.monotonic()
    done, _ = await asyncio.wait([task], timeout=max(0.0, wait))
    ok = bool(done)
    if data.turn_log is not None:
        data.turn_log.log(
            "pad_appraisal_wait",
            role=role,
            waited_s=round(time.monotonic() - t0, 3),
            ok=ok,
            room=getattr(data, "room_name", "") or "",
        )
    if not ok:
        logger.info("pad appraisal for role=%s still running after %.1fs", role, wait)
    return ok


__all__: Sequence[str] = (
    "AppraisalResult",
    "Utterance",
    "ai_roles",
    "appraisal_enabled",
    "appraisal_listeners",
    "apply_appraisal",
    "await_pending_appraisal",
    "build_appraisal_prompt",
    "expression_arousal_decay",
    "on_utterance",
    "relax_after_expression",
    "APPRAISAL_PROTOCOLS",
    "appraisal_protocol",
    "render_appraisal_prompt",
    "snapshot_context",
    "replied_line",
    "run_appraisal",
    "schedule_appraisals",
)
