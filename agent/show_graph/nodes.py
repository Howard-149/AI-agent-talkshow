"""Show graph nodes — async; all side effects go through ``Actuators``.

Every spoken line, canned or generated, goes through one path:

    <node that decides the line> → prepare_speak ─┬─ Send → speak (TTS)
                                                   └─ Send → appraise × listener
                                       speak + appraise ──→ after_speak → line["after"]

``after_speak`` runs only once the TTS playout *and* every listener appraisal
are done, so the next speaker's prompt always sees an up-to-date mood without
waiting on anything extra (appraisal ≈ 0.5 s, playout ≈ several seconds).
"""

from __future__ import annotations

import logging
from typing import Any

from langchain_core.runnables import RunnableConfig
from langgraph.graph import END
from langgraph.types import Command, Send, interrupt

from agent.show_graph.actuators import Actuators
from agent.show_graph.state import LineSpec, ShowState, beat_reset

logger = logging.getLogger(__name__)


def _act(config: RunnableConfig) -> Actuators:
    return config["configurable"]["act"]


def _finish(**update: Any) -> Command:
    return Command(goto="finish", update=update)


def _say(line: LineSpec, after: str, **update: Any) -> Command:
    return Command(goto="prepare_speak", update={"line": line, "after": after, **update})


# --- entry ---------------------------------------------------------------


def route_entry(state: ShowState) -> str:
    return {"human": "human_turn", "moderate": "moderate", "session": "wait_event"}.get(
        state.get("entry", "panel"), "panel_consume"
    )


# --- session graph: wait for the next event between beats -------------------


async def wait_event(state: ShowState, config: RunnableConfig) -> Command:
    """Pause the session graph until the driver resumes it with an event.

    Events (dicts with ``kind``):
      opening  {text, texts}            welcome line, then moderate (session_start)
      human    {event: HumanEvent}      host reply + panel round
      moderate {trigger, skip_open_floor} host takes the floor (idle, …)
      panel    {trigger}                route on the current [next] tag
      stop                              end the session graph
    """
    # Nothing may run before interrupt(): the node re-executes on resume.
    ev = interrupt({"waiting": True, "beats": state.get("beats", 0)})
    kind = (ev or {}).get("kind", "")
    if kind == "stop":
        return Command(goto=END, update={"stopped": True})
    beat = beat_reset()
    if kind == "human":
        return Command(
            goto="human_turn",
            update={**beat, "entry": "human", "trigger": "after_human",
                    "human_event": ev.get("event") or {}},
        )
    if kind == "moderate":
        return Command(
            goto="moderate",
            update={**beat, "entry": "moderate",
                    "trigger": ev.get("trigger") or "host_moderate",
                    "skip_open_floor": bool(ev.get("skip_open_floor"))},
        )
    if kind == "panel":
        return Command(
            goto="panel_consume",
            update={**beat, "entry": "panel", "trigger": ev.get("trigger") or "panel"},
        )
    if kind == "opening":
        line: LineSpec = {
            "role": state["listen_role"],
            "text": ev.get("text") or "",
            "step": "session_welcome",
            "kind": "welcome",
            "procedural": True,
            "texts": ev.get("texts") or {},
        }
        return Command(
            goto="prepare_speak",
            update={**beat, "entry": "moderate", "trigger": "session_start",
                    "skip_open_floor": True, "line": line, "after": "moderate"},
        )
    logger.warning("show session: unknown event kind=%r — ignored", kind)
    return Command(goto="wait_event")


async def human_turn(state: ShowState, config: RunnableConfig) -> Command:
    """Human spoke: commit the turn, then the host's reply is spoken like any line."""
    act = _act(config)
    event = state.get("human_event") or {}
    reply, human_utt = await act.commit_human_turn(event)
    if act.should_stop():
        return _finish(stopped=True)
    extra = [human_utt] if human_utt is not None else []
    if not (reply or "").strip():
        logger.warning("human_turn: empty host reply — straight to floor routing")
        return Command(goto="panel_consume", update={"extra_utterances": extra})
    line: LineSpec = {
        "role": state["listen_role"],
        "text": reply,
        "step": "host_reply",
        "kind": "host_reply",
        "procedural": False,
    }
    return _say(line, "panel_consume", extra_utterances=extra)


# --- floor routing ---------------------------------------------------------


async def panel_consume(state: ShowState, config: RunnableConfig) -> Command:
    """Route on the [next] tag left by the last line (chains panelists directly)."""
    act = _act(config)
    if act.should_stop():
        return _finish(stopped=True)
    if state.get("turn_idx", 0) >= state.get("max_turns", 12):
        return Command(goto="close")
    nxt = act.consume_floor_next()
    if nxt in state["panel_roles"]:
        return Command(
            goto="panelist_line",
            update={"next_role": nxt, "grant_reason": "next_tag"},
        )
    if nxt == "human":
        return Command(goto="grant_human", update={"grant_reason": "next_tag"})
    if nxt == "close":
        return Command(goto="close")
    return Command(goto="moderate")


async def moderate(state: ShowState, config: RunnableConfig) -> Command:
    """Host takes the floor back; open it for hand raises unless a queue exists."""
    act = _act(config)
    if act.should_stop():
        return _finish(stopped=True)
    if state.get("moderations", 0) >= state.get("max_moderations", 8):
        logger.info("show_graph: moderation cap reached")
        return _finish()
    trigger = state.get("trigger") or "host_moderate"
    await act.return_to_host(trigger)
    queue = act.queue_roles()
    if queue:
        logger.info("open floor skip queue=%s trigger=%s", queue, trigger)
        act.log("open_floor_skip", reason="queue_nonempty", trigger=trigger, queue=queue)
        return Command(goto="hand_raise", update={"skip_open_floor": False})
    if state.get("skip_open_floor"):
        return Command(goto="hand_raise", update={"skip_open_floor": False})
    line: LineSpec = {
        "role": state["listen_role"],
        "text": act.open_floor_line(),
        "step": f"open_floor_{trigger}",
        "kind": "open_floor",
        "procedural": True,
    }
    act.prefetch_hand_raise_poll()  # no-op unless TALKSHOW_POLL_DURING_OPEN_FLOOR=1
    return _say(line, "hand_raise")


async def hand_raise(state: ShowState, config: RunnableConfig) -> Command:
    """Poll / FIFO queue / host decision → who speaks next."""
    act = _act(config)
    turn_idx = state.get("turn_idx", 0)
    pick = await act.hand_raise_round(turn_idx, list(state.get("spoken_roles") or []))
    moderations = state.get("moderations", 0) + 1
    if act.should_stop():
        return _finish(stopped=True, moderations=moderations)
    role = pick.role or ""
    trigger = state.get("trigger") or "host_moderate"
    if role in state["panel_roles"]:
        update = {"next_role": role, "grant_reason": pick.reason, "moderations": moderations}
        if pick.host_line:
            # Nobody raised: the host's own call-out is the intro (and is appraised).
            line: LineSpec = {
                "role": state["listen_role"],
                "text": pick.host_line,
                "step": f"direct_call_no_raises_{turn_idx}",
                "kind": "direct_call",
                "procedural": False,
                "target": role,
            }
            return _say(line, "panelist_line", **update)
        intro: LineSpec = {
            "role": state["listen_role"],
            "text": act.intro_line(role),
            "step": f"intro_{role}_{trigger}",
            "kind": "intro",
            "procedural": True,
            "target": role,
        }
        return _say(intro, "panelist_line", **update)
    if role == "human":
        return Command(
            goto="grant_human",
            update={"grant_reason": pick.reason or "queue_fifo", "moderations": moderations},
        )
    if role == "close":
        return Command(goto="close", update={"moderations": moderations})
    return _finish(moderations=moderations)


async def panelist_line(state: ShowState, config: RunnableConfig) -> Command:
    act = _act(config)
    if act.should_stop():
        return _finish(stopped=True)
    role = state["next_role"]
    reason = state.get("grant_reason") or "next_tag"
    turn_idx = state.get("turn_idx", 0)
    await act.grant_panelist(role, reason)
    step = f"speech_{role}_{turn_idx}"
    generated = await act.generate_panelist_line(role, step)
    spoken = list(state.get("spoken_roles") or [])
    if role not in spoken:
        spoken.append(role)
    line: LineSpec = {
        "role": role,
        "text": generated.text,
        "step": step,
        "kind": "panelist",
        "procedural": False,
        "next_tag": generated.next_role,
    }
    return _say(line, "panel_consume", turn_idx=turn_idx + 1, spoken_roles=spoken)


async def grant_human(state: ShowState, config: RunnableConfig) -> Command:
    act = _act(config)
    line: LineSpec = {
        "role": state["listen_role"],
        "text": act.human_floor_line(),
        "step": "grant_human",
        "kind": "grant_human",
        "reason": state.get("grant_reason") or "queue_fifo",
        "procedural": True,
    }
    return _say(line, "finish", outcome="human")


async def close(state: ShowState, config: RunnableConfig) -> Command:
    act = _act(config)
    line: LineSpec = {
        "role": state["listen_role"],
        "text": act.close_line(),
        "step": "close_round",
        "kind": "close",
        "procedural": True,
    }
    return _say(line, "finish", outcome="close")


# --- the one speaking path -------------------------------------------------


async def prepare_speak(state: ShowState, config: RunnableConfig) -> Command:
    """Record the line, relax the speaker, fan out TTS + listener appraisals."""
    act = _act(config)
    line = state.get("line")
    if not line or not (line.get("text") or "").strip():
        return Command(goto="after_speak")
    utt = act.record_line(line)
    if not line.get("procedural"):
        act.relax(line["role"])
    jobs: list[dict[str, Any]] = []
    for extra in state.get("extra_utterances") or []:
        jobs += [{"utt": extra, "listener": r} for r in act.listeners(extra.speaker)]
    if utt is not None:
        jobs += [{"utt": utt, "listener": r} for r in act.listeners(line["role"])]
    sends = [Send("speak", {"line": line})] + [Send("appraise", {"job": j}) for j in jobs]
    return Command(goto=sends, update={"extra_utterances": []})


async def speak(state: dict[str, Any], config: RunnableConfig) -> dict[str, Any]:
    line = state["line"]
    await _act(config).speak(line)
    return {"spoken": [line.get("step") or line.get("kind") or "line"]}


async def appraise(state: dict[str, Any], config: RunnableConfig) -> dict[str, Any]:
    job = state["job"]
    result = await _act(config).appraise(job["utt"], job["listener"])
    return {"appraisals": [result]}


async def after_speak(state: ShowState, config: RunnableConfig) -> Command:
    """Joined: line played and every listener has appraised it."""
    act = _act(config)
    line = state.get("line")
    if line:
        await act.after_line(line)
    update: dict[str, Any] = {"line": None, **act.snapshot()}
    after = state.get("after") or "finish"
    if act.should_stop():
        return Command(goto="finish", update={**update, "stopped": True})
    return Command(goto=after, update=update)


async def finish(state: ShowState, config: RunnableConfig) -> Command:
    """End of a beat: one-shot runs end; the session graph waits for the next event."""
    if state.get("session"):
        return Command(
            goto="wait_event",
            update={"beats": state.get("beats", 0) + 1,
                    "last_outcome": state.get("outcome") or ""},
        )
    return Command(goto=END)
