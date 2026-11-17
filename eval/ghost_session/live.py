"""Live guest turns: the words come from an LLM, written from what the panel actually said.

A turn with ``generate: true`` has no ``text``; when its trigger line is announced the
client asks an OpenAI-compatible chat endpoint for one spoken reply (persona + topic +
recent transcript + the turn's ``goal``), then synthesizes it in the guest's voice.

Endpoint from env: ``GHOST_LLM_BASE_URL`` (default ``http://127.0.0.1:18000/v1``),
``GHOST_LLM_MODEL`` (default: the first id from ``/models``), ``GHOST_LLM_API_KEY``
(default ``EMPTY``).
"""

from __future__ import annotations

import json
import logging
import os
import re
import time
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger("ghost_session")

EMOTIONS = ("Neutral", "Happiness", "Anger", "Sadness", "Surprise")
RECENT_LINES = 12
MIN_WORDS, MAX_WORDS = 25, 70
# Words a listener has heard after N seconds of a line (a cut-in reacts to these only).
WORDS_PER_SEC = 2.7

FALLBACK_TEXT = (
    "Sorry, I kind of lost my train of thought there. Can I just ask the panel straight up, "
    "what would you actually tell someone like me who is stuck in the middle of all this?"
)

DEFAULT_PERSONA = "Howard, a guest on the show. Speaks casually and honestly."


@dataclass(frozen=True)
class TranscriptLine:
    role: str  # host / guest / commentator / human
    speaker: str  # display name ("" when the event had none)
    text: str
    step: str = ""

    def render(self) -> str:
        who = self.speaker or self.role or "?"
        return f"{who}: {self.text}"


@dataclass(frozen=True)
class GeneratedTurn:
    emotion: str
    text: str
    seconds: float
    attempts: int
    fallback: bool = False


@dataclass(frozen=True)
class LLMConfig:
    base_url: str
    model: str
    api_key: str
    timeout_sec: float = 45.0

    @classmethod
    def from_env(cls) -> "LLMConfig":
        return cls(
            base_url=(os.environ.get("GHOST_LLM_BASE_URL", "").strip() or "http://127.0.0.1:18000/v1").rstrip("/"),
            model=os.environ.get("GHOST_LLM_MODEL", "").strip(),
            api_key=os.environ.get("GHOST_LLM_API_KEY", "").strip() or "EMPTY",
            timeout_sec=float(os.environ.get("GHOST_LLM_TIMEOUT_SEC", "45") or 45),
        )


# --- prompt -----------------------------------------------------------------


def heard_prefix(text: str, after_sec: float) -> str:
    """The opening of a line a listener has heard ``after_sec`` seconds in."""
    words = text.split()
    n = max(6, int(round(after_sec * WORDS_PER_SEC)) + 2)
    if n >= len(words):
        return text
    return " ".join(words[:n]) + " …"


def _system_prompt(name: str, persona: str, topic: str) -> str:
    persona = persona.strip() or DEFAULT_PERSONA
    return (
        f"You are {name}, a real person who is a live guest on a TV talk show panel. "
        f"You are not an AI and never say you are one.\n"
        f"Who you are: {persona}\n"
        + (f"Show topic: {topic.strip()}\n" if topic.strip() else "")
        + "When you speak, write exactly what you say out loud, as casual natural speech: first person, "
        f"{MIN_WORDS} to {MAX_WORDS} words, contractions, plain words. You may disagree, and sometimes "
        "address a panelist by name. No stage directions, no actions in brackets or asterisks, "
        "no emojis, no lists, no hashtags. Write every number as words (twenty, not 20).\n"
    )


def _conversation(transcript: list[TranscriptLine]) -> str:
    recent = transcript[-RECENT_LINES:]
    if not recent:
        return "The show is just starting; nobody has said anything yet.\n"
    lines = "\n".join(line.render() for line in recent)
    return f"The conversation so far (most recent last):\n{lines}\n"


GOODBYE_MAX_WORDS = 32
GOODBYE_FALLBACK = (
    "Thank you all, this was really fun. I think I'll keep using AI, just with my own brain switched on. "
    "Sorry, I have to head out."
)


def build_goodbye_messages(
    *, name: str, persona: str, topic: str, transcript: list[TranscriptLine]
) -> list[dict[str, str]]:
    """The guest's short goodbye: he has to leave the show now."""
    system = _system_prompt(name, persona, topic) + (
        "Reply with JSON only, no other text: "
        '{"emotion": "<one of Neutral, Happiness, Anger, Sadness, Surprise>", "utterance": "<what you say>"}'
    )
    user = _conversation(transcript) + (
        "\nYou have to leave the show now. Say a short goodbye of fifteen to thirty words: thank the "
        "panel, give your one-sentence takeaway from this conversation, and say you have to head out. "
        "Do not ask a question. Answer with the JSON only."
    )
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


ACTIONS = ("silent", "respond", "cut_in")


def build_reaction_messages(
    *,
    name: str,
    persona: str,
    topic: str,
    transcript: list[TranscriptLine],
    line: TranscriptLine,
    addressed: str = "",
    turns_spoken: int = 0,
    lines_since_turn: int | None = None,
    allow_cut_in: bool = True,
    cut_after_sec: float = 2.5,
    must_respond: bool = False,
) -> list[dict[str, str]]:
    """Decide what the guest does about ``line`` (just announced, about to be heard) and,
    unless silent, write what he says: one call, so the audio can be ready by its end.

    ``addressed`` (addressing.py): "direct" (the floor is handed to him: must answer),
    "asked", "mentioned", or "". ``must_respond``: he has been quiet long enough that he
    jumps in after this line (the nudge)."""
    who = line.speaker or line.role
    system = _system_prompt(name, persona, topic) + (
        "You are an active participant, not an audience member. You do not comment on every line, "
        "but whenever you have something to add, you speak up: a quick experience of your own, "
        "a disagreement, a doubt, or a question for one of the panelists by name. A guest who "
        "never speaks unless asked is a bad guest.\n"
        "Reply with JSON only, no other text: "
        '{"action": "<silent, respond or cut_in>", "emotion": "<one of Neutral, Happiness, Anger, '
        'Sadness, Surprise>", "utterance": "<what you say, empty when silent>"}'
    )
    user = _conversation(transcript)
    user += f'\n{who} is saying this right now:\n"{line.text}"\n\n'
    if turns_spoken == 0:
        user += "You have not spoken yet.\n"
    else:
        ago = "" if lines_since_turn is None else f" {lines_since_turn} lines ago"
        user += f"You have spoken {turns_spoken} times so far; your last turn was{ago}.\n"
    if addressed == "direct":
        user += (
            f"{who} is talking to you directly. You must answer when {who} finishes: "
            'action "respond", and actually answer what you were asked.\n'
        )
    elif must_respond:
        user += (
            f"You have been quiet for a while, so you jump in when {who} finishes: action \"respond\". "
            f"React to something specific {who} just said: agree or disagree, add a short experience of "
            f"your own, or ask {who} or another panelist a pointed question.\n"
        )
    else:
        if addressed == "asked":
            user += f"{who} put a question to you in passing. You should answer it.\n"
        elif addressed == "mentioned":
            user += (
                f"{who} brought you into it. You should usually respond: pick up something specific "
                "they said and agree, push back, or add your own experience.\n"
            )
        user += (
            "What do you do about this line?\n"
            '- "respond": speak right after this line ends. Choose this whenever you have an experience, '
            "a disagreement, a doubt, or a question for a panelist.\n"
            '- "silent": keep listening, when you genuinely have nothing to add to this line.\n'
        )
        if allow_cut_in:
            heard = heard_prefix(line.text, cut_after_sec)
            user += (
                '- "cut_in": interrupt while they are still talking. Rare: only when the opening words, '
                f'"{heard}", say something you strongly disagree with; then react only to those words.\n'
            )
    user += "Answer with the JSON only."
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def build_messages(
    *,
    name: str,
    persona: str,
    topic: str,
    transcript: list[TranscriptLine],
    goal: str,
    cut_line: TranscriptLine | None = None,
    cut_after_sec: float = 0.0,
) -> list[dict[str, str]]:
    system = _system_prompt(name, persona, topic) + (
        "Reply with JSON only, no other text: "
        '{"emotion": "<one of Neutral, Happiness, Anger, Sadness, Surprise>", "utterance": "<what you say>"}'
    )
    convo = _conversation(transcript)
    if cut_line is not None:
        heard = heard_prefix(cut_line.text, cut_after_sec)
        who = cut_line.speaker or cut_line.role
        convo += (
            f"\n{who} has just started saying: \"{heard}\"\n"
            f"You cut in right now, while {who} is still talking, so react only to the part "
            f"you heard and to what came before.\n"
        )
    else:
        convo += "\nThe last speaker has just finished. It is your turn to speak.\n"
    task = f"What you want to do in this turn: {goal.strip()}\n" if goal.strip() else ""
    user = convo + task + "Answer with the JSON only."
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


# --- parsing / cleanup --------------------------------------------------------

_ONES = (
    "zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen "
    "fifteen sixteen seventeen eighteen nineteen"
).split()
_TENS = "_ _ twenty thirty forty fifty sixty seventy eighty ninety".split()


def number_words(n: int) -> str:
    if n < 0:
        return "minus " + number_words(-n)
    if n < 20:
        return _ONES[n]
    if n < 100:
        tens, ones = divmod(n, 10)
        return _TENS[tens] + ("" if not ones else "-" + _ONES[ones])
    if n < 1000:
        hundreds, rest = divmod(n, 100)
        return _ONES[hundreds] + " hundred" + ("" if not rest else " " + number_words(rest))
    for size, label in ((10**9, "billion"), (10**6, "million"), (1000, "thousand")):
        if n >= size:
            head, rest = divmod(n, size)
            return number_words(head) + f" {label}" + ("" if not rest else " " + number_words(rest))
    return str(n)


def _year_words(n: int) -> str:
    if 2000 <= n <= 2009:
        return number_words(n)
    head, tail = divmod(n, 100)
    return number_words(head) + " " + ("hundred" if tail == 0 else ("oh " + _ONES[tail] if tail < 10 else number_words(tail)))


def spell_numbers(text: str) -> str:
    text = re.sub(r"(\d+(?:\.\d+)?)\s*%", r"\1 percent", text)

    def repl(m: re.Match[str]) -> str:
        raw = m.group(0).replace(",", "")
        if "." in raw:
            whole, frac = raw.split(".", 1)
            return number_words(int(whole)) + " point " + " ".join(_ONES[int(d)] for d in frac)
        n = int(raw)
        if 1100 <= n <= 2099 and "," not in m.group(0):
            return _year_words(n)
        return number_words(n)

    return re.sub(r"\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+(?:\.\d+)?", repl, text)


_EMOJI = re.compile(
    "[\U0001F000-\U0001FAFF\U00002600-\U000027BF\U0001F900-\U0001F9FF‍️]+"
)


def clean_utterance(text: str) -> str:
    text = re.sub(r"\*[^*]*\*", " ", text)  # *laughs*
    text = re.sub(r"\[[^\]]*\]", " ", text)  # [sighs]
    text = re.sub(r"\((?:[a-z]+ly |laugh|sigh|pause|smil|chuckl|shrug|nod|lean)[^)]*\)", " ", text, flags=re.I)
    text = _EMOJI.sub("", text)
    text = text.replace("#", "").replace("*", "").replace("_", " ")
    text = re.sub(r"^\s*(?:Howard|Me|Guest)\s*:\s*", "", text)  # a leading "Howard:"
    text = spell_numbers(text)
    text = re.sub(r"\s+", " ", text).strip().strip('"').strip()
    return text


def trim_words(text: str, limit: int = MAX_WORDS) -> str:
    """Cut an over-long reply at the last sentence end within ``limit`` words."""
    words = text.split()
    if len(words) <= limit:
        return text
    head = " ".join(words[:limit])
    cut = max(head.rfind(". "), head.rfind("? "), head.rfind("! "))
    if head.endswith((".", "?", "!")):
        return head
    if cut > len(head) // 3:
        return head[: cut + 1]
    return head.rstrip(",;:") + "."


def normalize_emotion(raw: Any) -> str:
    value = str(raw or "").strip().strip(".").lower()
    aliases = {"happy": "Happiness", "joy": "Happiness", "angry": "Anger", "sad": "Sadness",
               "surprised": "Surprise", "calm": "Neutral"}
    for emo in EMOTIONS:
        if value == emo.lower():
            return emo
    return aliases.get(value, "Neutral")


def parse_reply(raw: str) -> tuple[str, str] | None:
    """(emotion, utterance) from the model's reply, or None when it has no usable line."""
    if not raw:
        return None
    body = raw.strip()
    fence = re.search(r"```(?:json)?\s*(.*?)```", body, re.S)
    if fence:
        body = fence.group(1).strip()
    emotion, utterance = "", ""
    match = re.search(r"\{.*\}", body, re.S)
    if match:
        try:
            obj = json.loads(match.group(0))
        except json.JSONDecodeError:
            obj = None
        if isinstance(obj, dict):
            emotion = str(obj.get("emotion") or "")
            utterance = str(obj.get("utterance") or obj.get("text") or obj.get("say") or "")
        if not utterance:
            # Almost-JSON (unescaped quotes, a trailing comma): pull the fields by regex.
            m_emo = re.search(r'"emotion"\s*:\s*"([^"]*)"', match.group(0))
            m_utt = re.search(r'"utterance"\s*:\s*"(.*)"\s*,?\s*\}?\s*$', match.group(0), re.S)
            emotion = emotion or (m_emo.group(1) if m_emo else "")
            utterance = m_utt.group(1) if m_utt else ""
    if not utterance:
        m_emo = re.search(r"emotion\s*[:=]\s*([A-Za-z]+)", body, re.I)
        m_utt = re.search(r"(?:utterance|say|text)\s*[:=]\s*(.+)", body, re.I | re.S)
        if m_utt:
            emotion = emotion or (m_emo.group(1) if m_emo else "")
            utterance = m_utt.group(1)
    utterance = clean_utterance(utterance)
    if len(utterance.split()) < 8:
        return None
    return normalize_emotion(emotion), trim_words(utterance)


def parse_reaction(raw: str) -> tuple[str, str, str] | None:
    """(action, emotion, utterance); silent → ("silent", "", ""). None: unusable reply."""
    if not raw:
        return None
    m = re.search(r'"?action"?\s*[:=]\s*"?([A-Za-z_\- ]+?)"?\s*[,}\n]', raw, re.I)
    action = (m.group(1).strip().lower().replace(" ", "_").replace("-", "_") if m else "")
    if action in ("cutin", "interrupt"):
        action = "cut_in"
    if action in ("speak", "answer", "reply"):
        action = "respond"
    if action == "silent":
        return "silent", "", ""
    parsed = parse_reply(raw)
    if parsed is None:
        return None
    if action not in ACTIONS:
        action = "respond"  # a line without a usable action: he wrote something to say
    return action, parsed[0], parsed[1]


@dataclass(frozen=True)
class Reaction:
    action: str  # silent / respond / cut_in
    emotion: str
    text: str
    seconds: float
    attempts: int
    fallback: bool = False


ANSWER_FALLBACK = (
    "Honestly? I'm not totally sure. I go back and forth on it every week, so I'd actually love "
    "to hear how the rest of you would answer that one."
)


# --- endpoint -----------------------------------------------------------------


class LiveWriter:
    """Writes the guest's live turns through an OpenAI-compatible chat endpoint."""

    def __init__(self, config: LLMConfig | None = None, *, transport: Any = None) -> None:
        self.config = config or LLMConfig.from_env()
        self._model = self.config.model
        self._transport = transport  # httpx transport (tests)
        self._client: Any = None

    def _http(self) -> Any:
        if self._client is None:
            import httpx

            self._client = httpx.AsyncClient(
                timeout=self.config.timeout_sec,
                headers={"Authorization": f"Bearer {self.config.api_key}"},
                transport=self._transport,
            )
        return self._client

    async def model(self) -> str:
        if not self._model:
            resp = await self._http().get(f"{self.config.base_url}/models")
            resp.raise_for_status()
            data = resp.json().get("data") or []
            if not data:
                raise RuntimeError(f"no models at {self.config.base_url}/models")
            self._model = str(data[0]["id"])
            logger.info("ghost LLM model=%s", self._model)
        return self._model

    async def _complete(self, messages: list[dict[str, str]], *, temperature: float) -> str:
        payload = {
            "model": await self.model(),
            "messages": messages,
            "temperature": temperature,
            "max_tokens": 300,
        }
        resp = await self._http().post(f"{self.config.base_url}/chat/completions", json=payload)
        resp.raise_for_status()
        choices = resp.json().get("choices") or []
        if not choices:
            return ""
        return str((choices[0].get("message") or {}).get("content") or "")

    async def write_turn(self, messages: list[dict[str, str]]) -> GeneratedTurn:
        """One spoken turn; retries once on an unparseable reply, then falls back."""
        t0 = time.monotonic()
        last_raw = ""
        for attempt in (1, 2):
            try:
                last_raw = await self._complete(messages, temperature=0.9 if attempt == 1 else 0.7)
            except Exception as exc:  # noqa: BLE001 — a dead endpoint must not stop the show
                logger.warning("ghost LLM call failed (attempt %d): %s", attempt, exc)
                continue
            parsed = parse_reply(last_raw)
            if parsed is not None:
                emotion, text = parsed
                return GeneratedTurn(emotion, text, time.monotonic() - t0, attempt)
            logger.warning("ghost LLM reply unparseable (attempt %d): %r", attempt, last_raw[:200])
        # Last resort: whatever prose came back, else a generic line; delivered Neutral.
        text = trim_words(clean_utterance(re.sub(r"[{}\"]|emotion|utterance", " ", last_raw)))
        if len(text.split()) < 8:
            text = FALLBACK_TEXT
        return GeneratedTurn("Neutral", text, time.monotonic() - t0, 2, fallback=True)

    async def react(self, messages: list[dict[str, str]], *, must_answer: bool = False) -> Reaction:
        """Decide + write. Retries once on an unusable reply (or a silent one when he must
        answer); then silent, or a generic answer when he was asked something."""
        t0 = time.monotonic()
        for attempt in (1, 2):
            try:
                raw = await self._complete(messages, temperature=0.9 if attempt == 1 else 0.7)
            except Exception as exc:  # noqa: BLE001
                logger.warning("ghost LLM call failed (attempt %d): %s", attempt, exc)
                continue
            parsed = parse_reaction(raw)
            if parsed is None or (must_answer and parsed[0] == "silent"):
                logger.warning("ghost LLM reaction unusable (attempt %d): %r", attempt, raw[:200])
                continue
            action, emotion, text = parsed
            if must_answer:
                action = "respond"
            return Reaction(action, emotion, text, time.monotonic() - t0, attempt)
        if must_answer:
            return Reaction("respond", "Neutral", ANSWER_FALLBACK, time.monotonic() - t0, 2, fallback=True)
        return Reaction("silent", "", "", time.monotonic() - t0, 2, fallback=True)

    async def aclose(self) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None
