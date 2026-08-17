"""Passthrough LiveKit LLM that replays TurnStore replies without calling Gemma again."""

from __future__ import annotations

import uuid

from livekit.agents import llm
from livekit.agents.types import (
    DEFAULT_API_CONNECT_OPTIONS,
    NOT_GIVEN,
    APIConnectOptions,
    NotGivenOr,
)

from agent.adapters.turn_store import TurnStore


def _has_user_message(ctx: llm.ChatContext) -> bool:
    for item in ctx.items:
        if item.type == "message" and item.role == "user":
            return True
    return False


class StoredReplyLLM(llm.LLM):
    """Session LLM stub.

    Host replies after human audio are spoken via speak_panel_line
    (TalkShowAgent.on_user_turn_completed + StopResponse) — not this class.
    Remains required by AgentSession; only used if generate_reply slips through.
    """

    def __init__(self, turn_store: TurnStore) -> None:
        super().__init__()
        self._turn_store = turn_store

    @property
    def model(self) -> str:
        return "gemma-stored-reply"

    @property
    def provider(self) -> str:
        return "talkshow"

    def chat(
        self,
        *,
        chat_ctx: llm.ChatContext,
        tools: list[llm.Tool] | None = None,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
        parallel_tool_calls: NotGivenOr[bool] = NOT_GIVEN,
        tool_choice: NotGivenOr[llm.ToolChoice] = NOT_GIVEN,
        extra_kwargs: NotGivenOr[dict] = NOT_GIVEN,
    ) -> llm.LLMStream:
        return _StoredReplyStream(
            self,
            chat_ctx=chat_ctx,
            tools=tools or [],
            conn_options=conn_options,
        )


class _StoredReplyStream(llm.LLMStream):
    def __init__(
        self,
        parent: StoredReplyLLM,
        *,
        chat_ctx: llm.ChatContext,
        tools: list[llm.Tool],
        conn_options: APIConnectOptions,
    ) -> None:
        super().__init__(parent, chat_ctx=chat_ctx, tools=tools, conn_options=conn_options)
        self._parent = parent

    async def _run(self) -> None:
        turn = self._parent._turn_store.consume_turn()
        if turn:
            text = turn.reply
        elif not _has_user_message(self.chat_ctx):
            from agent.config import load_persona_name

            host = load_persona_name("host")
            text = (
                f"Hey there! I'm {host}, your host. "
                "What would you like to talk about today?"
            )
        else:
            text = "Sorry, I didn't catch that. Could you say it again?"
        chunk = llm.ChatChunk(
            id=str(uuid.uuid4()),
            delta=llm.ChoiceDelta(role="assistant", content=text),
        )
        await self._event_ch.send(chunk)
