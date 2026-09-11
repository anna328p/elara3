"""Shared scaffolding: a throwaway queue, an event to put in it, a stand-in
for the model, and a query counter."""

from __future__ import annotations

from collections.abc import AsyncIterator, Generator, Sequence
from contextlib import contextmanager
from datetime import datetime
from types import SimpleNamespace
from typing import Any, cast

import pytest
from anthropic import Omit, omit
from anthropic.lib.tools import BetaAsyncToolRunner
from anthropic.types import Message, TextBlock, Usage
from anthropic.types.beta import BetaToolUseBlock, BetaUsage
from anthropic.types.beta.parsed_beta_message import (
    ParsedBetaContentBlock,
    ParsedBetaMessage,
    ParsedBetaTextBlock,
)
from sqlalchemy import event as sa_event
from sqlalchemy.engine import Engine

from event_prototype.agents import Agent, AgentRole
from event_prototype.events import MessageEvent
from event_prototype.queue import EventQueue
from event_prototype.store import IN_MEMORY, utcnow

#: A tool call as a scripted reply makes it: the call id, the tool, its input.
type ToolCall = tuple[str, str, dict[str, Any]]


@pytest.fixture
async def queue() -> AsyncIterator[EventQueue]:
    async with await EventQueue.open(IN_MEMORY) as queue:
        yield queue


@pytest.fixture
async def triage(queue: EventQueue) -> Agent:
    """A triage agent the queue knows about, for tests that act as one."""
    return await queue.spawn(AgentRole.TRIAGE)


def message(
    body: str = "hello",
    *,
    timestamp: datetime | None = None,
    description: str = "a message",
    sender: str = "mira",
    venue: str = "discord",
    conversation: str = "#workshop",
    direct: bool = False,
) -> MessageEvent:
    """A message with everything defaulted, so a test names only what it means."""
    return MessageEvent(
        timestamp=timestamp or utcnow(),
        description=description,
        sender=sender,
        venue=venue,
        conversation=conversation,
        body=body,
        direct=direct,
    )


@contextmanager
def counting_selects() -> Generator[list[str], None, None]:
    """Every SELECT issued inside the block.

    The queue promises a fixed number of queries for its views regardless of
    how much is on it, and this is what holds that promise to account.
    """
    statements: list[str] = []

    @sa_event.listens_for(Engine, "before_cursor_execute")
    def record(
        conn: object,
        cursor: object,
        statement: str,
        parameters: object,
        context: object,
        executemany: bool,
    ) -> None:
        if statement.lstrip().upper().startswith(("SELECT", "WITH")):
            statements.append(statement)

    try:
        yield statements
    finally:
        sa_event.remove(Engine, "before_cursor_execute", record)


def completion(text: str, *, cache_read: int = 0) -> Message:
    """A real `Message`, so what the code reads off a response is what the SDK gives."""
    return Message(
        id="msg_test",
        type="message",
        role="assistant",
        model="test",
        content=[TextBlock(type="text", text=text)],
        stop_reason="end_turn",
        stop_sequence=None,
        usage=Usage(input_tokens=10, output_tokens=5, cache_read_input_tokens=cache_read),
    )


def reply(
    text: str = "", *, tool_calls: Sequence[ToolCall] = (), cache_read: int = 0
) -> ParsedBetaMessage[Any]:
    """A completion as the tool runner returns one: text, and any tool calls."""
    content: list[ParsedBetaContentBlock[Any]] = []
    if text:
        content.append(ParsedBetaTextBlock(type="text", text=text, parsed_output=None))
    content.extend(
        BetaToolUseBlock(type="tool_use", id=call_id, name=name, input=arguments)
        for call_id, name, arguments in tool_calls
    )
    return ParsedBetaMessage(
        id="msg_test",
        type="message",
        role="assistant",
        model="test",
        content=content,
        stop_reason="tool_use" if tool_calls else "end_turn",
        stop_sequence=None,
        usage=BetaUsage(input_tokens=10, output_tokens=5, cache_read_input_tokens=cache_read),
    )


class FakeMessages:
    """Stands in for `client.messages`: answers every call with `text` and keeps
    what each call asked for."""

    def __init__(self, text: str) -> None:
        self.text = text
        self.calls: list[dict[str, Any]] = []

    async def create(self, **kwargs: Any) -> Message:
        self.calls.append(kwargs)
        return completion(self.text)


class FakeClient:
    """Stands in for `AsyncAnthropic`.

    `messages.create` (the digest) answers with `text`. `beta.messages.parse`
    (the subagent, through a real tool runner built by `tool_runner`) answers
    with the next scripted reply, or with `text` once the script runs out, and
    keeps every request it was sent. The runner is the SDK's own, so the tool
    loop, the tools, and the request shape are exercised for real.
    """

    def __init__(self, text: str = "done", *, script: Sequence[ParsedBetaMessage[Any]] = ()) -> None:
        self.messages = FakeMessages(text)
        self.beta = SimpleNamespace(messages=self)
        self.script = list(script)
        self.calls: list[dict[str, Any]] = []

    @property
    def text(self) -> str:
        return self.messages.text

    @text.setter
    def text(self, value: str) -> None:
        self.messages.text = value

    async def parse(self, **params: Any) -> ParsedBetaMessage[Any]:
        self.calls.append(params)
        return self.script.pop(0) if self.script else reply(self.messages.text)

    def tool_runner(
        self, *, tools: Any, max_iterations: int | Omit = omit, **params: Any
    ) -> BetaAsyncToolRunner[Any]:
        return BetaAsyncToolRunner(
            params=cast(Any, params),
            options={},
            tools=tools,
            client=cast(Any, self),
            max_iterations=None if isinstance(max_iterations, Omit) else max_iterations,
        )
