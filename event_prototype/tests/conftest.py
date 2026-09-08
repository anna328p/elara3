"""Shared scaffolding: a throwaway queue, an event to put in it, a stand-in
for the model, and a query counter."""

from __future__ import annotations

from collections.abc import AsyncIterator, Generator
from contextlib import contextmanager
from datetime import datetime
from typing import Any

import pytest
from anthropic.types import Message, TextBlock, Usage
from sqlalchemy import event as sa_event
from sqlalchemy.engine import Engine

from event_prototype.events import MessageEvent
from event_prototype.queue import EventQueue
from event_prototype.store import IN_MEMORY, utcnow


@pytest.fixture
async def queue() -> AsyncIterator[EventQueue]:
    async with await EventQueue.open(IN_MEMORY) as queue:
        yield queue


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
        if statement.lstrip().upper().startswith("SELECT"):
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


class FakeMessages:
    """Stands in for `client.messages`: answers every call with `text` and keeps
    what each call asked for."""

    def __init__(self, text: str) -> None:
        self.text = text
        self.calls: list[dict[str, Any]] = []

    async def create(self, **kwargs: Any) -> Message:
        self.calls.append(kwargs)
        return completion(self.text)
