"""Shared scaffolding: a throwaway queue, an event to put in it, a query counter."""

from __future__ import annotations

from collections.abc import AsyncIterator, Generator
from contextlib import contextmanager
from datetime import datetime

import pytest
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

