"""The handling log: what happened to an event, in order, and who did it."""

from __future__ import annotations

from collections.abc import AsyncIterator, Generator
from contextlib import contextmanager

import pytest
from sqlalchemy import event as sa_event
from sqlalchemy.engine import Engine

from event_prototype.agents import Agent, AgentRole
from event_prototype.events import MessageEvent, Priority
from event_prototype.queue import EventQueue
from event_prototype.render import PromptRenderer
from event_prototype.store import IN_MEMORY, LogAction, Status, utcnow

TRIAGE = Agent.spawn(AgentRole.TRIAGE)


@pytest.fixture
async def queue() -> AsyncIterator[EventQueue]:
    async with await EventQueue.open(IN_MEMORY) as queue:
        yield queue


def message(body: str = "hello") -> MessageEvent:
    return MessageEvent(
        timestamp=utcnow(),
        description="a message",
        sender="mira",
        channel="#workshop",
        body=body,
    )


async def test_an_events_history_reads_as_a_sequence(queue: EventQueue) -> None:
    event_id = await queue.submit(message(), Priority.NORMAL)
    subagent = Agent.spawn(AgentRole.SUBAGENT)

    await queue.defer([(event_id, "no rush")], agent=TRIAGE)
    await queue.assign(
        [event_id],
        agent=TRIAGE,
        subagent=subagent,
        instructions="answer her",
        action=LogAction.HANDLE_ONE_EVENT,
    )
    await queue.complete([event_id], agent=subagent, report="answered")

    entries = (await queue.history_for([event_id]))[event_id]

    assert [e.action for e in entries] == [
        LogAction.DEFER_EVENT,
        LogAction.HANDLE_ONE_EVENT,
        LogAction.REPORT,
    ]
    # The assignment names both ends: who decided, and who got the work.
    assert entries[1].agent == TRIAGE
    assert entries[1].assigned_agent_id == subagent.id
    assert entries[2].agent == subagent


async def test_a_sequence_assignment_shares_one_subagent(queue: EventQueue) -> None:
    ids = [await queue.submit(message(f"m{i}"), Priority.NORMAL) for i in range(3)]
    subagent = Agent.spawn(AgentRole.SUBAGENT)

    await queue.assign(
        ids,
        agent=TRIAGE,
        subagent=subagent,
        instructions="in order",
        action=LogAction.HANDLE_EVENT_SEQUENCE,
    )

    history = await queue.history_for(ids)
    assert set(history) == set(ids)
    assert {e.assigned_agent_id for entries in history.values() for e in entries} == {
        subagent.id
    }


async def test_failure_is_recorded_and_leaves_the_event_alone(queue: EventQueue) -> None:
    event_id = await queue.submit(message(), Priority.NORMAL)
    subagent = Agent.spawn(AgentRole.SUBAGENT)

    await queue.record_failure([event_id], agent=subagent, error="RuntimeError: boom")

    assert (await queue.get(event_id)).status is Status.PENDING
    (entry,) = (await queue.history_for([event_id]))[event_id]
    assert entry.action is LogAction.FAILED
    assert "boom" in entry.detail


@contextmanager
def counting_selects() -> Generator[list[str], None, None]:
    """Every SELECT issued inside the block."""
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


async def test_sweep_fetches_history_without_a_query_per_event(
    queue: EventQueue,
) -> None:
    deferred = [await queue.submit(message(f"m{i}"), Priority.LOW) for i in range(5)]
    await queue.submit(message("fresh"), Priority.HIGH)
    await queue.defer([(i, "later") for i in deferred], agent=TRIAGE)

    with counting_selects() as selects:
        view = await queue.sweep_view()

    assert len(selects) == 2, selects  # one for events, one for the whole log slice
    assert [row.id for row in view.rows] == deferred  # the pending one is not swept
    assert set(view.history) == set(deferred)


async def test_triage_reads_the_queue_in_one_query(queue: EventQueue) -> None:
    deferred = [await queue.submit(message(f"m{i}"), Priority.LOW) for i in range(5)]
    pending = await queue.submit(message("fresh"), Priority.HIGH)
    await queue.defer([(i, "later") for i in deferred], agent=TRIAGE)

    with counting_selects() as selects:
        view = await queue.triage_view()

    # Triage sees the backlog as one-liners, so it pays for no history at all.
    assert len(selects) == 1, selects
    assert [row.id for row in view.pending] == [pending]
    assert [row.id for row in view.deferred] == deferred


async def test_deferred_history_reaches_the_sweep_prompt_not_the_triage_one(
    queue: EventQueue,
) -> None:
    event_id = await queue.submit(message(), Priority.LOW)
    await queue.defer([(event_id, "revisit if it goes unanswered")], agent=TRIAGE)

    triage = await queue.triage_view()
    triage_prompt = PromptRenderer().triage(triage.pending, triage.deferred)
    sweep = await queue.sweep_view()
    sweep_prompt = PromptRenderer().sweep(sweep.rows, sweep.history)

    # Triage gets the digest line only...
    assert "a message" in triage_prompt
    assert "revisit if it goes unanswered" not in triage_prompt
    # ...the sweep gets the reasoning and who wrote it.
    assert "revisit if it goes unanswered" in sweep_prompt
    assert TRIAGE.label in sweep_prompt


async def test_the_backlog_line_prefers_a_written_digest(queue: EventQueue) -> None:
    event_id = await queue.submit(message(), Priority.LOW)
    await queue.defer([(event_id, "not now")], agent=TRIAGE)

    def backlog_line() -> str:
        return next(
            line
            for line in PromptRenderer()
            .triage([], [row for row in view.deferred])
            .splitlines()
            if line.startswith(f"{event_id}:")
        )

    # Until one is written, the description stands in...
    view = await queue.triage_view()
    assert "a message" in backlog_line()

    # ...and once written, it is what triage reads.
    await queue.set_digest(event_id, "mira asked in #workshop about the gradient banding")
    view = await queue.triage_view()
    assert "gradient banding" in backlog_line()
    assert "a message" not in backlog_line()


async def test_escalation_returns_an_event_to_triage(queue: EventQueue) -> None:
    event_id = await queue.submit(message(), Priority.LOW)
    sweeper = Agent.spawn(AgentRole.SWEEP)
    await queue.defer([(event_id, "not yet")], agent=TRIAGE)

    await queue.escalate(
        event_id, agent=sweeper, priority=Priority.HIGH, reason="third time round"
    )

    view = await queue.triage_view()
    assert [row.id for row in view.pending] == [event_id]
    assert view.deferred == []
    assert view.pending[0].priority is Priority.HIGH


async def test_archiving_from_the_sweep_is_recorded_and_final(queue: EventQueue) -> None:
    event_id = await queue.submit(message(), Priority.BACKGROUND)
    sweeper = Agent.spawn(AgentRole.SWEEP)
    await queue.defer([(event_id, "automated notice")], agent=TRIAGE)

    await queue.archive(event_id, agent=sweeper, reason="informational, never actionable")

    assert (await queue.sweep_view()).rows == []
    assert (await queue.triage_view()).deferred == []
    entries = (await queue.history_for([event_id]))[event_id]
    assert entries[-1].action is LogAction.ARCHIVE_EVENT
    assert entries[-1].agent == sweeper
