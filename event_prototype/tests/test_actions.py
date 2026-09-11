"""Event actions: what happened to an event, in order, and who did it."""

from __future__ import annotations

from uuid import uuid4

import pytest
from sqlalchemy.exc import IntegrityError

from event_prototype.agents import Agent, AgentRole
from event_prototype.events import Priority
from event_prototype.queue import EventQueue
from event_prototype.render import PromptRenderer
from event_prototype.store import Action, Status

from conftest import counting_selects, message


async def test_an_events_history_reads_as_a_sequence(
    queue: EventQueue, triage: Agent
) -> None:
    event_id = await queue.submit(message(), Priority.NORMAL)

    await queue.defer([(event_id, "no rush")], agent=triage)
    assignment = await queue.assign(
        [event_id], agent=triage, instructions="answer her", action=Action.HANDLE_ONE_EVENT
    )
    subagent = assignment.subagent
    await queue.complete([event_id], agent=subagent, report="answered")

    entries = (await queue.history_for([event_id]))[event_id]

    assert [e.action for e in entries] == [
        Action.DEFER_EVENT,
        Action.HANDLE_ONE_EVENT,
        Action.REPORT,
    ]
    # The assignment names both ends: who decided, and who got the work.
    assert entries[1].agent == triage
    assert entries[1].assigned_agent_id == subagent.id
    assert entries[2].agent == subagent


async def test_the_agents_on_an_entry_come_with_the_entry(
    queue: EventQueue, triage: Agent
) -> None:
    """Roles live on the agent row now, and reading them costs no extra query."""
    event_id = await queue.submit(message(), Priority.NORMAL)
    assignment = await queue.assign(
        [event_id], agent=triage, instructions="answer her", action=Action.HANDLE_ONE_EVENT
    )
    subagent = assignment.subagent

    with counting_selects() as selects:
        (entry,) = (await queue.history_for([event_id]))[event_id]

    assert len(selects) == 1, selects  # both agents joined in, not fetched after
    assert entry.agent.role is AgentRole.TRIAGE
    assert entry.assigned is not None
    assert entry.assigned == subagent
    assert entry.assigned.role is AgentRole.SUBAGENT
    assert triage.label in repr(entry)  # outside the session, still answerable


async def test_nothing_is_attributed_to_an_agent_the_store_never_minted(
    queue: EventQueue,
) -> None:
    """An `Agent` made by hand is not an agent: the foreign key refuses it."""
    event_id = await queue.submit(message(), Priority.NORMAL)
    stranger = Agent(AgentRole.TRIAGE, str(uuid4()))

    with pytest.raises(IntegrityError):
        await queue.defer([(event_id, "who?")], agent=stranger)

    # The status change and the action row are one transaction, so neither landed.
    assert (await queue.get(event_id)).status is Status.PENDING
    assert await queue.history_for([event_id]) == {}


async def test_a_sequence_assignment_shares_one_subagent(
    queue: EventQueue, triage: Agent
) -> None:
    ids = [await queue.submit(message(f"m{i}"), Priority.NORMAL) for i in range(3)]

    assignment = await queue.assign(
        ids, agent=triage, instructions="in order", action=Action.HANDLE_EVENT_SEQUENCE
    )
    subagent = assignment.subagent

    history = await queue.history_for(ids)
    assert set(history) == set(ids)
    assert {e.assigned_agent_id for entries in history.values() for e in entries} == {
        subagent.id
    }


async def test_failure_is_recorded_and_leaves_the_event_alone(queue: EventQueue) -> None:
    event_id = await queue.submit(message(), Priority.NORMAL)
    subagent = await queue.spawn(AgentRole.SUBAGENT)

    await queue.record_failure([event_id], agent=subagent, error="RuntimeError: boom")

    assert (await queue.get(event_id)).status is Status.PENDING
    (entry,) = (await queue.history_for([event_id]))[event_id]
    assert entry.action is Action.FAILED
    assert "boom" in entry.detail


async def test_sweep_fetches_history_without_a_query_per_event(
    queue: EventQueue, triage: Agent
) -> None:
    deferred = [await queue.submit(message(f"m{i}"), Priority.LOW) for i in range(5)]
    await queue.submit(message("fresh"), Priority.HIGH)
    await queue.defer([(i, "later") for i in deferred], agent=triage)

    with counting_selects() as selects:
        view = await queue.sweep_view()

    assert len(selects) == 2, selects  # one for events, one for the whole slice of actions
    assert [row.id for row in view.rows] == deferred  # the pending one is not swept
    assert set(view.history) == set(deferred)


async def test_triage_reads_the_queue_in_one_query(
    queue: EventQueue, triage: Agent
) -> None:
    deferred = [await queue.submit(message(f"m{i}"), Priority.LOW) for i in range(5)]
    pending = await queue.submit(message("fresh"), Priority.HIGH)
    await queue.defer([(i, "later") for i in deferred], agent=triage)

    with counting_selects() as selects:
        view = await queue.triage_view()

    # Triage sees the backlog as one-liners, so it pays for no history at all.
    assert len(selects) == 1, selects
    assert [row.id for row in view.pending] == [pending]
    assert [row.id for row in view.deferred] == deferred


async def test_deferred_history_reaches_the_sweep_prompt_not_the_triage_one(
    queue: EventQueue, triage: Agent
) -> None:
    event_id = await queue.submit(message(), Priority.LOW)
    await queue.defer([(event_id, "revisit if it goes unanswered")], agent=triage)

    triage_view = await queue.triage_view()
    triage_prompt = PromptRenderer().triage(triage_view.pending, triage_view.deferred)
    sweep = await queue.sweep_view()
    sweep_prompt = PromptRenderer().sweep(sweep.rows, sweep.history)

    # Triage gets the digest line only...
    assert "a message" in triage_prompt
    assert "revisit if it goes unanswered" not in triage_prompt
    # ...the sweep gets the reasoning and who wrote it.
    assert "revisit if it goes unanswered" in sweep_prompt
    assert triage.label in sweep_prompt


async def test_the_backlog_line_prefers_a_written_digest(
    queue: EventQueue, triage: Agent
) -> None:
    event_id = await queue.submit(message(), Priority.LOW)
    await queue.defer([(event_id, "not now")], agent=triage)

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


async def test_escalation_returns_an_event_to_triage(
    queue: EventQueue, triage: Agent
) -> None:
    event_id = await queue.submit(message(), Priority.LOW)
    sweeper = await queue.spawn(AgentRole.SWEEP)
    await queue.defer([(event_id, "not yet")], agent=triage)

    await queue.escalate(
        event_id, agent=sweeper, priority=Priority.HIGH, reason="third time round"
    )

    view = await queue.triage_view()
    assert [row.id for row in view.pending] == [event_id]
    assert view.deferred == []
    assert view.pending[0].priority is Priority.HIGH


async def test_archiving_from_the_sweep_is_recorded_and_final(
    queue: EventQueue, triage: Agent
) -> None:
    event_id = await queue.submit(message(), Priority.BACKGROUND)
    sweeper = await queue.spawn(AgentRole.SWEEP)
    await queue.defer([(event_id, "automated notice")], agent=triage)

    await queue.archive(event_id, agent=sweeper, reason="informational, never actionable")

    assert (await queue.sweep_view()).rows == []
    assert (await queue.triage_view()).deferred == []
    entries = (await queue.history_for([event_id]))[event_id]
    assert entries[-1].action is Action.ARCHIVE_EVENT
    assert entries[-1].agent == sweeper
