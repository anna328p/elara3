"""Queue behaviour, against a throwaway in-memory database."""

from __future__ import annotations

from datetime import timedelta

import pytest

from event_prototype.agents import Agent, AgentRole
from event_prototype.events import JobEvent, Priority, PriorityMismatch, ScheduledEvent
from event_prototype.queue import EventNotFound, EventQueue
from event_prototype.store import Action, Status, utcnow

from conftest import message


async def test_submit_round_trips_a_typed_event(queue: EventQueue) -> None:
    fires_at = utcnow() + timedelta(hours=2)
    event = ScheduledEvent(
        timestamp=utcnow(),
        description="reminder fired",
        fires_at=fires_at,
        note="post the devlog",
    )

    row = await queue.get(await queue.submit(event, Priority.NUDGE))

    assert row.priority is Priority.NUDGE
    assert row.status is Status.PENDING
    restored = row.to_event()
    assert isinstance(restored, ScheduledEvent)
    assert restored.note == "post the devlog"
    # datetimes survive the JSON payload round trip
    assert restored.fires_at == fires_at


async def test_listing_is_ordered_by_urgency_then_age(queue: EventQueue) -> None:
    now = utcnow()
    low = await queue.submit(message("low"), Priority.BACKGROUND)
    newer_high = await queue.submit(message("newer", timestamp=now), Priority.NUDGE)
    older_high = await queue.submit(
        message("older", timestamp=now - timedelta(hours=1)), Priority.NUDGE
    )

    ordered = [row.id for row in await queue.list_events()]

    assert ordered == [older_high, newer_high, low]


async def test_edit_changes_only_what_was_passed(queue: EventQueue) -> None:
    event_id = await queue.submit(message(), Priority.BACKGROUND)

    await queue.edit(event_id, priority=Priority.ACTIVE)
    row = await queue.get(event_id)

    assert row.priority is Priority.ACTIVE
    assert row.description == "a message"


async def test_completed_events_leave_the_active_list_but_keep_their_report(
    queue: EventQueue,
) -> None:
    event_id = await queue.submit(message(), Priority.BACKGROUND)
    subagent = await queue.spawn(AgentRole.SUBAGENT)

    await queue.complete([event_id], agent=subagent, report="replied in #workshop")

    assert await queue.list_events() == []
    assert (await queue.get(event_id)).status is Status.COMPLETED

    (entry,) = (await queue.history_for([event_id]))[event_id]
    assert entry.action is Action.REPORT
    assert entry.detail == "replied in #workshop"
    assert entry.agent == subagent


async def test_deferred_events_stay_active_for_the_next_pass(
    queue: EventQueue, triage: Agent
) -> None:
    event_id = await queue.submit(message(), Priority.BACKGROUND)

    await queue.defer([(event_id, "nothing needs doing yet")], agent=triage)

    (row,) = await queue.list_events()
    assert row.id == event_id
    assert row.status is Status.DEFERRED


async def test_archiving_retires_an_event_without_deleting_it(
    queue: EventQueue, triage: Agent
) -> None:
    event_id = await queue.submit(message(), Priority.BACKGROUND)

    await queue.archive(event_id, agent=triage, reason="no longer relevant")

    # Gone from the working queue, and no longer addressable...
    assert await queue.list_events() == []
    with pytest.raises(EventNotFound):
        await queue.get(event_id)

    # ...but still on the record.
    history = await queue.list_events(active_only=False)
    assert [row.id for row in history] == [event_id]
    assert history[0].archived_at is not None


def test_a_priority_is_two_facts() -> None:
    assert not Priority.BACKGROUND.to_context and not Priority.BACKGROUND.immediate
    assert not Priority.NUDGE.to_context and Priority.NUDGE.immediate
    assert Priority.ASYNC.to_context and not Priority.ASYNC.immediate
    assert Priority.ACTIVE.to_context and Priority.ACTIVE.immediate


def test_the_listing_puts_the_woken_first_and_a_person_before_a_job() -> None:
    assert sorted(Priority, reverse=True) == [
        Priority.ACTIVE,
        Priority.NUDGE,
        Priority.ASYNC,
        Priority.BACKGROUND,
    ]


async def test_a_context_bound_priority_needs_a_stream(queue: EventQueue) -> None:
    alarm = ScheduledEvent(
        timestamp=utcnow(), description="one-off", fires_at=utcnow(), note="once"
    )
    for priority in (Priority.ASYNC, Priority.ACTIVE):
        with pytest.raises(PriorityMismatch, match="has no stream"):
            await queue.submit(alarm, priority)
    assert await queue.list_events() == []

    # The triage-bound ones are fine without.
    await queue.submit(alarm, Priority.NUDGE)


async def test_active_needs_a_conversation(queue: EventQueue) -> None:
    report = JobEvent(
        timestamp=utcnow(),
        description="backup",
        job_id="b1",
        outcome="ok",
        summary="done",
        job="backup",
    )
    with pytest.raises(PriorityMismatch, match="job stream"):
        await queue.submit(report, Priority.ACTIVE)

    # Async is allowed: the job's context takes the report at its own pace.
    await queue.submit(report, Priority.ASYNC)
    await queue.submit(message(direct=True), Priority.ACTIVE)


async def test_edit_and_escalate_refuse_a_priority_the_event_cannot_carry(
    queue: EventQueue, triage: Agent
) -> None:
    alarm = ScheduledEvent(
        timestamp=utcnow(), description="one-off", fires_at=utcnow(), note="once"
    )
    event_id = await queue.submit(alarm, Priority.BACKGROUND)

    with pytest.raises(PriorityMismatch):
        await queue.edit(event_id, priority=Priority.ASYNC)
    await queue.defer([(event_id, "later")], agent=triage)
    with pytest.raises(PriorityMismatch):
        await queue.escalate(event_id, agent=triage, priority=Priority.ACTIVE, reason="now")

    row = await queue.get(event_id)
    assert row.priority is Priority.BACKGROUND
    assert row.status is Status.DEFERRED


async def test_get_many_preserves_the_requested_order(queue: EventQueue) -> None:
    first = await queue.submit(message("one"), Priority.BACKGROUND)
    second = await queue.submit(
        JobEvent(
            timestamp=utcnow(),
            description="job done",
            job_id="render-1",
            outcome="succeeded",
            summary="240 frames",
        ),
        Priority.BACKGROUND,
    )

    rows = await queue.get_many([second, first])

    assert [row.id for row in rows] == [second, first]
    with pytest.raises(EventNotFound):
        await queue.get_many([first, 9999])


async def test_a_disposition_over_several_events_is_one_transaction(
    queue: EventQueue, triage: Agent
) -> None:
    ids = [await queue.submit(message(f"m{i}"), Priority.BACKGROUND) for i in range(3)]

    # A bad id in the batch means nothing is written, not a partial write.
    with pytest.raises(EventNotFound):
        await queue.defer([(i, "nope") for i in [*ids, 9999]], agent=triage)

    assert all(row.status is Status.PENDING for row in await queue.list_events())
    assert await queue.history_for(ids) == {}
