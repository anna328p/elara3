"""Queue behaviour, against a throwaway in-memory database."""

from __future__ import annotations

from collections.abc import AsyncIterator
from datetime import datetime, timedelta

import pytest

from event_prototype.agents import Agent, AgentRole
from event_prototype.events import JobEvent, MessageEvent, Priority, ScheduledEvent
from event_prototype.queue import EventNotFound, EventQueue
from event_prototype.store import IN_MEMORY, LogAction, Status, utcnow

TRIAGE = Agent.spawn(AgentRole.TRIAGE)


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
) -> MessageEvent:
    return MessageEvent(
        timestamp=timestamp or utcnow(),
        description=description,
        sender=sender,
        channel="#workshop",
        body=body,
    )


async def test_submit_round_trips_a_typed_event(queue: EventQueue) -> None:
    fires_at = utcnow() + timedelta(hours=2)
    event = ScheduledEvent(
        timestamp=utcnow(),
        description="reminder fired",
        fires_at=fires_at,
        note="post the devlog",
    )

    row = await queue.get(await queue.submit(event, Priority.HIGH))

    assert row.priority is Priority.HIGH
    assert row.status is Status.PENDING
    restored = row.to_event()
    assert isinstance(restored, ScheduledEvent)
    assert restored.note == "post the devlog"
    # datetimes survive the JSON payload round trip
    assert restored.fires_at == fires_at


async def test_listing_is_ordered_by_urgency_then_age(queue: EventQueue) -> None:
    now = utcnow()
    low = await queue.submit(message("low"), Priority.LOW)
    newer_high = await queue.submit(message("newer", timestamp=now), Priority.HIGH)
    older_high = await queue.submit(
        message("older", timestamp=now - timedelta(hours=1)), Priority.HIGH
    )

    ordered = [row.id for row in await queue.list_events()]

    assert ordered == [older_high, newer_high, low]


async def test_edit_changes_only_what_was_passed(queue: EventQueue) -> None:
    event_id = await queue.submit(message(), Priority.LOW)

    await queue.edit(event_id, priority=Priority.REALTIME)
    row = await queue.get(event_id)

    assert row.priority is Priority.REALTIME
    assert row.description == "a message"


async def test_completed_events_leave_the_active_list_but_keep_their_report(
    queue: EventQueue,
) -> None:
    event_id = await queue.submit(message(), Priority.NORMAL)
    subagent = Agent.spawn(AgentRole.SUBAGENT)

    await queue.complete([event_id], agent=subagent, report="replied in #workshop")

    assert await queue.list_events() == []
    assert (await queue.get(event_id)).status is Status.COMPLETED

    (entry,) = (await queue.history_for([event_id]))[event_id]
    assert entry.action is LogAction.REPORT
    assert entry.detail == "replied in #workshop"
    assert entry.agent == subagent


async def test_deferred_events_stay_active_for_the_next_pass(queue: EventQueue) -> None:
    event_id = await queue.submit(message(), Priority.BACKGROUND)

    await queue.defer([(event_id, "nothing needs doing yet")], agent=TRIAGE)

    (row,) = await queue.list_events()
    assert row.id == event_id
    assert row.status is Status.DEFERRED


async def test_archiving_retires_an_event_without_deleting_it(queue: EventQueue) -> None:
    event_id = await queue.submit(message(), Priority.NORMAL)

    await queue.archive(event_id, agent=TRIAGE, reason="no longer relevant")

    # Gone from the working queue, and no longer addressable...
    assert await queue.list_events() == []
    with pytest.raises(EventNotFound):
        await queue.get(event_id)

    # ...but still on the record.
    history = await queue.list_events(active_only=False)
    assert [row.id for row in history] == [event_id]
    assert history[0].archived_at is not None


async def test_min_priority_filters_the_queue(queue: EventQueue) -> None:
    await queue.submit(message("background"), Priority.BACKGROUND)
    urgent = await queue.submit(message("urgent"), Priority.REALTIME)

    rows = await queue.list_events(min_priority=Priority.HIGH)

    assert [row.id for row in rows] == [urgent]


async def test_get_many_preserves_the_requested_order(queue: EventQueue) -> None:
    first = await queue.submit(message("one"), Priority.LOW)
    second = await queue.submit(
        JobEvent(
            timestamp=utcnow(),
            description="job done",
            job_id="render-1",
            outcome="succeeded",
            summary="240 frames",
        ),
        Priority.NORMAL,
    )

    rows = await queue.get_many([second, first])

    assert [row.id for row in rows] == [second, first]
    with pytest.raises(EventNotFound):
        await queue.get_many([first, 9999])


async def test_a_disposition_over_several_events_is_one_transaction(
    queue: EventQueue,
) -> None:
    ids = [await queue.submit(message(f"m{i}"), Priority.LOW) for i in range(3)]

    # A bad id in the batch means nothing is written, not a partial write.
    with pytest.raises(EventNotFound):
        await queue.defer([(i, "nope") for i in [*ids, 9999]], agent=TRIAGE)

    assert all(row.status is Status.PENDING for row in await queue.list_events())
    assert await queue.history_for(ids) == {}
