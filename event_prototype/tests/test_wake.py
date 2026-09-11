"""Wake-ups: who is told when the pending set grows, and what they are told."""

from __future__ import annotations

from event_prototype.agents import Agent, AgentRole
from event_prototype.events import Priority
from event_prototype.queue import EventQueue
from event_prototype.wake import Arrival, Arrived, Heartbeat, Wake

from conftest import message


def arrivals(wakes: list[Wake]) -> list[Arrived]:
    return [wake for wake in wakes if isinstance(wake, Arrived)]


async def test_a_submit_wakes_a_subscriber_with_the_arrival(queue: EventQueue) -> None:
    async with queue.subscribe("test") as subscription:
        event_id = await queue.submit(message(), Priority.NUDGE)

        (wake,) = await subscription.wait(timeout=0)

    assert isinstance(wake, Arrived)
    assert wake.event_id == event_id
    assert wake.priority is Priority.NUDGE
    assert wake.how is Arrival.SUBMITTED


async def test_arrivals_before_one_wait_come_back_together_in_order(
    queue: EventQueue,
) -> None:
    async with queue.subscribe("test") as subscription:
        ids = [await queue.submit(message(f"m{i}")) for i in range(3)]

        wakes = await subscription.wait(timeout=0)
        (after,) = await subscription.wait(timeout=0)

    assert [wake.event_id for wake in arrivals(wakes)] == ids
    assert len(wakes) == 3
    # Nothing else arrived, so the second wait is the timeout.
    assert isinstance(after, Heartbeat)


async def test_an_escalation_wakes_with_the_new_priority(
    queue: EventQueue, triage: Agent
) -> None:
    event_id = await queue.submit(message(), Priority.BACKGROUND)
    await queue.defer([(event_id, "later")], agent=triage)

    async with queue.subscribe("test") as subscription:
        await queue.escalate(event_id, agent=triage, priority=Priority.ACTIVE, reason="now")

        (wake,) = await subscription.wait(timeout=0)

    assert isinstance(wake, Arrived)
    assert wake.event_id == event_id
    assert wake.priority is Priority.ACTIVE
    assert wake.how is Arrival.ESCALATED


async def test_leaving_the_pending_set_wakes_nobody(queue: EventQueue, triage: Agent) -> None:
    deferred, done, gone = [await queue.submit(message(f"m{i}")) for i in range(3)]
    subagent = await queue.spawn(AgentRole.SUBAGENT)

    async with queue.subscribe("test") as subscription:
        await subscription.wait(timeout=0)  # the three submits
        await queue.defer([(deferred, "later")], agent=triage)
        await queue.complete([done], agent=subagent, report="done")
        await queue.archive(gone, agent=triage, reason="noise")

        (wake,) = await subscription.wait(timeout=0)

    assert isinstance(wake, Heartbeat)


async def test_a_subscription_ends_with_its_block(queue: EventQueue) -> None:
    async with queue.subscribe("the scheduler") as subscription:
        assert queue.subscriptions() == ["the scheduler"]
    assert queue.subscriptions() == []

    await queue.submit(message())

    (wake,) = await subscription.wait(timeout=0)
    assert isinstance(wake, Heartbeat)


async def test_pending_count_is_what_triage_would_see(queue: EventQueue, triage: Agent) -> None:
    assert await queue.pending_count() == 0

    kept, deferred, done, gone = [await queue.submit(message(f"m{i}")) for i in range(4)]
    subagent = await queue.spawn(AgentRole.SUBAGENT)
    await queue.defer([(deferred, "later")], agent=triage)
    await queue.complete([done], agent=subagent, report="done")
    await queue.archive(gone, agent=triage, reason="noise")

    assert await queue.pending_count() == 1
    (row,) = [row for row in await queue.list_events() if row.id == kept]
    assert row.id == kept
