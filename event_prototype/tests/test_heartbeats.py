"""Heartbeats: one schedule primitive, its ticks, and what a beat does to them.

Every test here hands `beat` a fixed clock, so nothing sleeps and nothing
depends on how fast the machine is.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest
from sqlalchemy.exc import IntegrityError

from event_prototype.agents import Agent, AgentRole
from event_prototype.queue import EventQueue
from event_prototype.store import (
    IN_MEMORY,
    HeartbeatRow,
    HeartbeatTickRow,
    create_engine,
    init_schema,
    session_factory,
)

from conftest import counting_selects, message

TRIAGE, SWEEP = AgentRole.TRIAGE, AgentRole.SWEEP
T = datetime(2026, 9, 11, 12, 0, tzinfo=UTC)


def at(seconds: float) -> datetime:
    return T + timedelta(seconds=seconds)


async def test_a_fresh_schedule_is_live_from_its_first_tick(queue: EventQueue) -> None:
    summary = await queue.schedule_heartbeat(TRIAGE, due_at=T, interval_seconds=300)

    assert summary.next_due == T
    assert summary.fired == 0
    assert summary.last_fired_at is None
    assert [s.schedule.id for s in await queue.heartbeats()] == [summary.schedule.id]
    assert await queue.next_heartbeat_due([TRIAGE]) == {TRIAGE: T}


async def test_a_beat_fires_every_due_tick_and_returns_its_messages_earliest_first(
    queue: EventQueue, triage: Agent
) -> None:
    await queue.schedule_heartbeat(TRIAGE, due_at=T, interval_seconds=300)
    await queue.schedule_heartbeat(TRIAGE, due_at=at(10), message="second", created_by=triage)
    await queue.schedule_heartbeat(TRIAGE, due_at=at(5), message="first", created_by=triage)

    beat = await queue.beat(TRIAGE, at=at(20))

    assert beat.due
    assert [note.message for note in beat.check_ins] == ["first", "second"]
    assert all(note.left_by == triage for note in beat.check_ins)
    assert beat.next_due == at(320)
    # The check-ins are spent; only the standing schedule is still live.
    (standing,) = await queue.heartbeats()
    assert standing.next_due == at(320)


async def test_a_beat_re_times_a_recurring_schedule_from_the_beat_whether_or_not_it_was_due(
    queue: EventQueue,
) -> None:
    await queue.schedule_heartbeat(TRIAGE, due_at=at(300), interval_seconds=300)

    beat = await queue.beat(TRIAGE, at=at(100))

    assert not beat.due  # nothing had come due: this was an early pass
    assert beat.next_due == at(400)
    (standing,) = await queue.heartbeats()
    assert standing.next_due == at(400)
    assert standing.fired == 1
    assert standing.last_fired_at == at(100)


async def test_a_check_in_is_left_alone_until_it_is_due(queue: EventQueue) -> None:
    await queue.schedule_heartbeat(TRIAGE, due_at=at(60), message="see if mira replied")

    early = await queue.beat(TRIAGE, at=at(30))
    assert early.check_ins == ()
    assert early.next_due == at(60)

    on_time = await queue.beat(TRIAGE, at=at(60))
    assert [note.message for note in on_time.check_ins] == ["see if mira replied"]
    assert on_time.next_due is None
    assert await queue.heartbeats() == []
    assert await queue.next_heartbeat_due([TRIAGE]) == {}


async def test_a_check_in_firing_counts_as_a_pass_and_re_times_the_standing_schedule(
    queue: EventQueue,
) -> None:
    await queue.schedule_heartbeat(TRIAGE, due_at=at(300), interval_seconds=300)
    await queue.schedule_heartbeat(TRIAGE, due_at=at(60), message="check the render")

    beat = await queue.beat(TRIAGE, at=at(60))

    assert [note.message for note in beat.check_ins] == ["check the render"]
    (standing,) = await queue.heartbeats()
    assert standing.next_due == at(360)


async def test_a_recurring_override_ends_once_its_successor_would_pass_its_expiry(
    queue: EventQueue,
) -> None:
    await queue.schedule_heartbeat(
        TRIAGE, due_at=at(60), interval_seconds=60, expires_at=at(150), message="busy"
    )

    first = await queue.beat(TRIAGE, at=at(60))
    assert first.next_due == at(120)

    second = await queue.beat(TRIAGE, at=at(120))
    assert second.next_due is None  # 180 would fall past 150
    assert await queue.heartbeats() == []


async def test_a_beat_touches_only_its_own_role(queue: EventQueue) -> None:
    await queue.schedule_heartbeat(TRIAGE, due_at=T, interval_seconds=300)
    await queue.schedule_heartbeat(SWEEP, due_at=T, interval_seconds=3600)

    await queue.beat(TRIAGE, at=T)

    assert await queue.next_heartbeat_due([TRIAGE, SWEEP]) == {TRIAGE: at(300), SWEEP: T}


async def test_next_due_is_reported_per_role_and_only_for_the_roles_asked(
    queue: EventQueue,
) -> None:
    await queue.schedule_heartbeat(TRIAGE, due_at=at(300), interval_seconds=300)
    await queue.schedule_heartbeat(TRIAGE, due_at=at(30), message="soon")
    await queue.schedule_heartbeat(SWEEP, due_at=at(3600), interval_seconds=3600)

    assert await queue.next_heartbeat_due([TRIAGE]) == {TRIAGE: at(30)}
    assert await queue.next_heartbeat_due([SWEEP]) == {SWEEP: at(3600)}
    assert await queue.next_heartbeat_due([]) == {}


async def test_ensure_standing_is_idempotent_when_the_interval_matches(
    queue: EventQueue,
) -> None:
    first = await queue.ensure_standing(TRIAGE, 300)
    second = await queue.ensure_standing(TRIAGE, 300)

    assert first.schedule.id == second.schedule.id
    assert first.schedule.standing
    assert len(await queue.heartbeats()) == 1


async def test_ensure_standing_retires_a_standing_schedule_whose_interval_changed(
    queue: EventQueue,
) -> None:
    old = await queue.ensure_standing(TRIAGE, 300)
    new = await queue.ensure_standing(TRIAGE, 60)

    assert new.schedule.id != old.schedule.id
    assert new.schedule.interval_seconds == 60
    assert new.fired == 0
    (live,) = await queue.heartbeats()
    assert live.schedule.id == new.schedule.id


async def test_ensure_standing_leaves_overrides_and_check_ins_alone(
    queue: EventQueue, triage: Agent
) -> None:
    override = await queue.schedule_heartbeat(
        TRIAGE, due_at=at(60), interval_seconds=60, expires_at=at(600), message="busy",
        created_by=triage,
    )
    check_in = await queue.schedule_heartbeat(TRIAGE, due_at=at(30), message="soon")

    standing = await queue.ensure_standing(TRIAGE, 300)

    live = {s.schedule.id for s in await queue.heartbeats()}
    assert live == {override.schedule.id, check_in.schedule.id, standing.schedule.id}


async def test_a_schedule_cannot_have_two_live_ticks() -> None:
    engine = create_engine(IN_MEMORY)
    await init_schema(engine)
    sessions = session_factory(engine)
    try:
        async with sessions.begin() as session:
            schedule = HeartbeatRow(role=TRIAGE, interval_seconds=300)
            session.add(schedule)
            await session.flush()
            session.add(HeartbeatTickRow(heartbeat_id=schedule.id, due_at=T))
            await session.flush()
            session.add(HeartbeatTickRow(heartbeat_id=schedule.id, due_at=at(1)))
            with pytest.raises(IntegrityError):
                await session.flush()
    finally:
        await engine.dispose()


async def test_a_beat_and_the_listing_are_one_query_each(
    queue: EventQueue, triage: Agent
) -> None:
    await queue.schedule_heartbeat(TRIAGE, due_at=T, interval_seconds=300)
    await queue.schedule_heartbeat(TRIAGE, due_at=T, message="now", created_by=triage)
    await queue.schedule_heartbeat(TRIAGE, due_at=at(30), message="later", created_by=triage)

    with counting_selects() as selects:
        beat = await queue.beat(TRIAGE, at=T)
    assert len(selects) == 1
    assert beat.check_ins[0].left_by == triage  # the creator rode along

    with counting_selects() as selects:
        summaries = await queue.heartbeats()
    assert len(selects) == 1
    assert len(summaries) == 2


async def test_workload_is_what_each_pass_would_see(queue: EventQueue, triage: Agent) -> None:
    assert await queue.workload(TRIAGE) == 0
    assert await queue.workload(SWEEP) == 0

    kept, deferred, done, gone = [await queue.submit(message(f"m{i}")) for i in range(4)]
    subagent = await queue.spawn(AgentRole.SUBAGENT)
    await queue.defer([(deferred, "later")], agent=triage)
    await queue.complete([done], agent=subagent, report="done")
    await queue.archive(gone, agent=triage, reason="noise")

    assert await queue.workload(TRIAGE) == 1
    assert await queue.workload(SWEEP) == 1
    (row,) = [row for row in await queue.list_events() if row.id == kept]
    assert row.id == kept
    with pytest.raises(ValueError, match="runs no pass"):
        await queue.workload(AgentRole.SUBAGENT)
