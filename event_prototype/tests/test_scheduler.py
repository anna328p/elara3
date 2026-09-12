"""The scheduler: when a wake-up turns into a pass. No model anywhere.

The loop tests run on real time with intervals of a few hundredths of a
second, and stop themselves through `until` from inside a pass.
"""

from __future__ import annotations

import asyncio
from datetime import timedelta

from event_prototype.agents import Agent, AgentRole
from event_prototype.events import Priority
from event_prototype.heartbeats import Beat
from event_prototype.queue import EventQueue
from event_prototype.scheduler import early, watch
from event_prototype.store import utcnow
from event_prototype.wake import Arrival, Arrived, Elapsed

from conftest import message

TRIAGE, SWEEP = AgentRole.TRIAGE, AgentRole.SWEEP


def arrived(priority: Priority) -> Arrived:
    return Arrived(1, priority, Arrival.SUBMITTED, utcnow())


def test_nothing_is_not_early() -> None:
    assert not early([])


def test_an_elapsed_wait_decides_nothing_on_its_own() -> None:
    assert not early([Elapsed(utcnow())])


def test_an_immediate_arrival_is_early_and_a_heartbeat_bound_one_is_not() -> None:
    assert early([arrived(Priority.NUDGE)])
    assert early([arrived(Priority.ACTIVE)])
    assert not early([arrived(Priority.BACKGROUND)])
    assert not early([arrived(Priority.ASYNC)])


def test_a_batch_is_early_if_any_wake_in_it_is() -> None:
    quiet = [arrived(Priority.BACKGROUND), arrived(Priority.ASYNC)]
    assert not early(quiet)
    assert early([*quiet, arrived(Priority.NUDGE)])


async def test_a_fresh_standing_schedule_runs_a_pass_at_once(queue: EventQueue) -> None:
    await queue.submit(message("already here"), Priority.BACKGROUND)
    await queue.ensure_standing(TRIAGE, 3600)
    until = asyncio.Event()
    beats: list[Beat] = []

    async def on_due(beat: Beat) -> None:
        beats.append(beat)
        until.set()

    async with asyncio.timeout(5):
        await watch(queue, on_due={TRIAGE: on_due}, until=until)

    (beat,) = beats
    assert beat.role is TRIAGE
    assert beat.due
    assert beat.next_due is not None


async def test_a_nudge_runs_a_pass_without_waiting_for_the_heartbeat(
    queue: EventQueue,
) -> None:
    await queue.submit(message("already here"), Priority.BACKGROUND)
    await queue.ensure_standing(TRIAGE, 3600)
    until = asyncio.Event()
    calls: list[int] = []

    async def on_due(beat: Beat) -> None:
        calls.append(len(calls) + 1)
        if len(calls) == 1:
            # The fresh schedule is due at once; this arrival is what the
            # second call is for.
            await queue.submit(message("look now"), Priority.NUDGE)
        else:
            assert not beat.due  # early: nothing on the schedule had come due
            until.set()

    async with asyncio.timeout(5):
        await watch(queue, on_due={TRIAGE: on_due}, until=until)

    assert calls == [1, 2]


async def test_a_background_arrival_waits_for_the_heartbeat(queue: EventQueue) -> None:
    await queue.submit(message("already here"), Priority.BACKGROUND)
    heartbeat = 0.05
    await queue.ensure_standing(TRIAGE, heartbeat)
    loop = asyncio.get_running_loop()
    until = asyncio.Event()
    called_at: list[float] = []

    async def on_due(beat: Beat) -> None:
        called_at.append(loop.time())
        if len(called_at) == 1:
            await queue.submit(message("can wait"), Priority.BACKGROUND)
        else:
            until.set()

    async with asyncio.timeout(5):
        await watch(queue, on_due={TRIAGE: on_due}, until=until)

    first, second = called_at
    # The timer may fire a hair early by the loop's clock resolution, never by more.
    assert second - first >= heartbeat - 0.005


async def test_an_idle_heartbeat_runs_nothing(queue: EventQueue) -> None:
    await queue.ensure_standing(TRIAGE, 0.01)
    until = asyncio.Event()
    calls = 0

    async def on_due(beat: Beat) -> None:
        nonlocal calls
        calls += 1

    watching = asyncio.create_task(watch(queue, on_due={TRIAGE: on_due}, until=until))
    await asyncio.sleep(0.05)  # several heartbeats with nothing pending
    until.set()
    async with asyncio.timeout(5):
        await watching

    assert calls == 0
    # The beats still happened: the schedule kept re-timing itself.
    (standing,) = await queue.heartbeats()
    assert standing.fired >= 2


async def test_an_early_pass_shifts_the_whole_schedule_back(queue: EventQueue) -> None:
    await queue.submit(message("already here"), Priority.BACKGROUND)
    heartbeat = 0.05
    await queue.ensure_standing(TRIAGE, heartbeat)
    loop = asyncio.get_running_loop()
    until = asyncio.Event()
    called_at: list[float] = []

    async def on_due(beat: Beat) -> None:
        called_at.append(loop.time())
        match len(called_at):
            case 1:
                # Part-way through the interval, a nudge lands; the
                # subscription holds it until this pass ends.
                await asyncio.sleep(0.02)
                await queue.submit(message("look now"), Priority.NUDGE)
            case 2:
                assert not beat.due
            case _:
                until.set()

    async with asyncio.timeout(5):
        await watch(queue, on_due={TRIAGE: on_due}, until=until)

    first, second, third = called_at
    assert second - first < heartbeat  # the early pass
    # The beat after it is a full interval after the early pass, not after
    # the first one — which would have put it well under 0.05 later.
    assert third - second >= heartbeat - 0.005


async def test_a_check_in_fires_on_time_and_its_message_reaches_the_pass(
    queue: EventQueue, triage: Agent
) -> None:
    await queue.submit(message("already here"), Priority.BACKGROUND)
    await queue.ensure_standing(TRIAGE, 3600)
    loop = asyncio.get_running_loop()
    until = asyncio.Event()
    beats: list[tuple[float, Beat]] = []

    async def on_due(beat: Beat) -> None:
        beats.append((loop.time(), beat))
        if len(beats) == 1:
            await queue.schedule_heartbeat(
                TRIAGE,
                due_at=utcnow() + timedelta(seconds=0.03),
                message="see if mira replied",
                created_by=triage,
            )
        else:
            until.set()

    async with asyncio.timeout(5):
        await watch(queue, on_due={TRIAGE: on_due}, until=until)

    (first_at, _), (second_at, second) = beats
    assert second_at - first_at >= 0.03 - 0.005
    assert second.due
    (note,) = second.check_ins
    assert note.message == "see if mira replied"
    assert note.left_by == triage


async def test_a_check_in_runs_a_pass_over_an_empty_queue(queue: EventQueue) -> None:
    await queue.schedule_heartbeat(TRIAGE, due_at=utcnow(), message="anything from quill?")
    until = asyncio.Event()
    beats: list[Beat] = []

    async def on_due(beat: Beat) -> None:
        beats.append(beat)
        until.set()

    async with asyncio.timeout(5):
        await watch(queue, on_due={TRIAGE: on_due}, until=until)

    (beat,) = beats
    assert [note.message for note in beat.check_ins] == ["anything from quill?"]


async def test_a_check_in_is_not_consumed_by_an_early_pass(queue: EventQueue) -> None:
    await queue.submit(message("already here"), Priority.BACKGROUND)
    await queue.ensure_standing(TRIAGE, 3600)
    check_in = await queue.schedule_heartbeat(
        TRIAGE, due_at=utcnow() + timedelta(seconds=10), message="later"
    )
    until = asyncio.Event()
    beats: list[Beat] = []

    async def on_due(beat: Beat) -> None:
        beats.append(beat)
        if len(beats) == 1:
            await queue.submit(message("look now"), Priority.NUDGE)
        else:
            until.set()

    async with asyncio.timeout(5):
        await watch(queue, on_due={TRIAGE: on_due}, until=until)

    _, nudged = beats
    assert nudged.check_ins == ()
    assert check_in.schedule.id in {s.schedule.id for s in await queue.heartbeats()}


async def test_an_override_lapses_on_its_own(queue: EventQueue) -> None:
    await queue.submit(message("already here"), Priority.BACKGROUND)
    # Long enough that it cannot fire during the test, short enough that the
    # loop's wait toward it ends and `until` is seen.
    standing = await queue.ensure_standing(TRIAGE, 1.0)
    now = utcnow()
    await queue.schedule_heartbeat(
        TRIAGE,
        due_at=now + timedelta(seconds=0.02),
        interval_seconds=0.02,
        expires_at=now + timedelta(seconds=0.07),
        message="busy",
    )
    until = asyncio.Event()
    calls = 0

    async def on_due(beat: Beat) -> None:
        nonlocal calls
        calls += 1

    watching = asyncio.create_task(watch(queue, on_due={TRIAGE: on_due}, until=until))
    await asyncio.sleep(0.3)
    # Startup, then the override at .02, .04 and .06; .08 would fall past .07.
    assert 2 <= calls <= 4
    (live,) = await queue.heartbeats()
    assert live.schedule.id == standing.schedule.id
    until.set()
    async with asyncio.timeout(5):
        await watching


async def test_each_role_runs_on_its_own_heartbeat(queue: EventQueue, triage: Agent) -> None:
    await queue.submit(message("pending"), Priority.BACKGROUND)
    deferred = await queue.submit(message("deferred"), Priority.BACKGROUND)
    await queue.defer([(deferred, "later")], agent=triage)
    await queue.ensure_standing(TRIAGE, 0.02)
    await queue.ensure_standing(SWEEP, 0.05)
    until = asyncio.Event()
    beats: dict[AgentRole, list[Beat]] = {TRIAGE: [], SWEEP: []}

    async def triage_pass(beat: Beat) -> None:
        beats[TRIAGE].append(beat)

    async def sweep_pass(beat: Beat) -> None:
        beats[SWEEP].append(beat)

    watching = asyncio.create_task(
        watch(queue, on_due={TRIAGE: triage_pass, SWEEP: sweep_pass}, until=until)
    )
    await asyncio.sleep(0.15)
    until.set()
    async with asyncio.timeout(5):
        await watching

    assert all(beat.role is SWEEP for beat in beats[SWEEP])
    assert all(beat.role is TRIAGE for beat in beats[TRIAGE])
    assert 2 <= len(beats[SWEEP]) < len(beats[TRIAGE])


async def test_a_tick_that_comes_due_during_a_pass_fires_when_the_pass_ends(
    queue: EventQueue, triage: Agent
) -> None:
    await queue.submit(message("pending"), Priority.BACKGROUND)
    deferred = await queue.submit(message("deferred"), Priority.BACKGROUND)
    await queue.defer([(deferred, "later")], agent=triage)
    await queue.ensure_standing(SWEEP, 0.02)
    await queue.ensure_standing(TRIAGE, 3600)
    loop = asyncio.get_running_loop()
    until = asyncio.Event()
    sweeps: list[float] = []
    triage_ended = 0.0

    async def sweep_pass(beat: Beat) -> None:
        sweeps.append(loop.time())
        if len(sweeps) == 2:
            until.set()

    async def triage_pass(beat: Beat) -> None:
        nonlocal triage_ended
        await asyncio.sleep(0.05)  # longer than the sweep's interval
        triage_ended = loop.time()

    async with asyncio.timeout(5):
        await watch(
            queue,
            # The sweep beats first, so its next tick comes due while triage runs.
            on_due={SWEEP: sweep_pass, TRIAGE: triage_pass},
            until=until,
        )

    assert triage_ended > 0
    # The second sweep ran as soon as triage let go, without another wait.
    assert sweeps[1] - triage_ended < 0.02
