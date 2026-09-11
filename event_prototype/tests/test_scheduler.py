"""The scheduler: when a wake-up turns into a pass. No model anywhere."""

from __future__ import annotations

import asyncio

from event_prototype.events import Priority
from event_prototype.queue import EventQueue
from event_prototype.scheduler import due, watch
from event_prototype.store import utcnow
from event_prototype.wake import Arrival, Arrived, Heartbeat

from conftest import message


def arrived(priority: Priority) -> Arrived:
    return Arrived(1, priority, Arrival.SUBMITTED, utcnow())


def test_nothing_is_not_due() -> None:
    assert not due([])


def test_a_heartbeat_is_always_due() -> None:
    assert due([Heartbeat(utcnow())])


def test_an_immediate_arrival_is_due_and_a_heartbeat_bound_one_is_not() -> None:
    assert due([arrived(Priority.NUDGE)])
    assert due([arrived(Priority.ACTIVE)])
    assert not due([arrived(Priority.BACKGROUND)])
    assert not due([arrived(Priority.ASYNC)])


def test_a_batch_is_due_if_any_wake_in_it_is() -> None:
    quiet = [arrived(Priority.BACKGROUND), arrived(Priority.ASYNC)]
    assert not due(quiet)
    assert due([*quiet, arrived(Priority.NUDGE)])


async def test_a_nudge_runs_a_pass_without_waiting_for_the_heartbeat(
    queue: EventQueue,
) -> None:
    await queue.submit(message("already here"), Priority.BACKGROUND)
    until = asyncio.Event()
    calls: list[int] = []

    async def on_due() -> None:
        calls.append(len(calls) + 1)
        if len(calls) == 1:
            # Startup is a heartbeat; this arrival is what the second call is for.
            await queue.submit(message("look now"), Priority.NUDGE)
        else:
            until.set()

    async with asyncio.timeout(5):
        await watch(queue, heartbeat_seconds=3600, on_due=on_due, until=until)

    assert calls == [1, 2]


async def test_a_background_arrival_waits_for_the_heartbeat(queue: EventQueue) -> None:
    await queue.submit(message("already here"), Priority.BACKGROUND)
    loop = asyncio.get_running_loop()
    until = asyncio.Event()
    heartbeat = 0.05
    called_at: list[float] = []

    async def on_due() -> None:
        called_at.append(loop.time())
        if len(called_at) == 1:
            await queue.submit(message("can wait"), Priority.BACKGROUND)
        else:
            until.set()

    async with asyncio.timeout(5):
        await watch(queue, heartbeat_seconds=heartbeat, on_due=on_due, until=until)

    first, second = called_at
    # The timer may fire a hair early by the loop's clock resolution, never by more.
    assert second - first >= heartbeat - 0.005


async def test_an_idle_heartbeat_runs_nothing(queue: EventQueue) -> None:
    until = asyncio.Event()
    calls = 0

    async def on_due() -> None:
        nonlocal calls
        calls += 1

    watching = asyncio.create_task(
        watch(queue, heartbeat_seconds=0.01, on_due=on_due, until=until)
    )
    await asyncio.sleep(0.05)  # several heartbeats with nothing pending
    until.set()
    async with asyncio.timeout(5):
        await watching

    assert calls == 0
