"""Deciding when each role's pass should happen.

The rule is the priority's own (see `Priority`): an immediate arrival is
addressed now, and the rest wait for the heartbeat. Here the heartbeat is a
row per role in the store (see `queue.beat`), so `watch` is a loop that sleeps
until the earliest tick across the roles it runs, fires whichever came due,
and runs triage early when an immediate arrival wakes it. Any pass re-times
its role's schedule from that moment, so an early pass shifts every later beat
back with it. The pass itself is passed in, so the loop can be tested without
a model.

Until a context has a heartbeat of its own, the context-bound priorities pass
through triage like the rest, so an active arrival runs a pass here and an
async one waits for the heartbeat; triage then assigns each into its stream's
context. When contexts wake on their own, `early` narrows to the triage-bound
priorities and nothing else here changes.
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable, Mapping, Sequence
from datetime import datetime

from .agents import AgentRole
from .heartbeats import Beat
from .queue import EventQueue
from .store import utcnow
from .wake import Arrived, Wake

#: A pass, given what its beat found on the schedule.
type OnDue = Callable[[Beat], Awaitable[None]]


def early(wakes: Sequence[Wake]) -> bool:
    """Whether this batch of wakes calls for a triage pass ahead of its heartbeat.

    An arrival does when its priority is immediate. An elapsed wait never
    does on its own: it means the schedule is worth reading, not that any
    one role is due.
    """
    return any(isinstance(wake, Arrived) and wake.priority.immediate for wake in wakes)


async def watch(
    queue: EventQueue,
    *,
    on_due: Mapping[AgentRole, OnDue],
    until: asyncio.Event | None = None,
) -> None:
    """Run each role's pass when its heartbeat comes due, and triage's when
    an immediate arrival lands.

    The schedule lives in the store, and the loop's one read of it serves
    both purposes: how long to sleep, and, on waking, which roles it was for.
    Passes run in `on_due`'s order, so triage precedes the sweep when both
    are due. A tick or arrival that lands during a pass is not lost: the
    re-read finds the tick, and the subscription still holds the arrival.

    Each pass is preceded by its beat, so an idle heartbeat still re-times
    the schedule; the pass itself is skipped when there is nothing for it to
    see, because a pass mints its agent before it looks and an idle beat
    should not leave one behind. A check-in is itself something to attend
    to, so it runs the pass even over an empty queue. Cancellation mid-pass
    propagates into the pass, as before.
    """
    roles = tuple(on_due)
    label = "watch: " + ", ".join(f"{role.value} on its heartbeat" for role in roles) + (
        "; triage on a nudge or an active arrival"
    )
    async with queue.subscribe(label) as subscription:
        pending: list[AgentRole] = []
        while True:
            for role in pending:
                beat = await queue.beat(role, at=utcnow())
                if beat.check_ins or await queue.workload(role):
                    await on_due[role](beat)
            if until is not None and until.is_set():
                return
            schedule = await queue.next_heartbeat_due(roles)
            now = utcnow()
            pending = [role for role in roles if role in schedule and schedule[role] <= now]
            if pending:
                continue
            wakes = await subscription.wait(_seconds_until(min(schedule.values(), default=None), now))
            if early(wakes):
                pending = [AgentRole.TRIAGE]
            # An elapsed wait leaves `pending` empty: the top of the loop
            # re-reads the schedule and finds who it was for.


def _seconds_until(at: datetime | None, now: datetime) -> float | None:
    """How long to wait for `at`; None for "nothing scheduled", which waits on
    arrivals alone. Wall-clock arithmetic handed to a monotonic timeout: a
    clock step costs one early wake and a re-read, never a spin or a miss."""
    if at is None:
        return None
    return max(0.0, (at - now).total_seconds())
