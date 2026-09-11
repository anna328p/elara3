"""Deciding when a triage pass should happen.

The rule is the priority's own (see `Priority`): an immediate arrival is
addressed now, and the rest wait for the heartbeat. `watch` is that rule as a
loop over the queue's wake-ups, with the pass itself passed in so the loop can
be tested without a model.

Until a context has a heartbeat of its own, the context-bound priorities pass
through triage like the rest, so an active arrival runs a pass here and an
async one waits for the heartbeat; triage then assigns each into its stream's
context. When contexts wake on their own, `due` narrows to the triage-bound
priorities and nothing else here changes.
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable, Sequence

from .queue import EventQueue
from .wake import Heartbeat, Wake


def due(wakes: Sequence[Wake]) -> bool:
    """Whether this batch of wakes calls for a pass now.

    A heartbeat always does. An arrival does when its priority is immediate;
    the others are in the table for the next heartbeat.
    """
    for wake in wakes:
        if isinstance(wake, Heartbeat) or wake.priority.immediate:
            return True
    return False


async def watch(
    queue: EventQueue,
    *,
    heartbeat_seconds: float,
    on_due: Callable[[], Awaitable[None]],
    until: asyncio.Event | None = None,
) -> None:
    """Run `on_due` when an immediate arrival lands, and on every heartbeat otherwise.

    Startup counts as a heartbeat, so work that arrived before the loop began
    is handled at once. A wake that lands during `on_due` is not lost: the
    subscription keeps it, and the next iteration sees it. The heartbeat is
    timed from the last pass, not the last wake, so a trickle of heartbeat-bound
    arrivals cannot starve it.

    `on_due` is skipped when nothing is pending, because a pass mints its
    triage agent before it looks, and an idle heartbeat should not leave one
    behind. Cancellation mid-pass propagates into the pass: its subagent tasks
    are cancelled, no failure row is written, and the events stay pending with
    their assignment action, to be triaged again on the next start.
    """
    loop = asyncio.get_running_loop()
    label = f"watch: triage on a nudge or an active arrival, otherwise every {heartbeat_seconds:g}s"
    async with queue.subscribe(label) as subscription:
        last_run = loop.time() - heartbeat_seconds
        while until is None or not until.is_set():
            remaining = max(0.0, last_run + heartbeat_seconds - loop.time())
            wakes = await subscription.wait(remaining)
            if not due(wakes):
                continue
            last_run = loop.time()
            if await queue.pending_count() == 0:
                continue
            await on_due()
