"""Being told that the pending set grew.

A subscription is a labelled, coalescing wake-up. It carries no filter: the
store is the truth and every consumer's read is idempotent, so all a wake needs
to say is that there is something new to read. What it does carry is why it
fired, so a consumer can tell an arrival from an escalation from an elapsed
wait and act on the priority without a read.

The queue owns one `Notifier` and wakes it after its own commits; nothing here
is global, and nothing polls. One process is assumed throughout.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum

from .events import Priority
from .store import utcnow


class Arrival(StrEnum):
    """How an event came to be pending."""

    SUBMITTED = "submitted"
    #: A deferred event the sweep returned to triage.
    ESCALATED = "escalated"


@dataclass(frozen=True, slots=True)
class Arrived:
    """An event entered the pending set."""

    event_id: int
    priority: Priority
    how: Arrival
    at: datetime


@dataclass(frozen=True, slots=True)
class Elapsed:
    """Nothing arrived; the time a subscriber asked to wait for ran out.

    What that means is the subscriber's to read — the schedule it was waiting
    on lives in the store, and the wake carries no copy of it.
    """

    at: datetime


type Wake = Arrived | Elapsed


class Subscription:
    """A label and whatever has arrived since the subscriber last looked.

    Many wakes before one wait come back as one list, in order, which is what
    turns a burst of messages into a single pass rather than one per message.
    """

    def __init__(self, label: str) -> None:
        #: Free text saying whose wake-up this is and what they do with it.
        #: Written once, read by people; nothing evaluates it.
        self.label = label
        self._arrived: list[Arrived] = []
        self._flag = asyncio.Event()

    def wake(self, arrived: Arrived) -> None:
        """Note an arrival. Synchronous, so a publisher never waits on a consumer."""
        self._arrived.append(arrived)
        self._flag.set()

    async def wait(self, timeout: float | None = None) -> list[Wake]:
        """Everything that arrived since the last wait, or an `Elapsed` if
        `timeout` ran out first.

        The accumulated list is taken and the flag cleared here, before the
        caller reads anything, so an arrival that lands during the caller's
        read survives to the next wait.
        """
        try:
            async with asyncio.timeout(timeout):
                await self._flag.wait()
        except TimeoutError:
            return [Elapsed(utcnow())]
        self._flag.clear()
        arrived, self._arrived = self._arrived, []
        return list(arrived)

    def __repr__(self) -> str:
        return f"Subscription({self.label!r})"


class Notifier:
    """Who is listening, and one way to wake them all. Owned by the queue."""

    def __init__(self) -> None:
        self._subscriptions: list[Subscription] = []

    def add(self, label: str) -> Subscription:
        subscription = Subscription(label)
        self._subscriptions.append(subscription)
        return subscription

    def remove(self, subscription: Subscription) -> None:
        self._subscriptions.remove(subscription)

    def wake(self, arrived: Arrived) -> None:
        """Fan an arrival out to every subscriber. Never awaits a consumer."""
        for subscription in self._subscriptions:
            subscription.wake(arrived)

    def labels(self) -> list[str]:
        """Who is listening, for a dashboard or a log line."""
        return [s.label for s in self._subscriptions]
