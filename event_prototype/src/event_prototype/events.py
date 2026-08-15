"""The event protocol and the concrete event kinds the queue knows how to carry.

An event is an immutable record of something that happened (or is scheduled to
happen). Everything common to all events — when, and a one-line human summary —
lives on the base; everything kind-specific becomes the JSON payload, derived
from the dataclass fields so the two never drift apart.
"""

from __future__ import annotations

from dataclasses import dataclass, fields
from datetime import datetime
from enum import IntEnum
from typing import Any, ClassVar, Protocol, get_type_hints, runtime_checkable


class Priority(IntEnum):
    """Ordering for the queue. Higher is more urgent.

    Realtime events want a response now; background events wait for the next
    idle moment. The gaps leave room to insert levels later without a migration.
    """

    BACKGROUND = 0
    LOW = 10
    NORMAL = 20
    HIGH = 30
    REALTIME = 40


@runtime_checkable
class Event(Protocol):
    """What the queue requires of anything it stores.

    Read-only throughout: events are a record of what happened, so the queue
    only ever reads them. Declaring them as properties is what lets frozen
    dataclasses satisfy the protocol.
    """

    kind: ClassVar[str]

    @property
    def timestamp(self) -> datetime: ...

    @property
    def description(self) -> str: ...

    def payload(self) -> dict[str, Any]:
        """Kind-specific data, JSON-serializable."""
        ...


_registry: dict[str, type[BaseEvent]] = {}

# Fields every event has; they get their own columns rather than living in the payload.
_COMMON_FIELDS = ("timestamp", "description")


def register[E: type[BaseEvent]](cls: E) -> E:
    """Make an event kind reconstructible from a stored row."""
    if cls.kind in _registry:
        raise ValueError(f"duplicate event kind: {cls.kind}")
    _registry[cls.kind] = cls
    return cls


@dataclass(frozen=True, slots=True)
class BaseEvent:
    timestamp: datetime
    description: str

    kind: ClassVar[str]

    def payload(self) -> dict[str, Any]:
        return {
            f.name: _to_json(getattr(self, f.name))
            for f in fields(self)
            if f.name not in _COMMON_FIELDS
        }


def from_payload(
    kind: str, timestamp: datetime, description: str, payload: dict[str, Any]
) -> BaseEvent:
    """Rebuild a typed event from its stored columns."""
    try:
        cls = _registry[kind]
    except KeyError:
        raise ValueError(f"unknown event kind: {kind}") from None

    hints = get_type_hints(cls)
    restored = {name: _from_json(value, hints.get(name)) for name, value in payload.items()}
    return cls(timestamp=timestamp, description=description, **restored)


def _to_json(value: Any) -> Any:
    return value.isoformat() if isinstance(value, datetime) else value


def _from_json(value: Any, hint: Any) -> Any:
    if hint is datetime and isinstance(value, str):
        return datetime.fromisoformat(value)
    return value


@register
@dataclass(frozen=True, slots=True)
class MessageEvent(BaseEvent):
    """Someone said something to the character on a messaging platform."""

    kind: ClassVar[str] = "message"

    sender: str
    channel: str
    body: str


@register
@dataclass(frozen=True, slots=True)
class ScheduledEvent(BaseEvent):
    """A timer or alarm the agent set for itself, now fired."""

    kind: ClassVar[str] = "scheduled"

    fires_at: datetime
    note: str


@register
@dataclass(frozen=True, slots=True)
class JobEvent(BaseEvent):
    """A backgrounded task reporting back."""

    kind: ClassVar[str] = "job"

    job_id: str
    outcome: str
    summary: str
