"""The event protocol and the concrete event kinds the queue knows how to carry.

An event is an immutable record of something that happened (or is scheduled to
happen). Everything common to all events — when, and a one-line human summary —
lives on the base; everything kind-specific becomes the JSON payload, derived
from the dataclass fields so the two never drift apart.

Each kind also derives the stream it belongs to, if any: see `streams.py`.
"""

from __future__ import annotations

from dataclasses import dataclass, fields
from datetime import datetime
from enum import IntEnum
from typing import Any, ClassVar, Literal, Protocol, get_args, get_type_hints, runtime_checkable

from .streams import StreamKind, StreamRef

#: How a priority level is named where a person or a model writes one: the
#: models' tool arguments and the config file. The stored value is an integer.
type PriorityName = Literal["background", "low", "normal", "high", "realtime"]
PRIORITY_NAMES: tuple[str, ...] = get_args(PriorityName.__value__)


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

    @classmethod
    def from_name(cls, name: str) -> Priority:
        try:
            return cls[name.upper()]
        except KeyError:
            raise ValueError(f"priority must be one of {PRIORITY_NAMES}, not {name!r}") from None


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

    @property
    def stream(self) -> StreamRef | None:
        """The ongoing context this event belongs to, if any.

        Derived from the event's own fields rather than supplied by the caller,
        so every message in a venue necessarily lands on the same stream. It
        is a property rather than a field, which keeps it out of `payload()`.
        """
        ...

    def payload(self) -> dict[str, Any]:
        """Kind-specific data, JSON-serializable."""
        ...


_registry: dict[str, type[BaseEvent]] = {}

# Fields every event has; they get their own columns rather than living in the payload.
#
# Everything else round-trips through the payload, which is why a new field on an
# existing kind must carry a default: `from_payload` passes only the keys a stored
# row actually has, so a required addition stops old rows reconstructing.
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

    @property
    def stream(self) -> StreamRef | None:
        """No stream by default: belonging to one is the exception, not the rule."""
        return None

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
    #: The medium this arrived over: "discord", "email", "irc". Orthogonal to
    #: the stream — nothing is ever "the email stream" — but it decides how a
    #: reply is written, so it travels with the event rather than being derived.
    venue: str
    #: The particular exchange within that venue: a room, a DM thread, a mail
    #: thread. Together with the venue this identifies the stream, so an
    #: ingestion layer should pass the platform's own id for it rather than
    #: reconstructing one — a Discord DM already has an id of its own.
    conversation: str
    body: str
    #: Whether the conversation is two-party rather than a shared room. Its id
    #: alone cannot say, and the two are not the same kind of thing: a room is a
    #: place, a direct exchange is a relationship.
    direct: bool = False

    @property
    def stream(self) -> StreamRef | None:
        """One conversation is one stream. The venue only qualifies its id."""
        key = f"{self.venue}:{self.conversation}"
        if self.direct:
            # Titled by correspondent: on an incoming message that is the other
            # party, and it reads better than a thread id would. A platform with
            # opaque ids would have to carry a display name of its own.
            return StreamRef(StreamKind.DIRECT, key, self.sender)
        return StreamRef(StreamKind.CHANNEL, key, self.conversation)


@register
@dataclass(frozen=True, slots=True)
class ScheduledEvent(BaseEvent):
    """A timer or alarm the agent set for itself, now fired."""

    kind: ClassVar[str] = "scheduled"

    fires_at: datetime
    note: str
    #: The recurring schedule this firing is one of. A one-off alarm has none,
    #: and so belongs to no stream.
    schedule: str | None = None

    @property
    def stream(self) -> StreamRef | None:
        if self.schedule is None:
            return None
        return StreamRef(
            StreamKind.JOB, f"schedule:{self.schedule}", f"{self.schedule} schedule"
        )


@register
@dataclass(frozen=True, slots=True)
class JobEvent(BaseEvent):
    """A backgrounded task reporting back."""

    kind: ClassVar[str] = "job"

    job_id: str
    outcome: str
    summary: str
    #: The recurring job these runs are of, as distinct from `job_id`, which
    #: names this one run. A one-off job has none.
    job: str | None = None

    @property
    def stream(self) -> StreamRef | None:
        if self.job is None:
            return None
        return StreamRef(StreamKind.JOB, f"job:{self.job}", f"{self.job} job")
