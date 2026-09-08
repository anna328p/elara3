"""Streams: the ongoing contexts that events belong to.

A stream is a durable locus of activity — a chat room, a direct exchange, a
recurring job or schedule. Events name their own stream rather than being
assigned one, so two events in the same venue cannot disagree about which
stream that is.

Not every event belongs to one. A one-off alarm or a single job run is complete
in itself, and says so by naming no stream.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum


class StreamKind(StrEnum):
    """The shape of an ongoing context, not the platform it arrived from.

    Discord and IRC channels are both `CHANNEL`; a nightly backup and a weekly
    timer are both `JOB`. What differs between them belongs in the key.
    """

    #: Many parties sharing one venue.
    CHANNEL = "channel"
    #: Two parties: a DM thread, a mail correspondent.
    DIRECT = "direct"
    #: An automated recurrence — a job that runs again, a schedule that fires again.
    JOB = "job"


@dataclass(frozen=True, slots=True)
class StreamRef:
    """The identity of a stream, as the event that belongs to it knows it.

    `key` is stable and machine-facing, unique within its kind; `title` is the
    label a prompt reads.
    """

    kind: StreamKind
    key: str
    title: str
