"""What a heartbeat hands a pass: values, no I/O.

The schedule itself is `HeartbeatRow` and `HeartbeatTickRow` in the store, and
`EventQueue.beat` is what fires it. These are the shapes that travel from the
beat into a pass, its prompt and its report — kept apart from the queue so the
renderer can read them without importing it.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

from .agents import Agent, AgentRole
from .store import HeartbeatRow


@dataclass(frozen=True, slots=True)
class CheckIn:
    """A note one pass left for a later one, delivered when its time came."""

    message: str
    #: Who left it; None is the operator.
    left_by: Agent | None
    left_at: datetime
    due_at: datetime


@dataclass(frozen=True, slots=True)
class Beat:
    """What one pass for a role found on its schedule.

    Returned by `EventQueue.beat` and handed to the pass, so the prompt can
    carry the check-ins and the report can say why the pass ran.
    """

    role: AgentRole
    at: datetime
    #: The check-ins that had come due, earliest due first.
    check_ins: tuple[CheckIn, ...]
    #: Whether any tick had come due. False is an early pass — an immediate
    #: arrival — that only re-timed the schedule.
    due: bool
    #: When the role is next due after this beat, from the rows in hand; None
    #: means nothing is scheduled.
    next_due: datetime | None


@dataclass(frozen=True, slots=True)
class HeartbeatSummary:
    """A live schedule, when it next fires, and how it has run so far."""

    schedule: HeartbeatRow
    next_due: datetime
    fired: int
    last_fired_at: datetime | None
