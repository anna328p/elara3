"""Turning queue rows into prompts.

Prompt text lives in Jinja templates on disk, not in Python string literals —
in the real framework these are invariants the agents can read but not edit.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta

from jinja2 import Environment, PackageLoader, StrictUndefined

from .store import EventLogRow, EventRow, utcnow


@dataclass(frozen=True, slots=True)
class LogView:
    """One past action on an event, flattened for a template."""

    timestamp: str
    action: str
    agent: str
    detail: str

    @classmethod
    def of(cls, entry: EventLogRow) -> LogView:
        return cls(
            timestamp=entry.timestamp.isoformat(timespec="seconds"),
            action=entry.action.value,
            agent=entry.agent.label,
            detail=entry.detail,
        )


@dataclass(frozen=True, slots=True)
class EventView:
    """One event, flattened for a template."""

    id: int
    kind: str
    priority: str
    status: str
    age: str
    timestamp: str
    description: str
    payload: str
    #: The line that stands in for this event where it is listed rather than
    #: shown, falling back to the description until one has been written.
    summary: str
    #: The ongoing context this event belongs to, as a prompt reads it. `None`
    #: for events that belong to nothing ongoing.
    stream: str | None = None
    history: tuple[LogView, ...] = ()

    @classmethod
    def of(
        cls, row: EventRow, now: datetime, history: Sequence[EventLogRow] = ()
    ) -> EventView:
        return cls(
            id=row.id,
            kind=row.kind,
            priority=row.priority.name.lower(),
            status=row.status.value,
            age=_humanize(now - row.timestamp),
            timestamp=row.timestamp.isoformat(timespec="seconds"),
            description=row.description,
            payload=json.dumps(row.payload, indent=2, sort_keys=True, ensure_ascii=False),
            summary=row.digest or row.description,
            stream=row.stream.title if row.stream else None,
            history=tuple(LogView.of(entry) for entry in history),
        )


def _humanize(delta: timedelta) -> str:
    seconds = int(delta.total_seconds())
    if seconds < 0:
        return f"in {_humanize(-delta)}"
    if seconds < 90:
        return f"{seconds}s ago"
    if seconds < 5400:
        return f"{seconds // 60}m ago"
    if seconds < 172800:
        return f"{seconds // 3600}h ago"
    return f"{seconds // 86400}d ago"


class PromptRenderer:
    """Renders the prompts this prototype needs."""

    def __init__(self, env: Environment | None = None) -> None:
        self._env = env or Environment(
            loader=PackageLoader("event_prototype", "templates"),
            undefined=StrictUndefined,
            trim_blocks=True,
            lstrip_blocks=True,
            keep_trailing_newline=True,
        )

    def triage(
        self,
        pending: Sequence[EventRow],
        deferred: Sequence[EventRow] = (),
        *,
        now: datetime | None = None,
    ) -> str:
        now = now or utcnow()
        return self._env.get_template("triage.md.j2").render(
            events=[EventView.of(row, now) for row in pending],
            deferred=[EventView.of(row, now) for row in deferred],
            now=now.isoformat(timespec="seconds"),
        )

    def digest(self, row: EventRow, reason: str, *, now: datetime | None = None) -> str:
        now = now or utcnow()
        return self._env.get_template("digest.md.j2").render(
            event=EventView.of(row, now), reason=reason
        )

    def sweep(
        self,
        rows: Sequence[EventRow],
        history: Mapping[int, Sequence[EventLogRow]],
        *,
        now: datetime | None = None,
    ) -> str:
        now = now or utcnow()
        return self._env.get_template("sweep.md.j2").render(
            events=[EventView.of(row, now, history.get(row.id, ())) for row in rows],
            now=now.isoformat(timespec="seconds"),
        )

    # -- the subagent's conversation -----------------------------------------
    #
    # Three pieces rather than one prompt, because the subagent's context is a
    # transcript that grows: the framing is the system prompt, stable across
    # every call so the cache can hold it, and each event and each brief is a
    # turn of its own.

    def subagent_system(self) -> str:
        return self._env.get_template("subagent_system.md.j2").render()

    def event(self, row: EventRow, *, now: datetime | None = None) -> str:
        """One event as a turn. The relative age freezes as of now, which is
        right for a transcript; the absolute timestamp beside it disambiguates."""
        now = now or utcnow()
        return self._env.get_template("subagent_event.md.j2").render(
            event=EventView.of(row, now)
        )

    def brief(self, instructions: str, *, sequence: bool) -> str:
        """Triage's instructions as a turn, following the events they cover."""
        return self._env.get_template("subagent_brief.md.j2").render(
            instructions=instructions, sequence=sequence
        )
