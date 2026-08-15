"""The event queue itself.

Owns its engine and session factory; callers hold a queue and pass it around
explicitly. Nothing here reads global state.

Every state change is paired with a log row in the same transaction, so an
event's status and the reasoning behind it can never disagree.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from types import TracebackType
from typing import Self

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker

from .agents import Agent
from .events import Event, Priority
from .store import (
    EventLogRow,
    EventRow,
    LogAction,
    Status,
    create_engine,
    init_schema,
    session_factory,
    utcnow,
)

#: An event's log entries, oldest first, keyed by event id.
type History = dict[int, list[EventLogRow]]

#: An event id paired with the text recorded against it.
type LogEntry = tuple[int, str]


class EventNotFound(KeyError):
    """No event with that id (or it was archived out from under you)."""


@dataclass(frozen=True, slots=True)
class TriageView:
    """What triage sees: pending events in full, the deferred backlog in outline."""

    pending: list[EventRow]
    deferred: list[EventRow]


@dataclass(frozen=True, slots=True)
class SweepView:
    """What the sweep sees: the deferred backlog in full, history and all."""

    rows: list[EventRow]
    history: History


class EventQueue:
    def __init__(self, engine: AsyncEngine, sessions: async_sessionmaker[AsyncSession]) -> None:
        self._engine = engine
        self._sessions = sessions

    @classmethod
    async def open(cls, db_path: Path | str) -> Self:
        engine = create_engine(db_path)
        await init_schema(engine)
        return cls(engine, session_factory(engine))

    async def aclose(self) -> None:
        await self._engine.dispose()

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        await self.aclose()

    async def submit(self, event: Event, priority: Priority = Priority.NORMAL) -> int:
        """Queue an event; returns the id used to refer to it from here on."""
        row = EventRow(
            kind=type(event).kind,
            timestamp=event.timestamp,
            description=event.description,
            payload=event.payload(),
            priority=priority,
            status=Status.PENDING,
        )
        async with self._sessions.begin() as session:
            session.add(row)
        return row.id

    async def get(self, event_id: int) -> EventRow:
        async with self._sessions() as session:
            (row,) = await self._require(session, [event_id])
            return row

    async def get_many(self, event_ids: Sequence[int]) -> list[EventRow]:
        """Fetch several events, in the order asked for."""
        async with self._sessions() as session:
            return await self._require(session, event_ids)

    async def edit(
        self,
        event_id: int,
        *,
        description: str | None = None,
        priority: Priority | None = None,
        timestamp: datetime | None = None,
    ) -> EventRow:
        """Revise an event in place. Unset arguments are left alone."""
        async with self._sessions.begin() as session:
            (row,) = await self._require(session, [event_id])
            if description is not None:
                row.description = description
            if priority is not None:
                row.priority = priority
            if timestamp is not None:
                row.timestamp = timestamp
            return row

    async def archive(self, event_id: int, *, agent: Agent, reason: str) -> EventRow:
        """Retire an event from view. This is as close to deletion as we get.

        Archived events appear only in the full history listing and the log,
        which is why the reason is required: it is the last word on the event.
        """
        async with self._sessions.begin() as session:
            (row,) = await self._require(session, [event_id])
            row.archived_at = utcnow()
            self._append(session, [(event_id, reason)], LogAction.ARCHIVE_EVENT, agent)
            return row

    # -- dispositions ------------------------------------------------------
    #
    # Plural throughout: triage acts on a set of events at a time, and one call
    # should land as one transaction rather than one per event.

    async def assign(
        self,
        event_ids: Sequence[int],
        *,
        agent: Agent,
        subagent: Agent,
        instructions: str,
        action: LogAction,
    ) -> None:
        """Record that `agent` handed these events to `subagent`.

        Written before the subagent runs, so a crash mid-handling still leaves
        evidence of what was attempted.
        """
        async with self._sessions.begin() as session:
            await self._require(session, event_ids)
            self._append(
                session,
                [(i, instructions) for i in event_ids],
                action,
                agent,
                assigned=subagent,
            )

    async def complete(self, event_ids: Sequence[int], *, agent: Agent, report: str) -> None:
        """Mark events handled, with the subagent's account of what it did."""
        async with self._sessions.begin() as session:
            for row in await self._require(session, event_ids):
                row.status = Status.COMPLETED
            self._append(
                session, [(i, report) for i in event_ids], LogAction.REPORT, agent
            )

    async def defer(self, deferrals: Sequence[LogEntry], *, agent: Agent) -> None:
        """Move events out of triage's working set and into the sweep's.

        Each event carries its own reason: the entry is read back as that
        event's history, so it has to make sense without the others.
        """
        async with self._sessions.begin() as session:
            for row in await self._require(session, [i for i, _ in deferrals]):
                row.status = Status.DEFERRED
            self._append(session, deferrals, LogAction.DEFER_EVENT, agent)

    async def escalate(
        self, event_id: int, *, agent: Agent, priority: Priority, reason: str
    ) -> None:
        """Return a deferred event to triage's working set, more urgent than before."""
        async with self._sessions.begin() as session:
            (row,) = await self._require(session, [event_id])
            row.status = Status.PENDING
            row.priority = priority
            self._append(session, [(event_id, reason)], LogAction.ESCALATE_EVENT, agent)

    async def set_digest(self, event_id: int, digest: str) -> None:
        """Cache the line that stands in for this event on the backlog list."""
        async with self._sessions.begin() as session:
            (row,) = await self._require(session, [event_id])
            row.digest = digest

    async def keep_deferred(self, event_id: int, *, agent: Agent, reason: str) -> None:
        """Leave a deferred event where it is, with a fresh look at why."""
        async with self._sessions.begin() as session:
            await self._require(session, [event_id])
            self._append(session, [(event_id, reason)], LogAction.KEEP_DEFERRED, agent)

    async def record_failure(
        self, event_ids: Sequence[int], *, agent: Agent, error: str
    ) -> None:
        """Note that handling fell over. The events keep their current status."""
        async with self._sessions.begin() as session:
            self._append(
                session, [(i, error) for i in event_ids], LogAction.FAILED, agent
            )

    # -- reading -----------------------------------------------------------

    async def list_events(
        self, *, active_only: bool = True, min_priority: Priority | None = None
    ) -> list[EventRow]:
        """Most urgent first, oldest first within a priority.

        `active_only` is the working queue: nothing completed, nothing archived.
        Turn it off for the full history — archived rows included, since that is
        the only view that shows them (`get` refuses them either way).
        """
        async with self._sessions() as session:
            query = self._events_query(active_only=active_only, min_priority=min_priority)
            return list((await session.scalars(query)).all())

    async def triage_view(self) -> TriageView:
        """Everything triage needs, in one query.

        Pending and deferred events come back together and are split here;
        deferred ones are shown to triage as an outline only, so no history is
        fetched — that lives with the sweep.
        """
        async with self._sessions() as session:
            rows = (await session.scalars(self._events_query())).all()
            by_status: dict[Status, list[EventRow]] = {}
            for row in rows:
                by_status.setdefault(row.status, []).append(row)
            return TriageView(
                pending=by_status.get(Status.PENDING, []),
                deferred=by_status.get(Status.DEFERRED, []),
            )

    async def sweep_view(self) -> SweepView:
        """The deferred backlog with its full history.

        Two queries regardless of backlog size: one for the events, one for the
        whole slice of log they point at.
        """
        async with self._sessions() as session:
            query = self._events_query().where(EventRow.status == Status.DEFERRED)
            rows = list((await session.scalars(query)).all())
            return SweepView(rows, await self._history(session, [r.id for r in rows]))

    async def history_for(self, event_ids: Sequence[int]) -> History:
        """Log entries for these events, oldest first. One query."""
        async with self._sessions() as session:
            return await self._history(session, event_ids)

    # -- internals ---------------------------------------------------------

    @staticmethod
    def _events_query(
        *, active_only: bool = True, min_priority: Priority | None = None
    ):
        query = select(EventRow)
        if active_only:
            query = query.where(
                EventRow.archived_at.is_(None), EventRow.status != Status.COMPLETED
            )
        if min_priority is not None:
            query = query.where(EventRow.priority >= min_priority)
        return query.order_by(EventRow.priority.desc(), EventRow.timestamp.asc())

    @staticmethod
    def _append(
        session: AsyncSession,
        entries: Sequence[LogEntry],
        action: LogAction,
        agent: Agent,
        *,
        assigned: Agent | None = None,
    ) -> None:
        session.add_all(
            EventLogRow(
                event_id=event_id,
                action=action,
                detail=detail,
                agent_id=agent.id,
                agent_role=agent.role,
                assigned_agent_id=assigned.id if assigned else None,
            )
            for event_id, detail in entries
        )

    @staticmethod
    async def _history(session: AsyncSession, event_ids: Sequence[int]) -> History:
        if not event_ids:
            return {}
        query = (
            select(EventLogRow)
            .where(EventLogRow.event_id.in_(event_ids))
            .order_by(EventLogRow.timestamp.asc(), EventLogRow.id.asc())
        )
        grouped: History = {}
        for entry in (await session.scalars(query)).all():
            grouped.setdefault(entry.event_id, []).append(entry)
        return grouped

    @staticmethod
    async def _require(session: AsyncSession, event_ids: Sequence[int]) -> list[EventRow]:
        """The named events, in the order asked for. Raises if any is missing."""
        query = select(EventRow).where(
            EventRow.id.in_(event_ids), EventRow.archived_at.is_(None)
        )
        found = {row.id: row for row in (await session.scalars(query)).all()}
        if missing := [i for i in event_ids if i not in found]:
            raise EventNotFound(f"no such event(s): {missing}")
        return [found[i] for i in event_ids]
