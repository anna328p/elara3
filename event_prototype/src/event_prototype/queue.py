"""The event queue itself.

Owns its engine and session factory; callers hold a queue and pass it around
explicitly. Nothing here reads global state.

Every state change is paired with an action row in the same transaction, so an
event's status and the reasoning behind it can never disagree.
"""

from __future__ import annotations

from collections.abc import AsyncGenerator, Sequence
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from types import TracebackType
from typing import Self

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker
from sqlalchemy.orm import aliased, contains_eager

from .agents import Agent, AgentRole
from .contexts import Turn
from .events import Event, MessageEvent, Priority, check_priority
from .heartbeats import Beat, CheckIn, HeartbeatSummary
from .memory import MemoryStore
from .people import PeopleStore, Profile, profiles_for
from .render import PromptRenderer
from .store import (
    Action,
    ActionRow,
    ContextRow,
    EventRow,
    HeartbeatRow,
    HeartbeatTickRow,
    Status,
    StreamNotFound,
    StreamRow,
    TurnRow,
    create_engine,
    init_schema,
    join_context,
    resolve_context,
    resolve_stream,
    route,
    session_factory,
    spawn_agent,
    utcnow,
)
from .wake import Arrival, Arrived, Notifier, Subscription

#: An event's actions, oldest first, keyed by event id.
type History = dict[int, list[ActionRow]]

#: An event id paired with the text recorded against it.
type EventDetail = tuple[int, str]


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


@dataclass(frozen=True, slots=True)
class StreamSummary:
    """A stream and how much is going on in it."""

    stream: StreamRow
    active: int
    last_event_at: datetime | None
    #: How long the conversation in its context has grown. Zero without one.
    turns: int


@dataclass(frozen=True, slots=True)
class Assignment:
    """What an assignment resolved to: the rows, who got them, and where."""

    rows: list[EventRow]
    subagent: Agent
    #: The context the work went to; None means a one-shot subagent.
    context: ContextRow | None
    #: The profile behind each event's sender, for the events whose sender is
    #: someone known. Resolved with the assignment, so the subagent is shown
    #: who it is talking to without a lookup of its own.
    people: dict[int, Profile] = field(default_factory=lambda: {})


def _earliest(known: datetime | None, candidate: datetime) -> datetime:
    return candidate if known is None or candidate < known else known


class EventQueue:
    def __init__(
        self,
        engine: AsyncEngine,
        sessions: async_sessionmaker[AsyncSession],
        renderer: PromptRenderer | None = None,
    ) -> None:
        self._engine = engine
        self._sessions = sessions
        self._notifier = Notifier()
        #: The character's memory and the people in it, over the same sessions.
        #: Held here so that whoever has the queue has the whole store, and
        #: nothing reaches for an engine of its own.
        self.memory = MemoryStore(sessions)
        self.people = PeopleStore(sessions, renderer or PromptRenderer())

    @classmethod
    async def open(cls, db_path: Path | str, renderer: PromptRenderer | None = None) -> Self:
        engine = create_engine(db_path)
        await init_schema(engine)
        return cls(engine, session_factory(engine), renderer)

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

    async def spawn(self, role: AgentRole) -> Agent:
        """Mint an agent. Everything attributed to one starts here."""
        async with self._sessions.begin() as session:
            return (await spawn_agent(session, role)).to_agent()

    async def submit(self, event: Event, priority: Priority = Priority.BACKGROUND) -> int:
        """Queue an event; returns the id used to refer to it from here on.

        The priority is the submitter's routing decision (see `Priority`), and
        one the event's stream cannot carry is refused before anything is
        written. The default is the one that commits to nothing: triage, at
        its next heartbeat.

        If the event names a stream, it is opened on first sight and joined
        thereafter, in the same transaction as the event itself — so an event
        can never end up pointing at a stream that was rolled back.

        The id is returned rather than the row on purpose: a row that was just
        added has no loaded `stream`, and reaching for one after the session
        closes raises. Callers that want the stream go through `get`.
        """
        ref = event.stream
        check_priority(priority, ref.kind if ref else None)
        async with self._sessions.begin() as session:
            row = EventRow(
                kind=type(event).kind,
                timestamp=event.timestamp,
                description=event.description,
                payload=event.payload(),
                priority=priority,
                status=Status.PENDING,
                stream_id=await resolve_stream(session, ref) if ref else None,
            )
            session.add(row)
        self._notifier.wake(Arrived(row.id, row.priority, Arrival.SUBMITTED, utcnow()))
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
                check_priority(priority, row.stream.kind if row.stream else None)
                row.priority = priority
            if timestamp is not None:
                row.timestamp = timestamp
            return row

    async def archive(self, event_id: int, *, agent: Agent, reason: str) -> EventRow:
        """Retire an event from view. This is as close to deletion as we get.

        Archived events appear only in the full history listing and `event_actions`,
        which is why the reason is required: it is the last word on the event.
        """
        async with self._sessions.begin() as session:
            (row,) = await self._require(session, [event_id])
            row.archived_at = utcnow()
            self._append(session, [(event_id, reason)], Action.ARCHIVE_EVENT, agent)
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
        instructions: str,
        action: Action,
    ) -> Assignment:
        """Route these events, mint a subagent if none is theirs, and record the handoff.

        One transaction: a context opened here exists only alongside the action
        rows that point at its agent, and a crash mid-handling still leaves
        evidence of what was attempted.

        The senders are looked up here too, in one query for the whole batch,
        so the events reach the subagent with the people behind them.
        """
        async with self._sessions.begin() as session:
            rows = await self._require(session, event_ids)
            context = await route(session, event_ids)
            actor = context.actor if context else await spawn_agent(session, AgentRole.SUBAGENT)
            subagent = actor.to_agent()
            self._append(
                session,
                [(i, instructions) for i in event_ids],
                action,
                agent,
                assigned=subagent,
            )
            senders = {
                row.id: (event.venue, event.sender)
                for row in rows
                if isinstance(event := row.to_event(), MessageEvent)
            }
            profiles = await profiles_for(session, set(senders.values()))
            people = {
                event_id: profiles[handle]
                for event_id, handle in senders.items()
                if handle in profiles
            }
            return Assignment(rows, subagent, context, people)

    async def complete(self, event_ids: Sequence[int], *, agent: Agent, report: str) -> None:
        """Mark events handled, with the subagent's account of what it did."""
        async with self._sessions.begin() as session:
            for row in await self._require(session, event_ids):
                row.status = Status.COMPLETED
            self._append(
                session, [(i, report) for i in event_ids], Action.REPORT, agent
            )

    async def defer(self, deferrals: Sequence[EventDetail], *, agent: Agent) -> None:
        """Move events out of triage's working set and into the sweep's.

        Each event carries its own reason: the entry is read back as that
        event's history, so it has to make sense without the others.
        """
        async with self._sessions.begin() as session:
            for row in await self._require(session, [i for i, _ in deferrals]):
                row.status = Status.DEFERRED
            self._append(session, deferrals, Action.DEFER_EVENT, agent)

    async def escalate(
        self, event_id: int, *, agent: Agent, priority: Priority, reason: str
    ) -> None:
        """Return a deferred event to the pending set, at the priority it should have had."""
        async with self._sessions.begin() as session:
            (row,) = await self._require(session, [event_id])
            check_priority(priority, row.stream.kind if row.stream else None)
            row.status = Status.PENDING
            row.priority = priority
            self._append(session, [(event_id, reason)], Action.ESCALATE_EVENT, agent)
        self._notifier.wake(Arrived(event_id, priority, Arrival.ESCALATED, utcnow()))

    async def set_digest(self, event_id: int, digest: str) -> None:
        """Cache the line that stands in for this event on the backlog list."""
        async with self._sessions.begin() as session:
            (row,) = await self._require(session, [event_id])
            row.digest = digest

    async def keep_deferred(self, event_id: int, *, agent: Agent, reason: str) -> None:
        """Leave a deferred event where it is, with a fresh look at why."""
        async with self._sessions.begin() as session:
            await self._require(session, [event_id])
            self._append(session, [(event_id, reason)], Action.KEEP_DEFERRED, agent)

    async def record_failure(
        self, event_ids: Sequence[int], *, agent: Agent, error: str
    ) -> None:
        """Note that handling fell over. The events keep their current status."""
        async with self._sessions.begin() as session:
            self._append(
                session, [(i, error) for i in event_ids], Action.FAILED, agent
            )

    # -- reading -----------------------------------------------------------

    async def list_events(self, *, active_only: bool = True) -> list[EventRow]:
        """Ordered as `Priority` orders, oldest first within a priority.

        `active_only` is the working queue: nothing completed, nothing archived.
        Turn it off for the full history — archived rows included, since that is
        the only view that shows them (`get` refuses them either way).
        """
        async with self._sessions() as session:
            query = self._events_query(active_only=active_only)
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
        whole slice of `event_actions` they point at.
        """
        async with self._sessions() as session:
            query = self._events_query().where(EventRow.status == Status.DEFERRED)
            rows = list((await session.scalars(query)).all())
            return SweepView(rows, await self._history(session, [r.id for r in rows]))

    async def list_streams(self) -> list[StreamSummary]:
        """Every stream with its active event count and last activity, busiest first.

        One query, and the aggregate is why `streams` caches neither number:
        both fall out of the same GROUP BY that has to happen anyway.
        """
        active = func.count(EventRow.id).filter(
            EventRow.archived_at.is_(None), EventRow.status != Status.COMPLETED
        )
        last_event_at = func.max(EventRow.timestamp)
        # Counted apart from the events join, or the two would multiply.
        turns = (
            select(TurnRow.context_id, func.count(TurnRow.id).label("turns"))
            .group_by(TurnRow.context_id)
            .subquery()
        )
        turn_count = func.coalesce(turns.c.turns, 0)
        query = (
            select(StreamRow, active, last_event_at, turn_count)
            .outerjoin(EventRow, EventRow.stream_id == StreamRow.id)
            .outerjoin(turns, turns.c.context_id == StreamRow.context_id)
            .group_by(StreamRow.id, turns.c.turns)
            .order_by(last_event_at.desc())
        )
        async with self._sessions() as session:
            rows = (await session.execute(query)).all()
        return [
            StreamSummary(stream, count, last, turns)
            for stream, count, last, turns in rows
        ]

    async def history_for(self, event_ids: Sequence[int]) -> History:
        """The actions taken on these events, oldest first. One query."""
        async with self._sessions() as session:
            return await self._history(session, event_ids)

    # -- contexts ----------------------------------------------------------

    async def get_stream(self, stream_id: int) -> StreamRow:
        async with self._sessions() as session:
            stream = await session.get(StreamRow, stream_id)
            if stream is None:
                raise StreamNotFound(f"no such stream: {stream_id}")
            return stream

    async def context_for(self, stream_id: int) -> ContextRow:
        """The context a stream's work goes to, opened if this is its first."""
        async with self._sessions.begin() as session:
            return await resolve_context(session, stream_id)

    async def join_context(self, stream_id: int, context_id: int) -> None:
        """Point a stream at an open context, so its work joins that conversation."""
        async with self._sessions.begin() as session:
            await join_context(session, stream_id, context_id)

    async def append_turns(self, context_id: int, turns: Sequence[Turn]) -> None:
        """Add turns to a context, in order, as one transaction."""
        async with self._sessions.begin() as session:
            context = await session.get(ContextRow, context_id)
            if context is None:
                raise KeyError(f"no such context: {context_id}")
            context.updated_at = utcnow()
            session.add_all(TurnRow.of(context_id, turn) for turn in turns)

    async def transcript(self, context_id: int) -> list[TurnRow]:
        """Every turn in a context, oldest first. One query."""
        query = (
            select(TurnRow)
            .where(TurnRow.context_id == context_id)
            .order_by(TurnRow.id.asc())
        )
        async with self._sessions() as session:
            return list((await session.scalars(query)).all())

    # -- waking ------------------------------------------------------------
    #
    # Wakes fire after the transaction block has exited, never inside it, so a
    # write that rolls back wakes nobody. Only `submit` and `escalate` wake:
    # they are the two ways an event enters the pending set. The heartbeat
    # writes below wake nobody either: a schedule is only ever written from
    # inside a pass, and the loop re-reads the schedule when the pass ends.

    @asynccontextmanager
    async def subscribe(self, label: str) -> AsyncGenerator[Subscription]:
        """Be woken whenever an event enters the pending set. Detaches on exit."""
        subscription = self._notifier.add(label)
        try:
            yield subscription
        finally:
            self._notifier.remove(subscription)

    def subscriptions(self) -> list[str]:
        """Who is listening, by label."""
        return self._notifier.labels()

    async def workload(self, role: AgentRole) -> int:
        """How many events a pass for `role` would see: pending for triage,
        deferred for the sweep. One query."""
        match role:
            case AgentRole.TRIAGE:
                status = Status.PENDING
            case AgentRole.SWEEP:
                status = Status.DEFERRED
            case _:
                raise ValueError(f"{role.value} runs no pass, so has no workload")
        query = select(func.count(EventRow.id)).where(
            EventRow.status == status, EventRow.archived_at.is_(None)
        )
        async with self._sessions() as session:
            return (await session.execute(query)).scalar_one()

    # -- heartbeats --------------------------------------------------------
    #
    # A schedule is a `HeartbeatRow`; when it next fires is its one unfired
    # `HeartbeatTickRow`. Every write here is one transaction, and every read
    # one statement.

    async def schedule_heartbeat(
        self,
        role: AgentRole,
        *,
        due_at: datetime,
        interval_seconds: float | None = None,
        message: str | None = None,
        expires_at: datetime | None = None,
        created_by: Agent | None = None,
    ) -> HeartbeatSummary:
        """Add a schedule for `role` and its first tick.

        `interval_seconds` None is a one-shot check-in, fired once when due.
        A recurring schedule is re-timed from every pass of its role (see
        `beat`), so `due_at` here is only the first tick.
        """
        async with self._sessions.begin() as session:
            schedule = HeartbeatRow(
                role=role,
                interval_seconds=interval_seconds,
                message=message,
                expires_at=expires_at,
                created_by=created_by.id if created_by else None,
            )
            session.add(schedule)
            await session.flush()  # for its id
            session.add(HeartbeatTickRow(heartbeat_id=schedule.id, due_at=due_at))
            await session.flush()
            return await self._summary(session, schedule.id)

    async def ensure_standing(self, role: AgentRole, interval_seconds: float) -> HeartbeatSummary:
        """The standing schedule for `role` — recurring, open-ended, silent,
        nobody's — at `interval_seconds`, made if missing.

        A live standing schedule at a different interval is ended (its tick
        fired now, no successor) and a fresh one inserted due now, so a config
        change shows up as two rows rather than an edited one. Overrides and
        check-ins are not touched.
        """
        now = utcnow()
        async with self._sessions.begin() as session:
            kept: HeartbeatRow | None = None
            for tick in (await session.scalars(self._live_ticks(role))).all():
                schedule = tick.schedule
                if not schedule.standing:
                    continue
                if kept is None and schedule.interval_seconds == interval_seconds:
                    kept = schedule
                else:
                    tick.fired_at = now
            if kept is None:
                kept = HeartbeatRow(role=role, interval_seconds=interval_seconds)
                session.add(kept)
                await session.flush()  # for its id, and so the ended ticks
                # are stamped before a live one is added beside them
                session.add(HeartbeatTickRow(heartbeat_id=kept.id, due_at=now))
            await session.flush()
            return await self._summary(session, kept.id)

    async def beat(self, role: AgentRole, *, at: datetime) -> Beat:
        """A pass for `role` is happening at `at`: fire what is due, re-time what recurs.

        Every unfired tick of the role's schedules with `due_at <= at` fires,
        and its message goes into the returned `Beat`. Every live recurring
        schedule, due or not, fires its tick and gets a successor at
        `at + interval` — or none if that would pass `expires_at`, which is
        how an override ends. A check-in not yet due is left for its time.

        Called before the pass, not after, so the successor is timed from
        when the pass began: a pass longer than its interval is followed at
        once by another.
        """
        check_ins: list[CheckIn] = []
        successors: list[HeartbeatTickRow] = []
        due = False
        next_due: datetime | None = None
        async with self._sessions.begin() as session:
            for tick in (await session.scalars(self._live_ticks(role))).all():
                schedule = tick.schedule
                is_due = tick.due_at <= at
                if not is_due and not schedule.recurring:
                    next_due = _earliest(next_due, tick.due_at)
                    continue
                tick.fired_at = at
                if is_due:
                    due = True
                    if schedule.message is not None:
                        check_ins.append(
                            CheckIn(schedule.message, schedule.agent, schedule.created_at, tick.due_at)
                        )
                if schedule.interval_seconds is not None:
                    successor = at + timedelta(seconds=schedule.interval_seconds)
                    if schedule.expires_at is None or successor <= schedule.expires_at:
                        successors.append(HeartbeatTickRow(heartbeat_id=schedule.id, due_at=successor))
                        next_due = _earliest(next_due, successor)
            # The stamps go out before the successors: the unit of work would
            # otherwise insert first, and the live-tick index refuses two.
            await session.flush()
            session.add_all(successors)
        return Beat(role, at, tuple(check_ins), due, next_due)

    async def next_heartbeat_due(self, roles: Sequence[AgentRole]) -> dict[AgentRole, datetime]:
        """When each of `roles` is next due, for those with anything live. One query.

        Per role rather than one minimum, so the loop can both sleep until
        the earliest and, on waking, tell which roles it was for.
        """
        query = (
            select(HeartbeatRow.role, func.min(HeartbeatTickRow.due_at))
            .join(HeartbeatRow, HeartbeatRow.id == HeartbeatTickRow.heartbeat_id)
            .where(HeartbeatTickRow.fired_at.is_(None), HeartbeatRow.role.in_(roles))
            .group_by(HeartbeatRow.role)
        )
        async with self._sessions() as session:
            return {role: due for role, due in (await session.execute(query)).tuples()}

    async def heartbeats(self) -> list[HeartbeatSummary]:
        """Every live schedule with its next due, soonest first. One query."""
        async with self._sessions() as session:
            rows = (await session.execute(self._summaries_query())).all()
        return [
            HeartbeatSummary(schedule, next_due, fired, last_fired_at)
            for schedule, next_due, fired, last_fired_at in rows
        ]

    # -- internals ---------------------------------------------------------

    @staticmethod
    def _live_ticks(role: AgentRole):
        """The role's unfired ticks with their schedules, soonest first."""
        return (
            select(HeartbeatTickRow)
            .join(HeartbeatTickRow.schedule)
            .options(contains_eager(HeartbeatTickRow.schedule))
            .where(HeartbeatTickRow.fired_at.is_(None), HeartbeatRow.role == role)
            .order_by(HeartbeatTickRow.due_at.asc(), HeartbeatTickRow.id.asc())
        )

    @staticmethod
    def _summaries_query():
        """Live schedules with their next due and firing history, soonest first.

        The count is aggregated apart from the join to the live tick, as the
        turn count is in `list_streams`, so the eager creator columns need no
        GROUP BY and the join cannot multiply.
        """
        fired = (
            select(
                HeartbeatTickRow.heartbeat_id,
                func.count(HeartbeatTickRow.id).label("fired"),
                func.max(HeartbeatTickRow.fired_at).label("last_fired_at"),
            )
            .where(HeartbeatTickRow.fired_at.is_not(None))
            .group_by(HeartbeatTickRow.heartbeat_id)
            .subquery()
        )
        live = aliased(HeartbeatTickRow)
        return (
            select(HeartbeatRow, live.due_at, func.coalesce(fired.c.fired, 0), fired.c.last_fired_at)
            .join(live, (live.heartbeat_id == HeartbeatRow.id) & live.fired_at.is_(None))
            .outerjoin(fired, fired.c.heartbeat_id == HeartbeatRow.id)
            .order_by(live.due_at.asc(), HeartbeatRow.id.asc())
        )

    async def _summary(self, session: AsyncSession, schedule_id: int) -> HeartbeatSummary:
        """One live schedule's summary, from within the caller's transaction."""
        query = self._summaries_query().where(HeartbeatRow.id == schedule_id)
        schedule, next_due, fired, last_fired_at = (await session.execute(query)).one()
        return HeartbeatSummary(schedule, next_due, fired, last_fired_at)

    @staticmethod
    def _events_query(*, active_only: bool = True):
        query = select(EventRow)
        if active_only:
            query = query.where(
                EventRow.archived_at.is_(None), EventRow.status != Status.COMPLETED
            )
        return query.order_by(EventRow.priority.desc(), EventRow.timestamp.asc())

    @staticmethod
    def _append(
        session: AsyncSession,
        entries: Sequence[EventDetail],
        action: Action,
        agent: Agent,
        *,
        assigned: Agent | None = None,
    ) -> None:
        session.add_all(
            ActionRow(
                event_id=event_id,
                action=action,
                detail=detail,
                agent_id=agent.id,
                assigned_agent_id=assigned.id if assigned else None,
            )
            for event_id, detail in entries
        )

    @staticmethod
    async def _history(session: AsyncSession, event_ids: Sequence[int]) -> History:
        if not event_ids:
            return {}
        query = (
            select(ActionRow)
            .where(ActionRow.event_id.in_(event_ids))
            .order_by(ActionRow.timestamp.asc(), ActionRow.id.asc())
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
