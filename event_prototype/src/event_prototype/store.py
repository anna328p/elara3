"""SQLite persistence for the event queue.

`events` is current state and `event_actions` the append-only record of what was
decided about each one and by whom — "whom" being a row in `agents`, which every
attribution points at. `streams` is the ongoing loci events belong to, each
routed to at most one `contexts` row: the conversation an agent is having there,
stored as `turns`. `memory_entries` are the pages of the character's memory and
`memory_versions` everything that has ever been true of them; `known_people` and
`person_names` say whose profile a page is and which handles are theirs.
`heartbeats` is when each loop-driven role runs a pass, and `heartbeat_ticks`
the append-only record of when each schedule was next due and when it fired.
Rows are never deleted — archiving stamps `archived_at`, deleting a page appends
a tombstone, and every default query filters those out, so the trail stays
intact.
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import Any
from uuid import uuid4

from sqlalchemy import (
    JSON,
    DateTime,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    TypeDecorator,
    UniqueConstraint,
    event,
    select,
    text,
)
from sqlalchemy import Enum as SAEnum
from sqlalchemy.dialects.sqlite import insert as sqlite_insert
from sqlalchemy.engine.interfaces import DBAPIConnection
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship
from sqlalchemy.pool import ConnectionPoolEntry, StaticPool

from .agents import Agent, AgentRole
from .contexts import Block, Role, Turn
from .events import BaseEvent, Priority, from_payload
from .streams import StreamKind, StreamRef

IN_MEMORY = ":memory:"


class Status(StrEnum):
    PENDING = "pending"
    DEFERRED = "deferred"
    COMPLETED = "completed"


class Action(StrEnum):
    """What an agent did about an event. Most correspond one-to-one with model tools."""

    # triage
    HANDLE_ONE_EVENT = "handle_one_event"
    HANDLE_EVENT_SEQUENCE = "handle_event_sequence"
    DEFER_EVENT = "defer_event"
    # sweep
    ESCALATE_EVENT = "escalate_event"
    ARCHIVE_EVENT = "archive_event"
    KEEP_DEFERRED = "keep_deferred"
    # subagents
    REPORT = "report"
    FAILED = "failed"


class PriorityType(TypeDecorator[Priority]):
    """Store `Priority` as its integer value so SQL can order by urgency."""

    impl = Integer
    cache_ok = True

    def process_bind_param(self, value: Priority | None, dialect: object) -> int | None:
        return None if value is None else int(value)

    def process_result_value(self, value: int | None, dialect: object) -> Priority | None:
        return None if value is None else Priority(value)


class UTCDateTime(TypeDecorator[datetime]):
    """Keep every datetime timezone-aware in UTC.

    SQLite's datetime storage has no room for an offset, so without this the
    values silently come back naive and comparing them raises.
    """

    impl = DateTime
    cache_ok = True

    def process_bind_param(self, value: datetime | None, dialect: object) -> datetime | None:
        if value is None:
            return None
        aware = value.replace(tzinfo=UTC) if value.tzinfo is None else value
        return aware.astimezone(UTC).replace(tzinfo=None)

    def process_result_value(self, value: datetime | None, dialect: object) -> datetime | None:
        return None if value is None else value.replace(tzinfo=UTC)


def utcnow() -> datetime:
    return datetime.now(UTC)


class Base(DeclarativeBase):
    # Every datetime in this schema is UTC-aware on the Python side.
    type_annotation_map = {datetime: UTCDateTime}


class StreamRow(Base):
    """An ongoing context events belong to: a room, a correspondent, a job.

    Identity is `(kind, key)`; the surrogate id is what `events.stream_id`
    points at. There is deliberately no cached "last event" column — that is
    `max(events.timestamp)`, and a cached copy would only be another thing to
    keep true.
    """

    __tablename__ = "streams"
    __table_args__ = (UniqueConstraint("kind", "key", name="uq_streams_kind_key"),)

    id: Mapped[int] = mapped_column(primary_key=True)

    kind: Mapped[StreamKind] = mapped_column(
        SAEnum(StreamKind, native_enum=False, length=16)
    )
    key: Mapped[str] = mapped_column(String(255))
    title: Mapped[str]

    #: Where work in this stream goes: the one live context, opened on the
    #: first assignment and kept thereafter. Several streams may point at the
    #: same context; re-pointing is how a stream moves to a fresh one. A plain
    #: column rather than a relationship, so it rides along on the join
    #: `EventRow.stream` already makes without fanning that query out further.
    context_id: Mapped[int | None] = mapped_column(
        ForeignKey("contexts.id"), default=None
    )

    created_at: Mapped[datetime] = mapped_column(default=utcnow)
    #: Touched every time an event joins, so this doubles as "last seen".
    updated_at: Mapped[datetime] = mapped_column(default=utcnow, onupdate=utcnow)

    def __repr__(self) -> str:
        return f"StreamRow(id={self.id}, kind={self.kind.value}, key={self.key!r})"


class AgentRow(Base):
    """An agent: the thing actions are attributed to and contexts belong to.

    Nothing more than an identity and a role for now. In the real framework
    this row grows an event loop, a name, a budget; here it exists so that
    every attribution is a foreign key to something, and the role is written
    once rather than alongside each reference.
    """

    __tablename__ = "agents"

    #: A UUID rather than a serial: agents are minted from many places at once
    #: and referred to across processes, so the id should not depend on order.
    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid4()))
    role: Mapped[AgentRole] = mapped_column(
        SAEnum(AgentRole, native_enum=False, length=16)
    )
    created_at: Mapped[datetime] = mapped_column(default=utcnow)

    def to_agent(self) -> Agent:
        return Agent(self.role, self.id)

    def __repr__(self) -> str:
        return f"AgentRow({self.to_agent().label!r})"


class ContextRow(Base):
    """A conversation an agent is having, replayable as one Messages API call."""

    __tablename__ = "contexts"

    id: Mapped[int] = mapped_column(primary_key=True)

    #: Whose conversation this is. Minted when the context opens and kept, so
    #: returning work in a stream reaches the same subagent — the identity in
    #: `event_actions` and the memory in the transcript are the same thing.
    agent_id: Mapped[str] = mapped_column(ForeignKey("agents.id"))
    #: Joined, for the same reason as `EventRow.stream`: rows outlive their
    #: session, and many-to-one costs nothing extra in the same SELECT.
    actor: Mapped[AgentRow] = relationship(lazy="joined")

    created_at: Mapped[datetime] = mapped_column(default=utcnow)
    #: Touched whenever a turn is appended.
    updated_at: Mapped[datetime] = mapped_column(default=utcnow, onupdate=utcnow)

    @property
    def agent(self) -> Agent:
        return self.actor.to_agent()

    def __repr__(self) -> str:
        return f"ContextRow(id={self.id}, agent={self.agent.label!r})"


class TurnRow(Base):
    """One message in a context. Append-only, ordered by id."""

    __tablename__ = "turns"
    __table_args__ = (
        # The one read is "this context's transcript, in order".
        Index("ix_turns_context", "context_id", "id"),
    )

    id: Mapped[int] = mapped_column(primary_key=True)
    context_id: Mapped[int] = mapped_column(ForeignKey("contexts.id"))
    timestamp: Mapped[datetime] = mapped_column(default=utcnow)

    role: Mapped[Role] = mapped_column(SAEnum(Role, native_enum=False, length=16))
    #: Anthropic content blocks, verbatim. Always a list, never a bare string,
    #: so there is one shape to read back.
    content: Mapped[list[Block]] = mapped_column(JSON, default=list)

    #: The event that occasioned this turn, when one did.
    event_id: Mapped[int | None] = mapped_column(
        ForeignKey("events.id"), index=True, default=None
    )

    @classmethod
    def of(cls, context_id: int, turn: Turn) -> TurnRow:
        return cls(
            context_id=context_id,
            role=turn.role,
            content=list(turn.content),
            event_id=turn.event_id,
        )

    def to_turn(self) -> Turn:
        return Turn(self.role, tuple(self.content), self.event_id)

    def __repr__(self) -> str:
        return f"TurnRow(id={self.id}, context_id={self.context_id}, role={self.role.value})"


class EventRow(Base):
    __tablename__ = "events"

    id: Mapped[int] = mapped_column(primary_key=True)

    kind: Mapped[str] = mapped_column(String(64))
    timestamp: Mapped[datetime]
    description: Mapped[str]
    payload: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)

    stream_id: Mapped[int | None] = mapped_column(
        ForeignKey("streams.id"), index=True, default=None
    )
    #: Eager by join rather than by choice at each call site. Rows outlive the
    #: session that produced them (`expire_on_commit=False`), so a lazy load
    #: would raise wherever a prompt is rendered; and being many-to-one, this
    #: rides along in the same SELECT, leaving every query count unchanged.
    stream: Mapped[StreamRow | None] = relationship(lazy="joined")

    priority: Mapped[Priority] = mapped_column(PriorityType)
    status: Mapped[Status] = mapped_column(
        SAEnum(Status, native_enum=False, length=16), default=Status.PENDING
    )

    # Why an event is in the state it is lives in `event_actions`, not here — the
    # row carries current state, the actions carry the reasoning behind it.

    #: One line standing in for this event on triage's backlog list, written
    #: when it is set aside and rewritten whenever that reasoning changes.
    #: Derived, not authoritative: absent until generated, and `description`
    #: serves in its place.
    digest: Mapped[str | None] = mapped_column(default=None)

    archived_at: Mapped[datetime | None] = mapped_column(default=None)
    created_at: Mapped[datetime] = mapped_column(default=utcnow)
    updated_at: Mapped[datetime] = mapped_column(default=utcnow, onupdate=utcnow)

    def to_event(self) -> BaseEvent:
        """Reconstruct the typed event this row was built from."""
        return from_payload(self.kind, self.timestamp, self.description, self.payload)

    def __repr__(self) -> str:
        return (
            f"EventRow(id={self.id}, kind={self.kind!r}, "
            f"priority={self.priority.name}, status={self.status.value})"
        )


class ActionRow(Base):
    """One thing an agent did about one event. Append-only."""

    __tablename__ = "event_actions"
    __table_args__ = (
        # Every read is "the history of these events, oldest first".
        Index("ix_event_actions_event_timestamp", "event_id", "timestamp"),
        # ...and occasionally "everything this agent touched".
        Index("ix_event_actions_agent", "agent_id"),
    )

    id: Mapped[int] = mapped_column(primary_key=True)
    event_id: Mapped[int] = mapped_column(ForeignKey("events.id"))
    timestamp: Mapped[datetime] = mapped_column(default=utcnow)

    action: Mapped[Action] = mapped_column(
        SAEnum(Action, native_enum=False, length=32)
    )
    #: The triage model's reason or instructions, or the subagent's report.
    detail: Mapped[str]

    #: Who did it.
    agent_id: Mapped[str] = mapped_column(ForeignKey("agents.id"))
    #: On an assignment, the subagent the work went to.
    assigned_agent_id: Mapped[str | None] = mapped_column(
        ForeignKey("agents.id"), default=None
    )
    # Both joined: actions are read to be rendered, after their session is gone,
    # and the role that `label` needs lives on the agent row now. Two foreign
    # keys into one table, so each relationship has to say which is its own.
    actor: Mapped[AgentRow] = relationship(lazy="joined", foreign_keys=[agent_id])
    assignee: Mapped[AgentRow | None] = relationship(
        lazy="joined", foreign_keys=[assigned_agent_id]
    )

    @property
    def agent(self) -> Agent:
        return self.actor.to_agent()

    @property
    def assigned(self) -> Agent | None:
        return self.assignee.to_agent() if self.assignee else None

    def __repr__(self) -> str:
        return (
            f"ActionRow(event_id={self.event_id}, action={self.action.value}, "
            f"agent={self.agent.label!r})"
        )


class MemoryEntryRow(Base):
    """A page's identity: its path, and when it came to be.

    Everything else about a page — its text, whether it exists right now, who
    last touched it — lives in its versions. The entry is what those hang off,
    so a page deleted and written again is one entry with one history.
    """

    __tablename__ = "memory_entries"

    id: Mapped[int] = mapped_column(primary_key=True)
    #: Absolute, under `/memories`, normalized (`memory.normalize`). Unique
    #: outright: a tombstoned page keeps its entry, and writing the path again
    #: continues that entry rather than opening a second one.
    path: Mapped[str] = mapped_column(String(512), unique=True)
    created_at: Mapped[datetime] = mapped_column(default=utcnow)
    #: Touched by rename, the one thing that changes on the entry itself.
    updated_at: Mapped[datetime] = mapped_column(default=utcnow, onupdate=utcnow)

    def __repr__(self) -> str:
        return f"MemoryEntryRow(id={self.id}, path={self.path!r})"


class MemoryVersionRow(Base):
    """A page as of one operation. Append-only; the newest one is the page."""

    __tablename__ = "memory_versions"
    __table_args__ = (
        # The head is the greatest id per entry, and a history is one entry's
        # versions in order. Global order by id is the rowid, indexed already.
        Index("ix_memory_versions_entry_id", "entry_id", "id"),
    )

    id: Mapped[int] = mapped_column(primary_key=True)
    entry_id: Mapped[int] = mapped_column(ForeignKey("memory_entries.id"))
    timestamp: Mapped[datetime] = mapped_column(default=utcnow)

    #: Who made the edit. NULL is the operator — an edit from outside the
    #: agents, at the CLI today — and anything else names an agent the store
    #: minted, which the foreign key enforces.
    agent_id: Mapped[str | None] = mapped_column(ForeignKey("agents.id"), default=None)
    actor: Mapped[AgentRow | None] = relationship(lazy="joined")

    #: The whole page as of this version. NULL is a tombstone: the page does
    #: not exist as of here, until a later version gives it a body again.
    body: Mapped[str | None] = mapped_column(Text, default=None)
    #: The memory tool command that produced this version, verbatim minus the
    #: path — `{"command": "str_replace", "old_str": …, "new_str": …}` — so the
    #: history says how each version came to differ from the one before.
    edit_metadata: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)

    @property
    def agent(self) -> Agent | None:
        return self.actor.to_agent() if self.actor else None

    @property
    def tombstone(self) -> bool:
        return self.body is None

    def __repr__(self) -> str:
        return (
            f"MemoryVersionRow(id={self.id}, entry_id={self.entry_id}, "
            f"tombstone={self.tombstone})"
        )


class PersonRow(Base):
    """Someone the character knows: a name, and the page that is their profile."""

    __tablename__ = "known_people"

    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str]
    #: The profile's root page. Joined, since a profile is read to be rendered
    #: after its session is gone.
    root_entry_id: Mapped[int] = mapped_column(ForeignKey("memory_entries.id"), unique=True)
    root: Mapped[MemoryEntryRow] = relationship(lazy="joined")
    #: NULL is the operator, as on `memory_versions`.
    created_by: Mapped[str | None] = mapped_column(ForeignKey("agents.id"), default=None)
    created_at: Mapped[datetime] = mapped_column(default=utcnow)
    updated_at: Mapped[datetime] = mapped_column(default=utcnow, onupdate=utcnow)

    def __repr__(self) -> str:
        return f"PersonRow(id={self.id}, name={self.name!r})"


class PersonNameRow(Base):
    """A handle on a venue, and whose it is. A handle belongs to one person."""

    __tablename__ = "person_names"
    __table_args__ = (
        UniqueConstraint("venue", "username", name="uq_person_names_venue_username"),
    )

    id: Mapped[int] = mapped_column(primary_key=True)
    person_id: Mapped[int] = mapped_column(ForeignKey("known_people.id"), index=True)
    person: Mapped[PersonRow] = relationship(lazy="joined")
    #: The medium, as `MessageEvent.venue` spells it: "discord", "email".
    venue: Mapped[str] = mapped_column(String(64))
    #: The handle, as `MessageEvent.sender` spells it.
    username: Mapped[str] = mapped_column(String(255))
    created_by: Mapped[str | None] = mapped_column(ForeignKey("agents.id"), default=None)
    created_at: Mapped[datetime] = mapped_column(default=utcnow)

    def __repr__(self) -> str:
        return f"PersonNameRow({self.venue}/{self.username} -> person {self.person_id})"


class HeartbeatRow(Base):
    """A schedule on which a role's loop runs a pass.

    One primitive for three things. The standing schedule is recurring, never
    expires, carries no message and is the loop's own (`created_by` NULL). A
    temporary override is recurring with `expires_at`. A check-in is one-shot,
    `interval_seconds` NULL, and carries the note a pass left for a later one.

    The row says how often; *when* lives in `heartbeat_ticks`. A schedule is
    live iff it has an unfired tick, so ending one means firing its last tick
    with no successor — never deleting or flagging the row, so what was
    scheduled and by whom stays on record.
    """

    __tablename__ = "heartbeats"

    id: Mapped[int] = mapped_column(primary_key=True)

    #: Whose pass this schedules. Not the creator's role: `everyone` lets a
    #: triage agent schedule the sweep.
    role: Mapped[AgentRole] = mapped_column(
        SAEnum(AgentRole, native_enum=False, length=16)
    )
    #: NULL is one-shot: fired once, when due, never re-timed.
    interval_seconds: Mapped[float | None] = mapped_column(default=None)
    #: What the pass that fires this is told. The standing schedule has none.
    message: Mapped[str | None] = mapped_column(default=None)
    #: A recurring schedule gets no successor tick that would fall past this.
    expires_at: Mapped[datetime | None] = mapped_column(default=None)

    #: The agent whose tool call wrote this; NULL is the operator, which
    #: includes `watch` declaring the standing schedule from config.
    created_by: Mapped[str | None] = mapped_column(
        ForeignKey("agents.id"), default=None
    )
    #: Joined, as `ContextRow.actor` is: rows are rendered after their session
    #: is gone, and the label needs the creator's role.
    creator: Mapped[AgentRow | None] = relationship(lazy="joined")
    created_at: Mapped[datetime] = mapped_column(default=utcnow)

    @property
    def recurring(self) -> bool:
        return self.interval_seconds is not None

    @property
    def standing(self) -> bool:
        """The loop's own rhythm: recurring, open-ended, silent, nobody's."""
        return (
            self.recurring
            and self.expires_at is None
            and self.message is None
            and self.created_by is None
        )

    @property
    def agent(self) -> Agent | None:
        return self.creator.to_agent() if self.creator else None

    def __repr__(self) -> str:
        every = f"every {self.interval_seconds:g}s" if self.recurring else "once"
        return f"HeartbeatRow(id={self.id}, role={self.role.value}, {every})"


class HeartbeatTickRow(Base):
    """When a schedule next fires, and when it did. Append-only.

    A schedule has at most one unfired tick; the partial unique index says so,
    so a bug that inserted a second fails loudly rather than doubling a pass.
    Firing stamps the tick and, for a recurring schedule, inserts the successor
    in the same transaction. A tick fired before it was due is the record of a
    shift: an immediate arrival or a check-in brought the pass forward.
    """

    __tablename__ = "heartbeat_ticks"
    __table_args__ = (
        Index(
            "uq_heartbeat_ticks_live",
            "heartbeat_id",
            unique=True,
            sqlite_where=text("fired_at IS NULL"),
        ),
        # The loop's one read is the earliest unfired tick; fired ones pile up
        # beneath it, one per pass, forever.
        Index("ix_heartbeat_ticks_due", "due_at", sqlite_where=text("fired_at IS NULL")),
    )

    id: Mapped[int] = mapped_column(primary_key=True)
    heartbeat_id: Mapped[int] = mapped_column(ForeignKey("heartbeats.id"))
    #: Joined: a tick is read to act on its schedule, after the session is gone.
    schedule: Mapped[HeartbeatRow] = relationship(lazy="joined")

    due_at: Mapped[datetime]
    fired_at: Mapped[datetime | None] = mapped_column(default=None)

    @property
    def live(self) -> bool:
        return self.fired_at is None

    def __repr__(self) -> str:
        when = f"due {self.due_at.isoformat(timespec='seconds')}"
        if self.fired_at is not None:
            when += f", fired {self.fired_at.isoformat(timespec='seconds')}"
        return f"HeartbeatTickRow(id={self.id}, heartbeat_id={self.heartbeat_id}, {when})"


async def resolve_stream(session: AsyncSession, ref: StreamRef) -> int:
    """The id of the stream `ref` names, opening it if this is its first event.

    One statement: SQLite's upsert either inserts or returns the row already
    there, so two events racing to open the same stream cannot produce two of
    it. The assignment on conflict looks redundant but is not — `DO NOTHING`
    yields no `RETURNING` row, which would cost a second query to recover.
    """
    stmt = (
        sqlite_insert(StreamRow)
        .values(kind=ref.kind, key=ref.key, title=ref.title)
        .on_conflict_do_update(
            index_elements=["kind", "key"], set_={"updated_at": utcnow()}
        )
        .returning(StreamRow.id)
    )
    stream_id = await session.scalar(stmt)
    if stream_id is None:  # RETURNING on an upsert always yields exactly one row
        raise RuntimeError(f"stream upsert returned nothing for {ref.key!r}")
    return stream_id


class StreamNotFound(KeyError):
    """No stream with that id."""


async def spawn_agent(session: AsyncSession, role: AgentRole) -> AgentRow:
    """Mint an agent: a new row, in the caller's transaction.

    This is the only way an agent comes into being, so nothing can be
    attributed to one the store has never heard of — the foreign keys on
    `event_actions` and `contexts` see to the rest.
    """
    row = AgentRow(role=role)
    session.add(row)
    await session.flush()  # so the id is settled before anything points at it
    return row


async def load_context(session: AsyncSession, context_id: int) -> ContextRow:
    """The context with this id. Only a foreign key ever supplies one, so missing is a bug."""
    context = await session.get(ContextRow, context_id)
    if context is None:
        raise RuntimeError(f"no such context: {context_id}")
    return context


async def resolve_context(session: AsyncSession, stream_id: int) -> ContextRow:
    """The context `stream_id` routes to, opened on first need.

    Opening mints the subagent whose context it will be and points the stream
    at it, all in the caller's transaction. Two writers opening one stream's
    context at the same moment would leave an orphan; the dispatcher resolves
    contexts from the pass's tool calls, which run one at a time, so that does
    not arise within a process.
    """
    stream = await session.get(StreamRow, stream_id)
    if stream is None:
        raise StreamNotFound(f"no such stream: {stream_id}")
    if stream.context_id is not None:
        return await load_context(session, stream.context_id)

    # Attached as the row rather than by id, so `context.agent` is answerable
    # right away instead of needing a load the async session would refuse.
    context = ContextRow(actor=await spawn_agent(session, AgentRole.SUBAGENT))
    session.add(context)
    await session.flush()  # for its id
    stream.context_id = context.id
    return context


async def join_context(session: AsyncSession, stream_id: int, context_id: int) -> None:
    """Point a stream at a context already open, so its work joins that conversation."""
    stream = await session.get(StreamRow, stream_id)
    if stream is None:
        raise StreamNotFound(f"no such stream: {stream_id}")
    stream.context_id = (await load_context(session, context_id)).id


async def route(session: AsyncSession, event_ids: Sequence[int]) -> ContextRow | None:
    """The one context these events belong in, or None for a one-shot subagent.

    One stream routes to its context, opened here if this is its first work.
    Several streams that already share a context route to it. Anything else —
    an event outside any stream, or streams whose contexts differ or are not
    yet open — is the case that will become a message between contexts, and
    until it is built goes to a subagent that remembers nothing.
    """
    query = (
        select(EventRow.stream_id, StreamRow.context_id)
        .outerjoin(StreamRow, EventRow.stream_id == StreamRow.id)
        .where(EventRow.id.in_(event_ids))
        .distinct()
    )
    pairs = list((await session.execute(query)).tuples())
    match pairs:
        case [(int() as stream_id, _)]:
            return await resolve_context(session, stream_id)
        case [(int(), int() as context_id), *rest] if all(
            s is not None and c == context_id for s, c in rest
        ):
            return await load_context(session, context_id)
        case _:
            return None


def _enforce_foreign_keys(
    dbapi_connection: DBAPIConnection, connection_record: ConnectionPoolEntry
) -> None:
    """SQLite declares foreign keys but ignores them unless told, per connection."""
    cursor = dbapi_connection.cursor()
    cursor.execute("PRAGMA foreign_keys=ON")
    cursor.close()


def create_engine(db_path: Path | str) -> AsyncEngine:
    """An async engine for `db_path`, or a shared in-memory database for tests."""
    if str(db_path) == IN_MEMORY:
        engine = create_async_engine(
            f"sqlite+aiosqlite:///{IN_MEMORY}",
            # One connection for the whole engine, so the schema survives between
            # sessions instead of each connection getting its own empty database.
            poolclass=StaticPool,
            connect_args={"check_same_thread": False},
        )
    else:
        engine = create_async_engine(f"sqlite+aiosqlite:///{db_path}")
    event.listen(engine.sync_engine, "connect", _enforce_foreign_keys)
    return engine


def session_factory(engine: AsyncEngine) -> async_sessionmaker[AsyncSession]:
    return async_sessionmaker(engine, expire_on_commit=False)


async def init_schema(engine: AsyncEngine) -> None:
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
