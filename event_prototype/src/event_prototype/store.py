"""SQLite persistence for the event queue.

`events` is current state and `event_log` the append-only record of what was
decided about each one and by whom — "whom" being a row in `agents`, which every
attribution points at. `streams` is the ongoing loci events belong to, each
routed to at most one `contexts` row: the conversation an agent is having there,
stored as `turns`. Rows are never deleted — archiving stamps `archived_at` and
every default query filters those out, so the audit trail stays intact.
"""

from __future__ import annotations

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
    TypeDecorator,
    UniqueConstraint,
    event,
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


class LogAction(StrEnum):
    """What was done to an event. Most correspond one-to-one with model tools."""

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
    """An agent: the thing log entries are attributed to and contexts belong to.

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
    #: the event log and the memory in the transcript are the same thing.
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

    # Why an event is in the state it is lives in `event_log`, not here — the
    # row carries current state, the log carries the reasoning behind it.

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


class EventLogRow(Base):
    """One thing an agent did about one event. Append-only."""

    __tablename__ = "event_log"
    __table_args__ = (
        # Every read is "the history of these events, oldest first".
        Index("ix_event_log_event_timestamp", "event_id", "timestamp"),
        # ...and occasionally "everything this agent touched".
        Index("ix_event_log_agent", "agent_id"),
    )

    id: Mapped[int] = mapped_column(primary_key=True)
    event_id: Mapped[int] = mapped_column(ForeignKey("events.id"))
    timestamp: Mapped[datetime] = mapped_column(default=utcnow)

    action: Mapped[LogAction] = mapped_column(
        SAEnum(LogAction, native_enum=False, length=32)
    )
    #: The triage model's reason or instructions, or the subagent's report.
    detail: Mapped[str]

    #: Who did it.
    agent_id: Mapped[str] = mapped_column(ForeignKey("agents.id"))
    #: On an assignment, the subagent the work went to.
    assigned_agent_id: Mapped[str | None] = mapped_column(
        ForeignKey("agents.id"), default=None
    )
    # Both joined: the log is read to be rendered, after its session is gone,
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
            f"EventLogRow(event_id={self.event_id}, action={self.action.value}, "
            f"agent={self.agent.label!r})"
        )


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
    `event_log` and `contexts` see to the rest.
    """
    row = AgentRow(role=role)
    session.add(row)
    await session.flush()  # so the id is settled before anything points at it
    return row


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
        context = await session.get(ContextRow, stream.context_id)
        if context is None:  # the foreign key says otherwise
            raise RuntimeError(f"stream {stream_id} routes to a missing context")
        return context

    # Attached as the row rather than by id, so `context.agent` is answerable
    # right away instead of needing a load the async session would refuse.
    context = ContextRow(actor=await spawn_agent(session, AgentRole.SUBAGENT))
    session.add(context)
    await session.flush()  # for its id
    stream.context_id = context.id
    return context


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
