"""SQLite persistence for the event queue.

Two tables: `events` is current state, `event_log` is the append-only record of
what was decided about each one and by whom. Rows are never deleted — archiving
stamps `archived_at` and every default query filters those out, so the audit
trail stays intact.
"""

from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import Any

from sqlalchemy import JSON, DateTime, ForeignKey, Index, Integer, String, TypeDecorator
from sqlalchemy import Enum as SAEnum
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column
from sqlalchemy.pool import StaticPool

from .agents import Agent, AgentRole
from .events import BaseEvent, Priority, from_payload

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


class EventRow(Base):
    __tablename__ = "events"

    id: Mapped[int] = mapped_column(primary_key=True)

    kind: Mapped[str] = mapped_column(String(64))
    timestamp: Mapped[datetime]
    description: Mapped[str]
    payload: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)

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

    # The role rides along with the id: there is no agents table yet, and the
    # log has to be readable on its own.
    agent_id: Mapped[str] = mapped_column(String(36))
    agent_role: Mapped[AgentRole] = mapped_column(
        SAEnum(AgentRole, native_enum=False, length=16)
    )
    #: On an assignment, the subagent the work went to.
    assigned_agent_id: Mapped[str | None] = mapped_column(String(36), default=None)

    @property
    def agent(self) -> Agent:
        return Agent(self.agent_role, self.agent_id)

    def __repr__(self) -> str:
        return (
            f"EventLogRow(event_id={self.event_id}, action={self.action.value}, "
            f"agent={self.agent.label!r})"
        )


def create_engine(db_path: Path | str) -> AsyncEngine:
    """An async engine for `db_path`, or a shared in-memory database for tests."""
    if str(db_path) == IN_MEMORY:
        return create_async_engine(
            f"sqlite+aiosqlite:///{IN_MEMORY}",
            # One connection for the whole engine, so the schema survives between
            # sessions instead of each connection getting its own empty database.
            poolclass=StaticPool,
            connect_args={"check_same_thread": False},
        )
    return create_async_engine(f"sqlite+aiosqlite:///{db_path}")


def session_factory(engine: AsyncEngine) -> async_sessionmaker[AsyncSession]:
    return async_sessionmaker(engine, expire_on_commit=False)


async def init_schema(engine: AsyncEngine) -> None:
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
