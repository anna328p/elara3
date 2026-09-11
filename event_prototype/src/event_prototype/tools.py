"""The MCP tools the triage and sweep models use to dispose of events.

Two tool sets over one dispatcher. Triage decides what needs attention now;
the sweep decides what the leftovers are worth. Handling is shared: it mints a
subagent, records the assignment, spawns its task and returns immediately, so
the caller can keep going while subagents run in parallel. `Dispatcher.drain()`
waits for them at the end of a pass.
"""

from __future__ import annotations

import asyncio
from collections.abc import Coroutine, Sequence
from dataclasses import dataclass, field, replace
from typing import Any

from anthropic import AsyncAnthropic
from anthropic.types import Usage
from mcp.server import MCPServer

from .agents import Agent
from .config import Config
from .contexts import Turn
from .digest import summarize
from .events import Priority, PriorityName
from .queue import Assignment, EventQueue
from .render import PromptRenderer
from .store import Action, EventRow
from .subagent import report_of, run_subagent


@dataclass(frozen=True, slots=True)
class Disposition:
    """What triage decided about a set of events, and how it turned out.

    The durable record is `event_actions`; this is the in-memory account of a
    single pass, for the report printed at the end of it.
    """

    action: Action
    event_ids: list[int]
    detail: str
    agent: Agent | None = None
    report: str | None = None
    error: str | None = None
    #: The streams these events came from, in order and deduplicated. More than
    #: one means the assignment crossed streams: into a context they already
    #: share, or else to a one-shot subagent — the case that will become a
    #: message between persistent contexts rather than one subagent reading both.
    streams: tuple[str, ...] = ()
    #: The context the work went to; None means a one-shot subagent.
    context_id: int | None = None
    #: What the subagent's call cost, cache reads and writes included.
    usage: Usage | None = None


@dataclass
class Dispatcher:
    """Holds the machinery the tools need, and tracks the work they spawn."""

    queue: EventQueue
    client: AsyncAnthropic
    config: Config
    renderer: PromptRenderer
    #: The triage agent every decision this pass is attributed to.
    agent: Agent

    dispositions: list[Disposition] = field(default_factory=lambda: [], init=False)
    _tasks: set[asyncio.Task[None]] = field(default_factory=lambda: set(), init=False, repr=False)
    #: What each event has already been given this pass, so it cannot be
    #: given a second, contradictory one.
    _claimed: dict[int, Action] = field(default_factory=lambda: {}, init=False, repr=False)

    async def drain(self) -> None:
        """Wait for every spawned subagent to finish."""
        while self._tasks:
            await asyncio.gather(*tuple(self._tasks))

    def _claim(self, event_ids: Sequence[int], action: Action) -> None:
        """Reserve these events for `action`, or say who got there first."""
        for event_id in event_ids:
            if (held := self._claimed.get(event_id)) is not None:
                raise ValueError(
                    f"Event {event_id} was already given a disposition this pass "
                    f"({held.value}). Each event gets exactly one; if you meant to "
                    f"change it, you cannot — say so in the next pass instead."
                )
        self._claimed.update(dict.fromkeys(event_ids, action))

    async def assign(
        self, event_ids: Sequence[int], instructions: str, action: Action
    ) -> Agent:
        """Hand events to a subagent and start it working.

        The queue decides where they go (see `store.route`) and records the
        assignment before the task is spawned, so the record exists even if
        handling never finishes.
        """
        self._claim(event_ids, action)
        assignment = await self.queue.assign(
            event_ids, agent=self.agent, instructions=instructions, action=action
        )
        self._spawn(self._handle(assignment, instructions, action))
        return assignment.subagent

    async def defer(self, event_id: int, reason: str) -> None:
        self._claim([event_id], Action.DEFER_EVENT)
        await self.queue.defer([(event_id, reason)], agent=self.agent)
        self._record(Action.DEFER_EVENT, event_id, reason)
        self._spawn(self._write_digest(event_id, reason))

    async def escalate(self, event_id: int, priority: Priority, reason: str) -> None:
        self._claim([event_id], Action.ESCALATE_EVENT)
        await self.queue.escalate(
            event_id, agent=self.agent, priority=priority, reason=reason
        )
        self._record(Action.ESCALATE_EVENT, event_id, f"[{priority.name.lower()}] {reason}")

    async def archive(self, event_id: int, reason: str) -> None:
        self._claim([event_id], Action.ARCHIVE_EVENT)
        await self.queue.archive(event_id, agent=self.agent, reason=reason)
        self._record(Action.ARCHIVE_EVENT, event_id, reason)

    async def keep_deferred(self, event_id: int, reason: str) -> None:
        self._claim([event_id], Action.KEEP_DEFERRED)
        await self.queue.keep_deferred(event_id, agent=self.agent, reason=reason)
        self._record(Action.KEEP_DEFERRED, event_id, reason)
        self._spawn(self._write_digest(event_id, reason))

    async def _write_digest(self, event_id: int, reason: str) -> None:
        """Refresh the line triage will see for a newly set-aside event.

        Spawned rather than awaited: this is for the *next* pass to read, so
        nothing in this one should wait on it. A failure leaves the previous
        line (or the description) standing, which is why it is swallowed.
        """
        try:
            row = await self.queue.get(event_id)
            digest = await summarize(self.client, self.config, self.renderer, row, reason)
            if digest:
                await self.queue.set_digest(event_id, digest)
        except Exception:
            return

    def _record(self, action: Action, event_id: int, detail: str) -> None:
        self.dispositions.append(Disposition(action, [event_id], detail, agent=self.agent))

    def _spawn(self, coro: Coroutine[Any, Any, None]) -> None:
        task = asyncio.create_task(coro)
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)

    async def _handle(self, assignment: Assignment, instructions: str, action: Action) -> None:
        """Run one subagent over the assigned rows and record what came back.

        With a context, the events and the brief are appended to it before the
        call and the reply after, so the transcript is the record of what the
        subagent was shown and what it said — a call that fails leaves the
        events in it, which is what happened.
        """
        rows, subagent, context = assignment.rows, assignment.subagent, assignment.context
        ids = [row.id for row in rows]
        outcome = Disposition(
            action,
            ids,
            instructions,
            agent=subagent,
            streams=_streams_of(rows),
            context_id=context.id if context else None,
        )
        try:
            turns = [Turn.user(self.renderer.event(row), event_id=row.id) for row in rows]
            turns.append(Turn.user(self.renderer.brief(instructions, sequence=len(rows) > 1)))
            if context:
                await self.queue.append_turns(context.id, turns)
                turns = [row.to_turn() for row in await self.queue.transcript(context.id)]
            response = await run_subagent(
                self.client,
                self.config,
                self.renderer.subagent_system(),
                turns,
                persistent=context is not None,
            )
            if context:
                await self.queue.append_turns(context.id, [Turn.of(response)])
            report = report_of(response)
            await self.queue.complete(ids, agent=subagent, report=report)
        except Exception as exc:  # a failed subagent leaves its events for the next pass
            error = f"{type(exc).__name__}: {exc}"
            await self.queue.record_failure(ids, agent=subagent, error=error)
            self.dispositions.append(replace(outcome, error=error))
            return
        self.dispositions.append(replace(outcome, report=report, usage=response.usage))


def _streams_of(rows: Sequence[EventRow]) -> tuple[str, ...]:
    """The streams these events came from, first seen first, without repeats."""
    titles = (row.stream.title for row in rows if row.stream is not None)
    return tuple(dict.fromkeys(titles))


def build_triage_server(dispatcher: Dispatcher) -> MCPServer:
    """An MCP server whose tools close over `dispatcher` — no module-level state."""
    server = MCPServer("elara-event-triage")

    # The wire names are given explicitly, so the Python names can carry the
    # underscore that marks them as registered-not-called.
    @server.tool(name="handle_one_event")
    async def _handle_one_event(event_id: int, instructions: str) -> str:
        """Assign one event to a subagent for handling.

        Use for an event that stands on its own. The subagent sees only this
        event and your instructions, so the instructions must be self-contained.

        Args:
            event_id: The id of the event to handle.
            instructions: The subagent's brief: what outcome you want and any
                context from the queue it needs.
        """
        subagent = await dispatcher.assign(
            [event_id], instructions, Action.HANDLE_ONE_EVENT
        )
        return f"Event {event_id} assigned to {subagent.label}."

    @server.tool(name="handle_event_sequence")
    async def _handle_event_sequence(event_ids: list[int], instructions: str) -> str:
        """Assign several related events to a single subagent, to handle in order.

        Use when the events are one piece of work, or when handling one changes
        how another should be handled. List them in the order they should be
        taken.

        Args:
            event_ids: The events to handle together, in processing order.
            instructions: The subagent's brief covering the whole group.
        """
        subagent = await dispatcher.assign(
            event_ids, instructions, Action.HANDLE_EVENT_SEQUENCE
        )
        listed = ", ".join(str(i) for i in event_ids)
        return f"Events {listed} assigned to {subagent.label}, in that order."

    @server.tool(name="defer_event")
    async def _defer_event(event_id: int, reason: str) -> str:
        """Postpone one event that does not need attention this pass.

        Call it once per event: unlike a sequence, deferrals are independent
        decisions, and batching them only tempts you to write one reason that
        covers several.

        Args:
            event_id: The event to postpone.
            reason: Why this event can wait and what would make it worth
                handling. It is kept as this event's history and read back at
                the next triage, so it must stand on its own without referring
                to other events.
        """
        await dispatcher.defer(event_id, reason)
        return f"Event {event_id} deferred."

    return server


def build_sweep_server(dispatcher: Dispatcher) -> MCPServer:
    """The backlog sweep's vocabulary: escalate, handle, archive, or leave be."""
    server = MCPServer("elara-backlog-sweep")

    @server.tool(name="escalate_event")
    async def _escalate_event(event_id: int, priority: PriorityName, reason: str) -> str:
        """Return a deferred event to triage, at the priority it should have had.

        Use when the event does need doing but is not so self-contained that you
        want to hand it straight to a subagent. It rejoins the working queue and
        triage picks it up on the next pass.

        Args:
            event_id: The event to escalate.
            priority: The priority it should carry from now on.
            reason: What makes it worth attention now, when it was not before.
        """
        await dispatcher.escalate(event_id, Priority.from_name(priority), reason)
        return f"Event {event_id} escalated to {priority} and returned to triage."

    @server.tool(name="handle_event")
    async def _handle_event(event_id: int, instructions: str) -> str:
        """Hand a deferred event straight to a subagent, without waiting for triage.

        Use when the event needs work now and you can brief it completely.

        Args:
            event_id: The event to handle.
            instructions: The subagent's brief. It sees only this event and your
                instructions, so include anything from the history it needs.
        """
        subagent = await dispatcher.assign(
            [event_id], instructions, Action.HANDLE_ONE_EVENT
        )
        return f"Event {event_id} assigned to {subagent.label}."

    @server.tool(name="archive_event")
    async def _archive_event(event_id: int, reason: str) -> str:
        """Retire an event that will never need action.

        Archiving drops the event from the queue for good; it survives only in
        its action record. Nothing undoes it, so be sure the event can never matter rather
        than merely not mattering today, and never archive something a person is
        waiting on.

        Args:
            event_id: The event to retire.
            reason: Why it can never need action. This is the last word on it.
        """
        await dispatcher.archive(event_id, reason)
        return f"Event {event_id} archived."

    @server.tool(name="keep_deferred")
    async def _keep_deferred(event_id: int, reason: str) -> str:
        """Leave an event deferred, with a fresh account of why.

        Args:
            event_id: The event to leave alone.
            reason: Why it still does not warrant action, and what would change
                that. Write it as it stands now rather than repeating the last
                reason; the next sweep reads both and the difference is the
                signal.
        """
        await dispatcher.keep_deferred(event_id, reason)
        return f"Event {event_id} left deferred."

    return server
