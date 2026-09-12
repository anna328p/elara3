"""Dispatcher rules that hold regardless of what the model asks for."""

from __future__ import annotations

from datetime import timedelta
from types import SimpleNamespace

import pytest
from mcp import Client

from event_prototype.agents import AgentRole
from event_prototype.config import Config
from event_prototype.events import JobEvent, Priority, ScheduledEvent
from event_prototype.queue import EventQueue
from event_prototype.render import PromptRenderer
from event_prototype.store import Action, Status, utcnow
from event_prototype.tools import Dispatcher, build_sweep_server, build_triage_server

from conftest import FakeMessages, message


async def dispatcher(queue: EventQueue, role: AgentRole = AgentRole.TRIAGE) -> Dispatcher:
    # No client: these tests never reach the point of running a subagent.
    return Dispatcher(
        queue,
        client=None,  # type: ignore[arg-type]
        config=Config(),
        renderer=PromptRenderer(),
        agent=await queue.spawn(role),
    )


async def test_an_event_gets_only_one_disposition_per_pass(queue: EventQueue) -> None:
    event_id = await queue.submit(message(), Priority.BACKGROUND)
    triage = await dispatcher(queue)

    await triage.defer(event_id, "not now")

    # A second, contradictory call is refused rather than quietly overwriting.
    with pytest.raises(ValueError, match="already given a disposition"):
        await triage.assign([event_id], "actually, do it", Action.HANDLE_ONE_EVENT)

    assert (await queue.get(event_id)).status is Status.DEFERRED
    (entry,) = (await queue.history_for([event_id]))[event_id]
    assert entry.action is Action.DEFER_EVENT


async def test_a_rejected_disposition_leaves_the_batch_untouched(
    queue: EventQueue,
) -> None:
    first = await queue.submit(message("one"), Priority.BACKGROUND)
    second = await queue.submit(message("two"), Priority.BACKGROUND)
    triage = await dispatcher(queue)

    await triage.defer(first, "not now")

    with pytest.raises(ValueError):
        await triage.assign([second, first], "together", Action.HANDLE_EVENT_SEQUENCE)

    # The untainted event in the batch is still free for a later disposition.
    assert await queue.history_for([second]) == {}
    await triage.defer(second, "on reflection, no")
    assert (await queue.get(second)).status is Status.DEFERRED


async def test_setting_an_event_aside_writes_its_backlog_line(queue: EventQueue) -> None:
    event_id = await queue.submit(message(), Priority.BACKGROUND)
    messages = FakeMessages("  mira asked about\n  the gradient banding  ")
    triage = await dispatcher(queue)
    triage.client = SimpleNamespace(messages=messages)  # type: ignore[assignment]

    await triage.defer(event_id, "not now")
    assert (await queue.get(event_id)).digest is None  # not written inline...
    await triage.drain()  # ...it is spawned alongside the subagents

    assert len(messages.calls) == 1
    # However the model lays it out, the backlog line is one line.
    assert (await queue.get(event_id)).digest == "mira asked about the gradient banding"


async def test_keeping_an_event_deferred_rewrites_its_backlog_line(
    queue: EventQueue,
) -> None:
    event_id = await queue.submit(message(), Priority.BACKGROUND)
    await queue.defer([(event_id, "not now")], agent=await queue.spawn(AgentRole.TRIAGE))
    await queue.set_digest(event_id, "the line triage has been reading")

    sweeper = await dispatcher(queue, AgentRole.SWEEP)
    sweeper.client = SimpleNamespace(messages=FakeMessages("still waiting on quill"))  # type: ignore[assignment]
    await sweeper.keep_deferred(event_id, "no follow-up yet")
    await sweeper.drain()

    assert (await queue.get(event_id)).digest == "still waiting on quill"


async def test_a_failed_digest_leaves_the_previous_line_standing(
    queue: EventQueue,
) -> None:
    event_id = await queue.submit(message(), Priority.BACKGROUND)
    triage = await dispatcher(queue)  # client is None, so summarizing raises

    await triage.defer(event_id, "not now")
    await triage.drain()  # must not propagate

    row = await queue.get(event_id)
    assert row.digest is None
    assert row.status is Status.DEFERRED  # the disposition itself still stands


async def test_an_assignment_records_the_streams_it_spanned(queue: EventQueue) -> None:
    """A sequence crossing streams is the case streams cannot express on their own."""
    job = await queue.submit(
        JobEvent(
            timestamp=utcnow(),
            description="render finished",
            job_id="render-1",
            outcome="ok",
            summary="done",
            job="render",
        ),
        Priority.BACKGROUND,
    )
    question = await queue.submit(message(), Priority.ACTIVE)
    triage = await dispatcher(queue)

    # No client, so the subagent fails — the streams are recorded either way.
    await triage.assign([job, question], "answer her", Action.HANDLE_EVENT_SEQUENCE)
    await triage.drain()

    (disposition,) = triage.dispositions
    assert disposition.streams == ("render job", "#workshop")


async def test_a_streamless_event_contributes_no_stream(queue: EventQueue) -> None:
    alarm = await queue.submit(
        ScheduledEvent(
            timestamp=utcnow(), description="one-off", fires_at=utcnow(), note="once"
        ),
        Priority.BACKGROUND,
    )
    triage = await dispatcher(queue)

    await triage.assign([alarm], "do it", Action.HANDLE_ONE_EVENT)
    await triage.drain()

    (disposition,) = triage.dispositions
    assert disposition.streams == ()


async def test_separate_passes_may_revisit_the_same_event(queue: EventQueue) -> None:
    event_id = await queue.submit(message(), Priority.BACKGROUND)

    triage = await dispatcher(queue)
    await triage.defer(event_id, "not now")
    # The claim is per pass, so a later sweep can still act on it.
    sweeper = await dispatcher(queue, AgentRole.SWEEP)
    await sweeper.archive(event_id, "never actionable")

    # Archived events are no longer addressable, so look in the full listing.
    (row,) = await queue.list_events(active_only=False)
    assert row.archived_at is not None


async def test_a_check_in_is_written_for_the_calling_role_by_the_pass_agent(
    queue: EventQueue,
) -> None:
    sweeper = await dispatcher(queue, AgentRole.SWEEP)
    before = utcnow()

    summary = await sweeper.schedule_check_in(5, "  see whether the render job reported  ")

    schedule = summary.schedule
    assert schedule.role is AgentRole.SWEEP
    assert schedule.created_by == sweeper.agent.id
    assert schedule.agent == sweeper.agent
    assert schedule.interval_seconds is None
    assert schedule.message == "see whether the render job reported"
    assert timedelta(minutes=5) <= summary.next_due - before < timedelta(minutes=5, seconds=1)
    assert sweeper.scheduled == [summary]


async def test_an_interval_override_is_attributed_and_expires(queue: EventQueue) -> None:
    triage = await dispatcher(queue)
    before = utcnow()

    (summary,) = await triage.set_heartbeat_interval(5, 60, "live conversation with mira")

    schedule = summary.schedule
    assert schedule.role is AgentRole.TRIAGE
    assert schedule.created_by == triage.agent.id
    assert schedule.interval_seconds == 300
    assert schedule.message == "live conversation with mira"
    assert schedule.expires_at is not None
    assert timedelta(minutes=60) <= schedule.expires_at - before < timedelta(minutes=60, seconds=1)
    assert timedelta(minutes=5) <= summary.next_due - before < timedelta(minutes=5, seconds=1)


async def test_everyone_expands_to_every_role_with_a_heartbeat(queue: EventQueue) -> None:
    triage = await dispatcher(queue)

    summaries = await triage.set_heartbeat_interval(5, 60, "busy hour", everyone=True)

    assert [s.schedule.role for s in summaries] == list(Config().heartbeat_seconds)
    # A triage agent wrote the sweep's row; the attribution says so.
    assert all(s.schedule.created_by == triage.agent.id for s in summaries)
    assert len(await queue.heartbeats()) == len(summaries)


async def test_an_override_that_could_never_fire_is_refused(queue: EventQueue) -> None:
    triage = await dispatcher(queue)

    with pytest.raises(ValueError, match="could not fire once"):
        await triage.set_heartbeat_interval(10, 5, "too short")
    with pytest.raises(ValueError, match="must be positive"):
        await triage.set_heartbeat_interval(0, 5, "no interval")
    with pytest.raises(ValueError, match="in the future"):
        await triage.schedule_check_in(0, "now")
    with pytest.raises(ValueError, match="needs a message"):
        await triage.schedule_check_in(5, "   ")

    assert await queue.heartbeats() == []


async def test_the_schedule_tools_are_on_both_tool_sets(queue: EventQueue) -> None:
    triage = await dispatcher(queue)
    for build in (build_triage_server, build_sweep_server):
        async with Client(build(triage)) as mcp:
            names = {tool.name for tool in (await mcp.list_tools()).tools}
        assert {"schedule_check_in", "set_heartbeat_interval"} <= names
