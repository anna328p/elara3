"""Dispatcher rules that hold regardless of what the model asks for."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from event_prototype.agents import Agent, AgentRole
from event_prototype.config import Config
from event_prototype.events import JobEvent, Priority, ScheduledEvent
from event_prototype.queue import EventQueue
from event_prototype.render import PromptRenderer
from event_prototype.store import LogAction, Status, utcnow
from event_prototype.tools import Dispatcher

from conftest import message


def dispatcher(queue: EventQueue, role: AgentRole = AgentRole.TRIAGE) -> Dispatcher:
    # No client: these tests never reach the point of running a subagent.
    return Dispatcher(
        queue,
        client=None,  # type: ignore[arg-type]
        config=Config(),
        renderer=PromptRenderer(),
        agent=Agent.spawn(role),
    )


async def test_an_event_gets_only_one_disposition_per_pass(queue: EventQueue) -> None:
    event_id = await queue.submit(message(), Priority.NORMAL)
    triage = dispatcher(queue)

    await triage.defer(event_id, "not now")

    # A second, contradictory call is refused rather than quietly overwriting.
    with pytest.raises(ValueError, match="already given a disposition"):
        await triage.assign([event_id], "actually, do it", LogAction.HANDLE_ONE_EVENT)

    assert (await queue.get(event_id)).status is Status.DEFERRED
    (entry,) = (await queue.history_for([event_id]))[event_id]
    assert entry.action is LogAction.DEFER_EVENT


async def test_a_rejected_disposition_leaves_the_batch_untouched(
    queue: EventQueue,
) -> None:
    first = await queue.submit(message("one"), Priority.NORMAL)
    second = await queue.submit(message("two"), Priority.NORMAL)
    triage = dispatcher(queue)

    await triage.defer(first, "not now")

    with pytest.raises(ValueError):
        await triage.assign([second, first], "together", LogAction.HANDLE_EVENT_SEQUENCE)

    # The untainted event in the batch is still free for a later disposition.
    assert await queue.history_for([second]) == {}
    await triage.defer(second, "on reflection, no")
    assert (await queue.get(second)).status is Status.DEFERRED


class FakeMessages:
    """Stands in for `client.messages`, returning a fixed one-liner."""

    def __init__(self, text: str) -> None:
        self.text = text
        self.calls = 0

    async def create(self, **kwargs: object) -> SimpleNamespace:
        self.calls += 1
        return SimpleNamespace(
            content=[SimpleNamespace(type="text", text=self.text)], stop_reason="end_turn"
        )


async def test_setting_an_event_aside_writes_its_backlog_line(queue: EventQueue) -> None:
    event_id = await queue.submit(message(), Priority.LOW)
    messages = FakeMessages("  mira asked about\n  the gradient banding  ")
    triage = dispatcher(queue)
    triage.client = SimpleNamespace(messages=messages)  # type: ignore[assignment]

    await triage.defer(event_id, "not now")
    assert (await queue.get(event_id)).digest is None  # not written inline...
    await triage.drain()  # ...it is spawned alongside the subagents

    assert messages.calls == 1
    # However the model lays it out, the backlog line is one line.
    assert (await queue.get(event_id)).digest == "mira asked about the gradient banding"


async def test_keeping_an_event_deferred_rewrites_its_backlog_line(
    queue: EventQueue,
) -> None:
    event_id = await queue.submit(message(), Priority.LOW)
    await queue.defer([(event_id, "not now")], agent=Agent.spawn(AgentRole.TRIAGE))
    await queue.set_digest(event_id, "the line triage has been reading")

    sweeper = dispatcher(queue, AgentRole.SWEEP)
    sweeper.client = SimpleNamespace(messages=FakeMessages("still waiting on quill"))  # type: ignore[assignment]
    await sweeper.keep_deferred(event_id, "no follow-up yet")
    await sweeper.drain()

    assert (await queue.get(event_id)).digest == "still waiting on quill"


async def test_a_failed_digest_leaves_the_previous_line_standing(
    queue: EventQueue,
) -> None:
    event_id = await queue.submit(message(), Priority.LOW)
    triage = dispatcher(queue)  # client is None, so summarizing raises

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
        Priority.NORMAL,
    )
    question = await queue.submit(message(), Priority.REALTIME)
    triage = dispatcher(queue)

    # No client, so the subagent fails — the streams are recorded either way.
    await triage.assign([job, question], "answer her", LogAction.HANDLE_EVENT_SEQUENCE)
    await triage.drain()

    (disposition,) = triage.dispositions
    assert disposition.streams == ("render job", "#workshop")


async def test_a_streamless_event_contributes_no_stream(queue: EventQueue) -> None:
    alarm = await queue.submit(
        ScheduledEvent(
            timestamp=utcnow(), description="one-off", fires_at=utcnow(), note="once"
        ),
        Priority.LOW,
    )
    triage = dispatcher(queue)

    await triage.assign([alarm], "do it", LogAction.HANDLE_ONE_EVENT)
    await triage.drain()

    (disposition,) = triage.dispositions
    assert disposition.streams == ()


async def test_separate_passes_may_revisit_the_same_event(queue: EventQueue) -> None:
    event_id = await queue.submit(message(), Priority.LOW)

    await dispatcher(queue).defer(event_id, "not now")
    # The claim is per pass, so a later sweep can still act on it.
    await dispatcher(queue, AgentRole.SWEEP).archive(event_id, "never actionable")

    # Archived events are no longer addressable, so look in the full listing.
    (row,) = await queue.list_events(active_only=False)
    assert row.archived_at is not None
