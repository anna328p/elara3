"""Streams: which ongoing context an event belongs to, and what that costs to read."""

from __future__ import annotations

from datetime import timedelta

from event_prototype.agents import Agent, AgentRole
from event_prototype.events import JobEvent, Priority, ScheduledEvent
from event_prototype.queue import EventQueue
from event_prototype.render import PromptRenderer
from event_prototype.store import utcnow
from event_prototype.streams import StreamKind

from conftest import counting_selects, message


# -- derivation, no database needed ---------------------------------------


def test_a_room_is_one_stream_however_many_people_are_in_it() -> None:
    mira = message(sender="mira", conversation="#workshop").stream
    hal = message(sender="hal", conversation="#workshop").stream

    assert mira == hal
    assert mira is not None and mira.kind is StreamKind.CHANNEL


def test_each_direct_exchange_is_its_own_stream() -> None:
    tobias = message(sender="tobias", conversation="dm:tobias", direct=True).stream
    ines = message(sender="ines", conversation="dm:ines", direct=True).stream

    assert tobias is not None and ines is not None
    assert tobias != ines
    assert tobias.kind is StreamKind.DIRECT
    assert tobias.title == "tobias"


def test_a_venue_is_not_a_stream() -> None:
    """There is no "all email" stream: the venue only qualifies a conversation."""
    first = message(sender="ines", conversation="thread-1", venue="email", direct=True).stream
    second = message(sender="tobias", conversation="thread-2", venue="email", direct=True).stream

    assert first != second

    # And the same conversation id in two venues is two streams, not one.
    assert (
        message(sender="q", conversation="#general", venue="discord").stream
        != message(sender="q", conversation="#general", venue="irc").stream
    )


def test_a_one_off_belongs_to_nothing_ongoing() -> None:
    alarm = ScheduledEvent(
        timestamp=utcnow(), description="once", fires_at=utcnow(), note="do the thing"
    )
    run = JobEvent(
        timestamp=utcnow(), description="once", job_id="render-1", outcome="ok", summary=""
    )

    assert alarm.stream is None
    assert run.stream is None


def test_a_recurrence_belongs_to_the_series_not_the_run() -> None:
    first = JobEvent(
        timestamp=utcnow(),
        description="failed",
        job_id="backup-1182",
        outcome="failed",
        summary="",
        job="backup",
    )
    second = JobEvent(
        timestamp=utcnow(),
        description="failed again",
        job_id="backup-1183",
        outcome="failed",
        summary="",
        job="backup",
    )

    assert first.stream == second.stream


def test_a_new_field_does_not_break_a_stored_event() -> None:
    """`stream` is a property, so it stays out of the payload entirely."""
    event = message(sender="mira", conversation="#workshop")
    assert "stream" not in event.payload()
    # And the new fields round-trip as ordinary payload data.
    assert event.payload()["direct"] is False
    assert event.payload()["conversation"] == "#workshop"


# -- persistence ----------------------------------------------------------


async def test_events_in_a_room_join_one_stream_row(queue: EventQueue) -> None:
    first = await queue.submit(message(sender="mira", conversation="#workshop"), Priority.NORMAL)
    second = await queue.submit(message(sender="hal", conversation="#workshop"), Priority.LOW)
    elsewhere = await queue.submit(message(sender="quill", conversation="#general"), Priority.LOW)

    rows = {row.id: row for row in await queue.get_many([first, second, elsewhere])}

    assert rows[first].stream_id == rows[second].stream_id
    assert rows[elsewhere].stream_id != rows[first].stream_id
    # Opened once, not once per event.
    assert len(await queue.list_streams()) == 2


async def test_a_streamless_event_stores_null_and_renders_without_one(
    queue: EventQueue,
) -> None:
    event_id = await queue.submit(
        ScheduledEvent(
            timestamp=utcnow(), description="one-off", fires_at=utcnow(), note="once"
        ),
        Priority.LOW,
    )

    row = await queue.get(event_id)
    assert row.stream_id is None
    assert row.stream is None

    # StrictUndefined would raise if the template were unguarded.
    prompt = PromptRenderer().triage([row])
    assert "stream=" not in prompt


async def test_last_activity_is_the_newest_event_not_the_last_submitted(
    queue: EventQueue,
) -> None:
    now = utcnow()
    recent = message(timestamp=now, sender="mira", description="recent")
    old = message(timestamp=now - timedelta(days=2), sender="hal", description="old")

    # Newest first, oldest second: a stamp-on-write column would record the old one.
    await queue.submit(recent, Priority.NORMAL)
    await queue.submit(old, Priority.LOW)

    (summary,) = await queue.list_streams()
    assert summary.last_event_at == now
    assert summary.active == 2


async def test_a_completed_event_leaves_the_stream_but_not_the_count(
    queue: EventQueue,
) -> None:
    first = await queue.submit(message(sender="mira", conversation="#workshop"), Priority.NORMAL)
    await queue.submit(message(sender="hal", conversation="#workshop"), Priority.LOW)
    await queue.complete([first], agent=await queue.spawn(AgentRole.SUBAGENT), report="done")

    (summary,) = await queue.list_streams()
    assert summary.active == 1  # the stream itself stays


# -- the cost of reading it ------------------------------------------------


async def test_the_stream_rides_along_and_survives_the_session(
    queue: EventQueue, triage: Agent
) -> None:
    """The regression this guards is a raise, not a wrong answer.

    Rows outlive the session that produced them, so a lazily-loaded stream
    would raise wherever a prompt is rendered. `lazy="joined"` means the join
    happens in the same statement — hence the counts below are unchanged.
    """
    await queue.submit(message(sender="mira", conversation="#workshop"), Priority.HIGH)
    deferred = await queue.submit(message(sender="hal", conversation="#workshop"), Priority.LOW)
    await queue.defer([(deferred, "later")], agent=triage)

    with counting_selects() as selects:
        view = await queue.triage_view()
    assert len(selects) == 1, selects
    assert view.pending[0].stream is not None  # outside the session, does not raise

    with counting_selects() as selects:
        sweep = await queue.sweep_view()
    assert len(selects) == 2, selects
    assert sweep.rows[0].stream is not None

    with counting_selects() as selects:
        row = await queue.get(deferred)
    assert len(selects) == 1, selects
    assert row.stream is not None


async def test_listing_every_stream_is_one_query(queue: EventQueue) -> None:
    for sender, where in (("mira", "#workshop"), ("hal", "#workshop"), ("q", "#general")):
        await queue.submit(message(sender=sender, conversation=where), Priority.LOW)

    with counting_selects() as selects:
        summaries = await queue.list_streams()

    assert len(selects) == 1, selects
    assert len(summaries) == 2


# -- what the prompts show -------------------------------------------------


async def test_the_stream_reaches_the_prompt_and_the_backlog_line(
    queue: EventQueue, triage: Agent
) -> None:
    pending = await queue.submit(message(sender="mira", conversation="#workshop"), Priority.HIGH)
    deferred = await queue.submit(message(sender="quill", conversation="#general"), Priority.LOW)
    await queue.defer([(deferred, "no question in it")], agent=triage)

    view = await queue.triage_view()
    prompt = PromptRenderer().triage(view.pending, view.deferred)

    # The full event carries it as an attribute...
    assert 'stream="#workshop"' in prompt
    # ...and the backlog one-liner keeps its id prefix while gaining the stream.
    line = next(
        entry for entry in prompt.splitlines() if entry.startswith(f"{deferred}:")
    )
    assert "(#general)" in line
    assert str(pending) not in line
