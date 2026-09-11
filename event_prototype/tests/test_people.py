"""Known people: handles, profile pages, and the sender behind an event."""

from __future__ import annotations

import pytest

from event_prototype.agents import Agent, AgentRole
from event_prototype.events import JobEvent, Priority
from event_prototype.queue import EventQueue
from event_prototype.render import PromptRenderer
from event_prototype.store import Action, utcnow

from conftest import counting_selects, message

FRESH_MIRA = "# Mira\n\nKnown as discord/mira.\n\n## Notes\n\n"


async def test_a_handle_links_to_one_person_and_a_second_venue_joins_them(
    queue: EventQueue,
) -> None:
    mira = await queue.people.link("discord", "mira", "Mira", None)
    again = await queue.people.link("email", "mira@example.org", "Mira", None)

    assert mira.id == again.id and mira.path == "/memories/people/mira.md"
    assert await queue.memory.read(mira.path) == FRESH_MIRA
    with pytest.raises(ValueError, match="already Mira's"):
        await queue.people.link("discord", "mira", "Miranda", None)
    ((person, handles),) = await queue.people.people()
    assert person.name == "Mira" and person.created_by is None
    assert set(handles) == {"discord/mira", "email/mira@example.org"}


async def test_linking_adopts_a_page_the_model_wrote_first(queue: EventQueue) -> None:
    writer = await queue.spawn(AgentRole.SUBAGENT)
    await queue.memory.create("/memories/people/hal.md", "# Hal\n\nBrush engine person.\n", writer)

    hal = await queue.people.link("discord", "hal", "Hal", writer)

    assert hal.path == "/memories/people/hal.md" and hal.body.startswith("# Hal")
    assert len(await queue.memory.history(hal.path)) == 1  # nothing rewritten
    ((person, _),) = await queue.people.people()
    assert person.created_by == writer.id


async def test_a_slug_someone_else_holds_is_not_taken_over(queue: EventQueue) -> None:
    mira = await queue.people.link("discord", "mira", "Mira", None)
    # Another person whose name slugs the same way gets the next path.
    other = await queue.people.link("irc", "mira", "mira", None)

    assert other.id != mira.id
    assert other.path == "/memories/people/mira-2.md"
    assert await queue.memory.read(mira.path) == FRESH_MIRA


async def test_an_assignment_resolves_senders_in_one_query(
    queue: EventQueue, triage: Agent
) -> None:
    await queue.people.link("discord", "mira", "Mira", None)
    known = await queue.submit(message("hi", sender="mira"), Priority.NUDGE)
    unknown = await queue.submit(message("yo", sender="quill"), Priority.NUDGE)

    with counting_selects() as selects:
        assignment = await queue.assign(
            [known, unknown], agent=triage, instructions="both", action=Action.HANDLE_EVENT_SEQUENCE
        )

    # The rows, the routing pairs, the profiles: three, not three plus one per event.
    assert len(selects) == 3, selects
    assert set(assignment.people) == {known}
    assert assignment.people[known].name == "Mira"
    assert assignment.people[known].body == FRESH_MIRA


async def test_events_without_a_sender_need_no_lookup(queue: EventQueue, triage: Agent) -> None:
    job = await queue.submit(
        JobEvent(timestamp=utcnow(), description="done", job_id="j1", outcome="ok", summary="ok"),
        Priority.BACKGROUND,  # a streamless job can go nowhere but triage
    )

    with counting_selects() as selects:
        assignment = await queue.assign(
            [job], agent=triage, instructions="note it", action=Action.HANDLE_ONE_EVENT
        )

    assert len(selects) == 2, selects  # the rows and the routing pairs; no profile query
    assert assignment.people == {}


async def test_the_event_turn_carries_the_profile(queue: EventQueue) -> None:
    profile = await queue.people.link("discord", "mira", "Mira", None)
    row = await queue.get(await queue.submit(message("hi"), Priority.NUDGE))
    renderer = PromptRenderer()

    turn = renderer.event(row, person=profile.view)

    assert '<sender name="Mira" memory="/memories/people/mira.md">' in turn
    assert "Known as discord/mira." in turn
    assert "<sender" not in renderer.event(row)
