"""Known people: handles, profile pages, and the sender behind an event."""

from __future__ import annotations

import pytest

from event_prototype.agents import Agent, AgentRole
from event_prototype.events import JobEvent, Priority
from event_prototype.queue import EventQueue
from event_prototype.render import PromptRenderer
from event_prototype.store import Action, utcnow
from event_prototype.tools import build_subagent_server, mcp_tools

from conftest import counting_selects, message

FRESH_MIRA = "# Mira\n\nKnown as discord/mira.\n\n## Notes\n\n"


async def test_registering_scaffolds_a_profile_from_starter_notes(queue: EventQueue) -> None:
    writer = await queue.spawn(AgentRole.SUBAGENT)

    ines = await queue.people.register(
        "Ines", "Asked about the August brush pack.\n", ("discord", "ines"), writer
    )

    assert ines.path == "/memories/people/ines.md"
    assert ines.body == (
        "# Ines\n\nKnown as discord/ines.\n\n## Notes\n\nAsked about the August brush pack.\n"
    )
    ((person, handles),) = await queue.people.people()
    assert (person.id, handles, person.created_by) == (ines.id, ["discord/ines"], writer.id)
    (version,) = await queue.memory.history(ines.path)
    assert version.row.agent == writer and version.row.edit_metadata == {"command": "create"}

    # No handle yet is fine; the same name again is not, nor is someone else's handle.
    quill = await queue.people.register("Quill", "", None, None)
    assert quill.body == "# Quill\n\n## Notes\n\n"
    with pytest.raises(ValueError, match="already known"):
        await queue.people.register("Ines", "", None, None)
    with pytest.raises(ValueError, match="already Ines's"):
        await queue.people.register("Inez", "", ("discord", "ines"), None)
    # Starter notes cannot be dropped on a page that already exists.
    await queue.memory.create("/memories/people/hal.md", "# Hal\n", writer)
    with pytest.raises(ValueError, match="already exists"):
        await queue.people.register("Hal", "brushes", None, writer)
    assert (await queue.people.register("Hal", "", None, writer)).body == "# Hal\n"


async def test_the_subagent_server_offers_both_tools(queue: EventQueue) -> None:
    agent = await queue.spawn(AgentRole.SUBAGENT)
    async with mcp_tools(build_subagent_server(queue.people, agent)) as tools:
        by_name = {tool.name: tool for tool in tools}
        assert set(by_name) == {"register_person", "link_person"}
        said = await by_name["register_person"].call(
            {"name": "Ines", "notes": "Brush pack.", "venue": "discord", "username": "ines"}
        )
        assert "person 1" in str(said) and "/memories/people/ines.md" in str(said)
        said = await by_name["link_person"].call(
            {"venue": "email", "username": "ines@example.org", "name": "Ines"}
        )
        assert "/memories/people/ines.md" in str(said)
    ((person, handles),) = await queue.people.people()
    assert (person.name, set(handles)) == ("Ines", {"discord/ines", "email/ines@example.org"})
    body = await queue.memory.read("/memories/people/ines.md")
    assert body is not None and body.endswith("## Notes\n\nBrush pack.\n")


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
