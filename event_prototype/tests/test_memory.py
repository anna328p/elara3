"""Memory: pages as versions, tombstones, and the tool's filesystem view of them."""

from __future__ import annotations

import pytest
from anthropic.lib.tools import ToolError
from sqlalchemy.exc import IntegrityError

from event_prototype.agents import Agent, AgentRole
from event_prototype.memory import MemoryTool, normalize
from event_prototype.queue import EventQueue

from conftest import counting_selects

NOTES = "/memories/notes.md"


@pytest.fixture
async def writer(queue: EventQueue) -> Agent:
    return await queue.spawn(AgentRole.SUBAGENT)


def test_paths_stay_under_the_memory_root() -> None:
    assert normalize("/memories") == "/memories"
    assert normalize("/memories/people/") == "/memories/people"
    assert normalize("/memories/people/mira.md") == "/memories/people/mira.md"
    for escape in (
        "/etc/passwd",
        "memories/notes.md",
        "/memories/../secrets",
        "/memories/./notes.md",
        "/memories//notes.md",
    ):
        with pytest.raises(ToolError):
            normalize(escape)


async def test_a_page_is_its_newest_version(queue: EventQueue, writer: Agent) -> None:
    await queue.memory.create(NOTES, "one\ntwo\n", writer)
    await queue.memory.str_replace(NOTES, "two", "three", writer)
    await queue.memory.insert(NOTES, 0, "zero", writer)

    assert await queue.memory.read(NOTES) == "zero\none\nthree\n"
    versions = await queue.memory.history(NOTES)
    assert [v.row.body for v in versions] == ["one\ntwo\n", "one\nthree\n", "zero\none\nthree\n"]
    # Each version says how it came to differ from the one before.
    assert [v.row.edit_metadata for v in versions] == [
        {"command": "create"},
        {"command": "str_replace", "old_str": "two", "new_str": "three"},
        {"command": "insert", "insert_line": 0, "insert_text": "zero"},
    ]
    assert all(v.row.agent == writer and v.actor == writer.label for v in versions)
    assert len({v.row.entry_id for v in versions}) == 1


async def test_view_numbers_lines_and_honours_a_range(queue: EventQueue, writer: Agent) -> None:
    await queue.memory.create(NOTES, "a\nb\nc", writer)

    assert await queue.memory.view(NOTES) == (
        "Here's the content of /memories/notes.md with line numbers:\n"
        "     1\ta\n     2\tb\n     3\tc"
    )
    assert (await queue.memory.view(NOTES, [2, -1])).endswith("     2\tb\n     3\tc")
    with pytest.raises(ToolError, match="does not exist"):
        await queue.memory.view("/memories/missing.md")


async def test_the_listing_is_two_levels_deep_and_one_query(
    queue: EventQueue, writer: Agent
) -> None:
    for path, body in (
        ("/memories/MEMORY.md", "index"),
        ("/memories/people/mira.md", "mira!"),
        ("/memories/people/tobias.md", "tobias"),
        ("/memories/projects/showcase/plan.md", "x" * 2048),
    ):
        await queue.memory.create(path, body, writer)

    with counting_selects() as selects:
        listing = await queue.memory.listing()

    assert len(selects) == 1, selects
    assert listing.split("\n") == [
        "Here're the files and directories up to 2 levels deep in /memories, "
        "excluding hidden items and node_modules:",
        "2.0K\t/memories",
        "5B\t/memories/MEMORY.md",
        "11B\t/memories/people/",
        "5B\t/memories/people/mira.md",
        "6B\t/memories/people/tobias.md",
        "2K\t/memories/projects/",
        "2K\t/memories/projects/showcase/",
    ]
    # A directory views as its listing; a name that is neither is an error.
    assert (await queue.memory.view("/memories/people")).split("\n")[1:] == [
        "11B\t/memories/people",
        "5B\t/memories/people/mira.md",
        "6B\t/memories/people/tobias.md",
    ]
    with pytest.raises(ToolError, match="does not exist"):
        await queue.memory.view("/memories/nothing")
    # An empty store still lists its root.
    async with await EventQueue.open(":memory:") as empty:
        assert (await empty.memory.listing()).split("\n")[1:] == ["0B\t/memories"]


async def test_delete_is_a_tombstone_and_the_page_can_return(
    queue: EventQueue, writer: Agent
) -> None:
    await queue.memory.create(NOTES, "gone soon", writer)

    assert await queue.memory.delete(NOTES, writer) == "Successfully deleted /memories/notes.md"
    assert await queue.memory.read(NOTES) is None
    with pytest.raises(ToolError, match="does not exist"):
        await queue.memory.str_replace(NOTES, "gone", "here", writer)

    await queue.memory.create(NOTES, "back", writer)
    versions = await queue.memory.history(NOTES)
    assert [(v.row.edit_metadata["command"], v.row.body) for v in versions] == [
        ("create", "gone soon"),
        ("delete", None),
        ("create", "back"),
    ]
    assert len({v.row.entry_id for v in versions}) == 1  # one entry, one life
    with pytest.raises(ToolError, match="Cannot delete"):
        await queue.memory.delete("/memories", writer)


async def test_renaming_a_directory_moves_every_page_and_records_each(
    queue: EventQueue, writer: Agent
) -> None:
    await queue.memory.create("/memories/people/mira.md", "m", writer)
    await queue.memory.create("/memories/people/tobias.md", "t", writer)
    await queue.memory.create("/memories/people/old.md", "o", writer)
    await queue.memory.delete("/memories/people/old.md", writer)

    await queue.memory.rename("/memories/people", "/memories/folks", writer)

    assert await queue.memory.read("/memories/folks/mira.md") == "m"
    assert await queue.memory.read("/memories/people/mira.md") is None
    (moved,) = [
        v
        for v in await queue.memory.history("/memories/folks/tobias.md")
        if v.row.edit_metadata["command"] == "rename"
    ]
    assert moved.row.edit_metadata == {
        "command": "rename",
        "old_path": "/memories/people/tobias.md",
        "new_path": "/memories/folks/tobias.md",
    }
    assert moved.row.body == "t" and moved.row.agent == writer
    # The tombstoned page was not part of the directory, so it stayed put...
    assert [v.row.tombstone for v in await queue.memory.history("/memories/people/old.md")] == [False, True]
    # ...and its entry, dead or not, still refuses a move onto it.
    await queue.memory.create("/memories/scratch.md", "s", writer)
    with pytest.raises(ToolError, match="already exists"):
        await queue.memory.rename("/memories/scratch.md", "/memories/people/old.md", writer)
    with pytest.raises(ToolError, match="into itself"):
        await queue.memory.rename("/memories/folks", "/memories/folks/inner", writer)


async def test_str_replace_needs_exactly_one_match(queue: EventQueue, writer: Agent) -> None:
    await queue.memory.create(NOTES, "x\ny\nx\n", writer)

    with pytest.raises(ToolError, match="did not appear verbatim"):
        await queue.memory.str_replace(NOTES, "z", "w", writer)
    with pytest.raises(ToolError, match="lines: 1, 3"):
        await queue.memory.str_replace(NOTES, "x", "w", writer)
    assert len(await queue.memory.history(NOTES)) == 1  # nothing was written


async def test_insert_is_bounded_by_the_file(queue: EventQueue, writer: Agent) -> None:
    await queue.memory.create(NOTES, "a\nb", writer)

    with pytest.raises(ToolError, match=r"\[0, 2\]"):
        await queue.memory.insert(NOTES, 3, "c", writer)
    await queue.memory.insert(NOTES, 2, "c\n", writer)
    assert await queue.memory.read(NOTES) == "a\nb\nc"


async def test_a_name_is_a_file_or_a_directory_not_both(queue: EventQueue, writer: Agent) -> None:
    await queue.memory.create("/memories/people/mira.md", "m", writer)

    with pytest.raises(ToolError, match="is a directory"):
        await queue.memory.create("/memories/people", "?", writer)
    with pytest.raises(ToolError, match="is a file"):
        await queue.memory.create("/memories/people/mira.md/more.md", "?", writer)
    with pytest.raises(ToolError, match="already exists"):
        await queue.memory.create("/memories/people/mira.md", "again", writer)


async def test_every_version_names_its_agent_or_the_operator(
    queue: EventQueue, writer: Agent
) -> None:
    await queue.memory.put("/memories/MEMORY.md", "by hand", None)
    await queue.memory.str_replace("/memories/MEMORY.md", "hand", "agent", writer)

    versions = await queue.memory.history("/memories/MEMORY.md")
    assert [v.actor for v in versions] == ["operator", writer.label]
    assert versions[0].row.edit_metadata == {"command": "put"}
    # A NULL agent is the operator; a made-up one is refused by the foreign key.
    with pytest.raises(IntegrityError):
        await queue.memory.create("/memories/x.md", "?", Agent(AgentRole.SUBAGENT, "never-minted"))


async def test_the_tool_speaks_for_one_agent(queue: EventQueue, writer: Agent) -> None:
    tool = MemoryTool(queue.memory, writer)
    assert tool.to_dict() == {"type": "memory_20250818", "name": "memory"}

    created = await tool.call({"command": "create", "path": "/memories/a.md", "file_text": "hi"})
    assert created == "File created successfully at: /memories/a.md"
    listed = await tool.call({"command": "view", "path": "/memories"})
    assert isinstance(listed, str) and listed.endswith("2B\t/memories/a.md")
    # `new_str` may be left out, meaning delete the match.
    edited = await tool.call({"command": "str_replace", "path": "/memories/a.md", "old_str": "hi"})
    assert isinstance(edited, str) and edited.startswith("The memory file has been edited")
    assert await queue.memory.read("/memories/a.md") == ""
    assert [v.row.agent for v in await queue.memory.history("/memories/a.md")] == [writer, writer]
