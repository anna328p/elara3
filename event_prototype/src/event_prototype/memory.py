"""The character's memory: a wiki of markdown pages, shown to the model as a
directory of files through Claude's memory tool.

A page is an entry (its path) and a run of versions (its text after each
operation). The head version is the page; a head with no body is a tombstone,
and the page does not exist right now. Directories are not stored: they are
whatever the live paths imply, derived in SQL when a listing is asked for.

`MemoryStore` reads and writes pages over the queue's sessions, one transaction
per operation, and attributes every version to an agent or, with `None`, to the
operator. `MemoryTool` is the memory tool's six commands over it, returning the
strings the tool's documentation describes, so the model sees what it expects.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from anthropic.lib.tools import BetaAsyncAbstractMemoryTool, ToolError
from anthropic.types.beta import (
    BetaMemoryTool20250818CreateCommand,
    BetaMemoryTool20250818DeleteCommand,
    BetaMemoryTool20250818InsertCommand,
    BetaMemoryTool20250818RenameCommand,
    BetaMemoryTool20250818StrReplaceCommand,
    BetaMemoryTool20250818ViewCommand,
)
from sqlalchemy import Select, Subquery, case, func, literal, null, or_, select, union_all
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from .agents import Agent
from .store import MemoryEntryRow, MemoryVersionRow

ROOT = "/memories"
LINE_NUMBER_WIDTH = 6
#: What the tool's own description promises: longer views are cut, and the
#: model is told to page with `view_range`.
MAX_VIEW_CHARS = 16_000

#: An entry with its head version. The head may be a tombstone.
type Head = tuple[MemoryEntryRow, MemoryVersionRow]
#: Something the tool commands say about an edit, stored beside the version.
type EditMetadata = dict[str, Any]


def normalize(path: str) -> str:
    """`path` as the store spells it, or a `ToolError` saying why it cannot be.

    Everything lives under `/memories`; the check on each segment is what keeps
    `..` from meaning anything. A trailing slash is tolerated, since that is
    how a directory is often written.
    """
    if path != ROOT and not path.startswith(ROOT + "/"):
        raise ToolError(f"Path must start with {ROOT}, got: {path}")
    rest = path[len(ROOT) :].rstrip("/")  # "" for the root, "/a/b" otherwise
    segments = rest.split("/")[1:] if rest else []
    if any(segment in ("", ".", "..") for segment in segments):
        raise ToolError(f"Path {path} would escape {ROOT} directory")
    return "/".join([ROOT, *segments])


def ancestors(path: str) -> list[str]:
    """The directories above `path`, nearest first, stopping short of the root."""
    found: list[str] = []
    while (cut := path.rfind("/")) > len(ROOT):
        path = path[:cut]
        found.append(path)
    return found


# -- queries -------------------------------------------------------------


def heads() -> Subquery:
    """`(entry_id, id)` of every entry's newest version: what makes a page current."""
    return (
        select(
            MemoryVersionRow.entry_id.label("entry_id"),
            func.max(MemoryVersionRow.id).label("id"),
        )
        .group_by(MemoryVersionRow.entry_id)
        .subquery("heads")
    )


def _head_query() -> Select[tuple[MemoryEntryRow, MemoryVersionRow]]:
    newest = heads()
    return (
        select(MemoryEntryRow, MemoryVersionRow)
        .join(newest, newest.c.entry_id == MemoryEntryRow.id)
        .join(MemoryVersionRow, MemoryVersionRow.id == newest.c.id)
    )


def _under(prefix: str):
    """Paths inside the directory `prefix`, at any depth."""
    return MemoryEntryRow.path.startswith(prefix + "/", autoescape=True)


async def head_of(session: AsyncSession, path: str) -> Head | None:
    """The entry at `path` and its head, tombstone or not; None if no entry ever was."""
    query = _head_query().where(MemoryEntryRow.path == path)
    return (await session.execute(query)).tuples().first()


async def live_page(session: AsyncSession, path: str) -> Head | None:
    """The page at `path`, if one exists right now."""
    found = await head_of(session, path)
    return found if found is not None and found[1].body is not None else None


async def live_under(session: AsyncSession, prefix: str) -> list[Head]:
    """Every page inside the directory `prefix`, by path."""
    query = (
        _head_query()
        .where(_under(prefix), MemoryVersionRow.body.is_not(None))
        .order_by(MemoryEntryRow.path)
    )
    return list((await session.execute(query)).tuples().all())


def _listing_query(prefix: str) -> Select[tuple[str | None, str | None, int | None, int]]:
    """The two-level listing of `prefix`, as SQL.

    `live` is every page under the prefix with its size; `split` cuts each
    path, relative to the prefix, into its first segment and the first segment
    of what follows, noting when more path lies beyond either. Three grouped
    selects — the whole directory, each first-level name, each second-level
    name — are unioned into one result, and NULLs sorting first in SQLite is
    what puts the total ahead of its entries and each directory ahead of its
    contents.
    """
    newest = heads()
    live = (
        select(
            MemoryEntryRow.path.label("path"),
            func.length(MemoryVersionRow.body).label("size"),
        )
        .join(newest, newest.c.entry_id == MemoryEntryRow.id)
        .join(MemoryVersionRow, MemoryVersionRow.id == newest.c.id)
        .where(_under(prefix), MemoryVersionRow.body.is_not(None))
        .cte("live")
    )
    rest = func.substr(live.c.path, len(prefix) + 2)
    cut1 = func.instr(rest, "/")
    first = case((cut1 == 0, rest), else_=func.substr(rest, 1, cut1 - 1))
    tail = func.substr(rest, cut1 + 1)  # only meaningful when cut1 > 0
    cut2 = func.instr(tail, "/")
    second = case((cut1 == 0, null()), (cut2 == 0, tail), else_=func.substr(tail, 1, cut2 - 1))
    split = select(
        first.label("first"),
        second.label("second"),
        case((cut1 == 0, 0), else_=1).label("first_is_dir"),
        case((cut1 == 0, 0), (cut2 == 0, 0), else_=1).label("second_is_dir"),
        live.c.size.label("size"),
    ).cte("split")

    total = select(
        null().label("first"),
        null().label("second"),
        func.sum(split.c.size).label("size"),
        literal(1).label("is_dir"),
    )
    level1 = select(
        split.c.first,
        null().label("second"),
        func.sum(split.c.size),
        func.max(split.c.first_is_dir),
    ).group_by(split.c.first)
    level2 = (
        select(split.c.first, split.c.second, func.sum(split.c.size), func.max(split.c.second_is_dir))
        .where(split.c.second.is_not(None))
        .group_by(split.c.first, split.c.second)
    )
    listing = union_all(total, level1, level2).subquery("listing")
    return select(listing).order_by(listing.c.first, listing.c.second)


# -- writing -------------------------------------------------------------


def _version(
    session: AsyncSession,
    entry: MemoryEntryRow,
    agent: Agent | None,
    body: str | None,
    metadata: EditMetadata,
) -> MemoryVersionRow:
    row = MemoryVersionRow(
        entry_id=entry.id,
        agent_id=agent.id if agent else None,
        body=body,
        edit_metadata=metadata,
    )
    session.add(row)
    return row


async def write_page(
    session: AsyncSession,
    path: str,
    body: str,
    agent: Agent | None,
    metadata: EditMetadata,
) -> Head:
    """Give `path` a body: a new entry if none was ever there, else a version on the one there.

    Rejects a path that is a directory (has pages beneath it) or lies beneath
    a page, since the model reads memory as a filesystem and one name cannot
    be both. Whether a page already exists at `path` is the caller's business.
    """
    conflicts = await _neighbours(session, path)
    for entry, head in conflicts:
        if head.body is None or entry.path == path:
            continue
        if entry.path.startswith(path + "/"):
            raise ToolError(f"Error: {path} is a directory")
        raise ToolError(f"Error: {entry.path} is a file, so nothing can be created under it")
    found = next((entry for entry, _ in conflicts if entry.path == path), None)
    if found is None:
        found = MemoryEntryRow(path=path)
        session.add(found)
        await session.flush()  # so the version can point at it
    return found, _version(session, found, agent, body, metadata)


async def _neighbours(session: AsyncSession, path: str) -> list[Head]:
    """The entry at `path`, everything under it, and its ancestors, in one query."""
    query = _head_query().where(
        or_(MemoryEntryRow.path == path, _under(path), MemoryEntryRow.path.in_(ancestors(path)))
    )
    return list((await session.execute(query)).tuples().all())


# -- formatting ----------------------------------------------------------


def _human_size(size: int | None) -> str:
    if not size:
        return "0B"
    units = ("B", "K", "M", "G")
    index = min((size.bit_length() - 1) // 10, len(units) - 1)
    scaled = size / (1024**index)
    return f"{int(scaled)}{units[index]}" if scaled == int(scaled) else f"{scaled:.1f}{units[index]}"


def _numbered(lines: Sequence[str], start: int = 1) -> str:
    return "\n".join(
        f"{str(number).rjust(LINE_NUMBER_WIDTH)}\t{line}"
        for number, line in enumerate(lines, start)
    )


def _viewed(path: str, body: str, view_range: Sequence[int] | None) -> str:
    """A page as the model reads it: numbered lines, a range if it asked, a cap otherwise."""
    lines = body.split("\n")
    start = 1
    if view_range is not None and len(view_range) == 2:
        first, last = view_range
        start = max(1, first)
        end = len(lines) if last == -1 else last
        lines = lines[start - 1 : end]
    header = f"Here's the content of {path} with line numbers:\n"
    text = _numbered(lines, start)
    if view_range is None and len(text) > MAX_VIEW_CHARS:
        kept = text[:MAX_VIEW_CHARS].rsplit("\n", 1)[0]
        shown = kept.count("\n") + 1
        text = (
            f"{kept}\n... (truncated after line {shown} of {len(lines)}; "
            "use view_range to read the rest)"
        )
    return header + text


def _format_listing(prefix: str, rows: Sequence[tuple[str | None, str | None, int | None, int]]) -> str:
    header = (
        f"Here're the files and directories up to 2 levels deep in {prefix}, "
        "excluding hidden items and node_modules:"
    )
    lines: list[str] = []
    for first, second, size, is_dir in rows:
        shown = "/".join(part for part in (prefix, first, second) if part is not None)
        if is_dir and shown != prefix:
            shown += "/"
        lines.append(f"{_human_size(size)}\t{shown}")
    return "\n".join([header, *lines])


def _occurrences(text: str, needle: str) -> list[int]:
    """The 1-based line of every occurrence of `needle`."""
    found: list[int] = []
    at = text.find(needle)
    while at != -1:
        found.append(text.count("\n", 0, at) + 1)
        at = text.find(needle, at + 1)
    return found


# -- the store -------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Version:
    """One version of a page as the CLI shows it: when, by whom, what changed."""

    row: MemoryVersionRow

    @property
    def actor(self) -> str:
        return self.row.agent.label if self.row.agent else "operator"


class MemoryStore:
    """Pages, read and written one transaction at a time."""

    def __init__(self, sessions: async_sessionmaker[AsyncSession]) -> None:
        self._sessions = sessions

    # -- reading ---------------------------------------------------------

    async def read(self, path: str) -> str | None:
        """The page's text, or None if there is no page at `path` right now."""
        path = normalize(path)
        async with self._sessions() as session:
            page = await live_page(session, path)
        return None if page is None else page[1].body

    async def view(self, path: str, view_range: Sequence[int] | None = None) -> str:
        """A page, numbered; or a directory, listed; or the error the tool expects."""
        path = normalize(path)
        async with self._sessions() as session:
            page = await live_page(session, path)
            if page is not None:
                assert page[1].body is not None  # that is what live means
                return _viewed(path, page[1].body, view_range)
            rows = list((await session.execute(_listing_query(path))).tuples().all())
        if path != ROOT and len(rows) == 1:  # nothing but the total: no such directory
            raise ToolError(f"The path {path} does not exist. Please provide a valid path.")
        return _format_listing(path, rows)

    async def listing(self, prefix: str = ROOT) -> str:
        """The directory `prefix`, two levels deep. One query."""
        prefix = normalize(prefix)
        async with self._sessions() as session:
            rows = list((await session.execute(_listing_query(prefix))).tuples().all())
        return _format_listing(prefix, rows)

    async def history(self, path: str) -> list[Version]:
        """Every version of the entry at `path`, oldest first, tombstones included."""
        path = normalize(path)
        query = (
            select(MemoryVersionRow)
            .join(MemoryEntryRow, MemoryEntryRow.id == MemoryVersionRow.entry_id)
            .where(MemoryEntryRow.path == path)
            .order_by(MemoryVersionRow.id)
        )
        async with self._sessions() as session:
            return [Version(row) for row in (await session.scalars(query)).all()]

    # -- writing ---------------------------------------------------------

    async def create(self, path: str, file_text: str, agent: Agent | None) -> str:
        """A new page. Refuses a live one: the model can edit or delete that first."""
        path = normalize(path)
        if path == ROOT:
            raise ToolError(f"Error: {ROOT} is the memory directory itself")
        async with self._sessions.begin() as session:
            if await live_page(session, path) is not None:
                raise ToolError(f"Error: File {path} already exists")
            await write_page(session, path, file_text, agent, {"command": "create"})
        return f"File created successfully at: {path}"

    async def put(self, path: str, body: str, agent: Agent | None) -> str:
        """The whole page, written or replaced: the operator's way in from outside."""
        path = normalize(path)
        if path == ROOT:
            raise ToolError(f"Error: {ROOT} is the memory directory itself")
        async with self._sessions.begin() as session:
            existed = await live_page(session, path) is not None
            await write_page(session, path, body, agent, {"command": "put"})
        return f"{'Replaced' if existed else 'Created'} {path}"

    async def str_replace(self, path: str, old_str: str, new_str: str, agent: Agent | None) -> str:
        path = normalize(path)
        async with self._sessions.begin() as session:
            page = await self._require(session, path)
            entry, head = page
            assert head.body is not None
            lines = _occurrences(head.body, old_str)
            if not lines:
                raise ToolError(
                    f"No replacement was performed, old_str `{old_str}` did not appear "
                    f"verbatim in {path}."
                )
            if len(lines) > 1:
                listed = ", ".join(str(line) for line in lines)
                raise ToolError(
                    f"No replacement was performed. Multiple occurrences of old_str "
                    f"`{old_str}` in lines: {listed}. Please ensure it is unique"
                )
            body = head.body.replace(old_str, new_str, 1)
            _version(
                session, entry, agent, body,
                {"command": "str_replace", "old_str": old_str, "new_str": new_str},
            )
        # The doc's snippet: the changed line with a couple either side.
        new_lines = body.split("\n")
        changed = lines[0] - 1
        first, last = max(0, changed - 2), min(len(new_lines), changed + 3)
        return (
            "The memory file has been edited. Here is the snippet showing the change "
            f"(with line numbers):\n{_numbered(new_lines[first:last], first + 1)}"
        )

    async def insert(self, path: str, insert_line: int, insert_text: str, agent: Agent | None) -> str:
        path = normalize(path)
        async with self._sessions.begin() as session:
            entry, head = await self._require(session, path)
            assert head.body is not None
            lines = head.body.split("\n")
            if insert_line < 0 or insert_line > len(lines):
                raise ToolError(
                    f"Error: Invalid `insert_line` parameter: {insert_line}. It should be "
                    f"within the range of lines of the file: [0, {len(lines)}]"
                )
            lines.insert(insert_line, insert_text.rstrip("\n"))
            _version(
                session, entry, agent, "\n".join(lines),
                {"command": "insert", "insert_line": insert_line, "insert_text": insert_text},
            )
        return f"The file {path} has been edited."

    async def delete(self, path: str, agent: Agent | None) -> str:
        """A tombstone on the page, or on every page under the directory."""
        path = normalize(path)
        if path == ROOT:
            raise ToolError(f"Error: Cannot delete the {ROOT} directory itself")
        async with self._sessions.begin() as session:
            page = await live_page(session, path)
            pages = [page] if page is not None else await live_under(session, path)
            if not pages:
                raise ToolError(f"Error: The path {path} does not exist")
            for entry, _ in pages:
                _version(session, entry, agent, None, {"command": "delete"})
        return f"Successfully deleted {path}"

    async def rename(self, old_path: str, new_path: str, agent: Agent | None) -> str:
        """Move a page, or every page under a directory, keeping each one's history.

        The entry's path changes and a version records the move, so the old
        name is still in the trail. A destination that has an entry at all —
        alive or tombstoned — is refused: two entries cannot share a path, and
        merging their histories is not a thing a rename should do.
        """
        old_path, new_path = normalize(old_path), normalize(new_path)
        if ROOT in (old_path, new_path):
            raise ToolError(f"Error: Cannot rename the {ROOT} directory itself")
        if old_path == new_path or new_path.startswith(old_path + "/"):
            raise ToolError(f"Error: Cannot move {old_path} into itself")
        async with self._sessions.begin() as session:
            page = await live_page(session, old_path)
            pages = [page] if page is not None else await live_under(session, old_path)
            if not pages:
                raise ToolError(f"Error: The path {old_path} does not exist")
            moves = {entry.path: new_path + entry.path[len(old_path) :] for entry, _ in pages}
            taken = await session.scalars(
                select(MemoryEntryRow.path).where(MemoryEntryRow.path.in_([new_path, *moves.values()]))
            )
            if clash := taken.first():
                raise ToolError(f"Error: The destination {clash} already exists")
            for entry, head in pages:
                moved = moves[entry.path]
                metadata = {"command": "rename", "old_path": entry.path, "new_path": moved}
                entry.path = moved
                _version(session, entry, agent, head.body, metadata)
        return f"Successfully renamed {old_path} to {new_path}"

    @staticmethod
    async def _require(session: AsyncSession, path: str) -> Head:
        page = await live_page(session, path)
        if page is None:
            raise ToolError(f"Error: The path {path} does not exist. Please provide a valid path.")
        return page


# -- the tool --------------------------------------------------------------


class MemoryTool(BetaAsyncAbstractMemoryTool):
    """Claude's memory tool over the store, every edit attributed to one agent."""

    def __init__(self, store: MemoryStore, agent: Agent) -> None:
        super().__init__()
        self._store = store
        self._agent = agent

    async def view(self, command: BetaMemoryTool20250818ViewCommand) -> str:
        return await self._store.view(command.path, command.view_range)

    async def create(self, command: BetaMemoryTool20250818CreateCommand) -> str:
        return await self._store.create(command.path, command.file_text, self._agent)

    async def str_replace(self, command: BetaMemoryTool20250818StrReplaceCommand) -> str:
        # The doc allows `new_str` to be absent, meaning delete the match.
        new_str: str = getattr(command, "new_str", None) or ""
        return await self._store.str_replace(command.path, command.old_str, new_str, self._agent)

    async def insert(self, command: BetaMemoryTool20250818InsertCommand) -> str:
        return await self._store.insert(
            command.path, command.insert_line, command.insert_text, self._agent
        )

    async def delete(self, command: BetaMemoryTool20250818DeleteCommand) -> str:
        return await self._store.delete(command.path, self._agent)

    async def rename(self, command: BetaMemoryTool20250818RenameCommand) -> str:
        return await self._store.rename(command.old_path, command.new_path, self._agent)
