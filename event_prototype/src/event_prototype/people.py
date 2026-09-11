"""Who the character knows.

A person is a name and a page: the root of their profile, at
`/memories/people/<slug>.md`, which the memory tool can read and edit like any
other. A person also has handles — a username on a venue — and a handle
belongs to one person, so an event's sender resolves to at most one profile.
That resolution is how a subagent is shown who it is talking to before it has
read a word of memory.
"""

from __future__ import annotations

import re
from collections.abc import Collection
from dataclasses import dataclass

from sqlalchemy import func, select, tuple_
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker
from sqlalchemy.orm import contains_eager

from .agents import Agent
from .memory import ROOT, head_of, heads, live_page, write_page
from .render import PersonView, PromptRenderer
from .store import MemoryEntryRow, MemoryVersionRow, PersonNameRow, PersonRow

PEOPLE_DIR = f"{ROOT}/people"

#: A handle as an event carries it: `(venue, username)`.
type Handle = tuple[str, str]


@dataclass(frozen=True, slots=True)
class Profile:
    """A known person as a prompt needs them: who, where their page is, what it says."""

    id: int
    name: str
    path: str
    #: The root page's current text; empty if the page has been deleted.
    body: str

    @property
    def view(self) -> PersonView:
        return PersonView(self.name, self.path, self.body)


def slugify(name: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")
    return slug or "person"


async def profiles_for(
    session: AsyncSession, handles: Collection[Handle]
) -> dict[Handle, Profile]:
    """The profiles behind these handles, one query however many there are.

    The joins are spelled out so the root page's head can ride along, and
    `contains_eager` tells the relationships to read from those same joins
    rather than adding their own.
    """
    if not handles:
        return {}
    newest = heads()
    query = (
        select(PersonNameRow, MemoryVersionRow.body)
        .join(PersonNameRow.person)
        .join(PersonRow.root)
        .join(newest, newest.c.entry_id == MemoryEntryRow.id)
        .join(MemoryVersionRow, MemoryVersionRow.id == newest.c.id)
        .options(contains_eager(PersonNameRow.person).contains_eager(PersonRow.root))
        .where(tuple_(PersonNameRow.venue, PersonNameRow.username).in_(list(handles)))
    )
    return {
        (name.venue, name.username): Profile(
            name.person.id, name.person.name, name.person.root.path, body or ""
        )
        for name, body in (await session.execute(query)).tuples().all()
    }


class PeopleStore:
    """People and their handles, over the queue's sessions."""

    def __init__(self, sessions: async_sessionmaker[AsyncSession], renderer: PromptRenderer) -> None:
        self._sessions = sessions
        self._renderer = renderer

    async def link(self, venue: str, username: str, name: str, agent: Agent | None) -> Profile:
        """Say that `username` on `venue` is `name`.

        A person by that name is found or made; making one also makes their
        root page, unless a page at the slug's path already exists and belongs
        to nobody, in which case that page becomes the profile — the model may
        well have written it before saying whose it is. A handle already bound
        to someone else is refused, since a handle has one owner.
        """
        async with self._sessions.begin() as session:
            bound = await session.scalar(
                select(PersonNameRow).where(
                    PersonNameRow.venue == venue, PersonNameRow.username == username
                )
            )
            if bound is not None:
                if bound.person.name != name:
                    raise ValueError(
                        f"{venue}/{username} is already {bound.person.name}'s handle "
                        f"(profile at {bound.person.root.path}); it cannot also be {name}'s."
                    )
                person = bound.person
            else:
                person = await session.scalar(select(PersonRow).where(PersonRow.name == name))
                if person is None:
                    person = PersonRow(
                        name=name,
                        root=await self._root_page(session, name, venue, username, agent),
                        created_by=agent.id if agent else None,
                    )
                    session.add(person)
                    await session.flush()
                session.add(
                    PersonNameRow(
                        person_id=person.id,
                        venue=venue,
                        username=username,
                        created_by=agent.id if agent else None,
                    )
                )
            page = await live_page(session, person.root.path)
            body = page[1].body if page is not None else None
            return Profile(person.id, person.name, person.root.path, body or "")

    async def _root_page(
        self, session: AsyncSession, name: str, venue: str, username: str, agent: Agent | None
    ) -> MemoryEntryRow:
        """The entry for a new person's profile: adopted if free, else written fresh."""
        owned = select(PersonRow.root_entry_id)
        slug = slugify(name)
        for attempt in range(1, 100):
            path = f"{PEOPLE_DIR}/{slug}{'' if attempt == 1 else f'-{attempt}'}.md"
            found = await head_of(session, path)
            if found is not None and found[0].id in (await session.scalars(owned)).all():
                continue  # someone else's profile happens to sit at this slug
            if found is not None and found[1].body is not None:
                return found[0]
            entry, _ = await write_page(
                session,
                path,
                self._renderer.person_root(name, venue, username),
                agent,
                {"command": "create"},
            )
            return entry
        raise RuntimeError(f"no free profile path for {name!r}")

    async def people(self) -> list[tuple[PersonRow, list[str]]]:
        """Everyone, with their handles as `venue/username`, by name. One query."""
        handles = func.group_concat(PersonNameRow.venue + "/" + PersonNameRow.username, ", ")
        query = (
            select(PersonRow, handles)
            .outerjoin(PersonNameRow, PersonNameRow.person_id == PersonRow.id)
            .group_by(PersonRow.id)
            .order_by(PersonRow.name)
        )
        async with self._sessions() as session:
            rows = (await session.execute(query)).tuples().all()
        return [(person, joined.split(", ") if joined else []) for person, joined in rows]
