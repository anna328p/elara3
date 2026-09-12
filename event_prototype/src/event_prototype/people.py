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

    async def register(
        self, name: str, notes: str, handle: Handle | None, agent: Agent | None
    ) -> Profile:
        """Bring a new person into memory: a profile page scaffolded from `notes`,
        and a handle bound to them if one is known.

        A name already known is refused, and the message names the page, since
        the thing to do then is edit it or link another handle, not make a twin.
        """
        async with self._sessions.begin() as session:
            known = await session.scalar(select(PersonRow).where(PersonRow.name == name))
            if known is not None:
                raise ValueError(
                    f"{name} is already known; their profile is at {known.root.path}. "
                    "Edit that page, or use link_person to add a handle."
                )
            if handle is not None:
                await self._require_free(session, handle, name)
            entry = await self._root_page(session, name, handle, notes, agent)
            person = PersonRow(name=name, root=entry, created_by=agent.id if agent else None)
            session.add(person)
            await session.flush()
            if handle is not None:
                self._bind(session, person, handle, agent)
            return await self._profile(session, person)

    async def link(self, venue: str, username: str, name: str, agent: Agent | None) -> Profile:
        """Say that `username` on `venue` is `name`.

        A person by that name is found or made; making one also makes their
        root page. A handle already bound to someone else is refused, since a
        handle has one owner; one already bound to this person is left alone.
        """
        handle = (venue, username)
        async with self._sessions.begin() as session:
            bound = await self._require_free(session, handle, name)
            if bound is not None:
                return await self._profile(session, bound)
            person = await session.scalar(select(PersonRow).where(PersonRow.name == name))
            if person is None:
                person = PersonRow(
                    name=name,
                    root=await self._root_page(session, name, handle, "", agent),
                    created_by=agent.id if agent else None,
                )
                session.add(person)
                await session.flush()
            self._bind(session, person, handle, agent)
            return await self._profile(session, person)

    @staticmethod
    async def _require_free(
        session: AsyncSession, handle: Handle, name: str
    ) -> PersonRow | None:
        """Whoever holds `handle` if it is `name`; None if nobody does; refused otherwise."""
        venue, username = handle
        bound = await session.scalar(
            select(PersonNameRow).where(
                PersonNameRow.venue == venue, PersonNameRow.username == username
            )
        )
        if bound is None:
            return None
        if bound.person.name != name:
            raise ValueError(
                f"{venue}/{username} is already {bound.person.name}'s handle "
                f"(profile at {bound.person.root.path}); it cannot also be {name}'s."
            )
        return bound.person

    @staticmethod
    def _bind(
        session: AsyncSession, person: PersonRow, handle: Handle, agent: Agent | None
    ) -> None:
        venue, username = handle
        session.add(
            PersonNameRow(
                person_id=person.id,
                venue=venue,
                username=username,
                created_by=agent.id if agent else None,
            )
        )

    @staticmethod
    async def _profile(session: AsyncSession, person: PersonRow) -> Profile:
        page = await live_page(session, person.root.path)
        body = page[1].body if page is not None else None
        return Profile(person.id, person.name, person.root.path, body or "")

    async def _root_page(
        self,
        session: AsyncSession,
        name: str,
        handle: Handle | None,
        notes: str,
        agent: Agent | None,
    ) -> MemoryEntryRow:
        """The entry for a new person's profile: adopted if free, else written fresh.

        A page at the slug's path that belongs to nobody becomes the profile —
        the model may well have written it before saying whose it is — unless
        starter notes were given, which that page would silently lose; then the
        caller is told to edit the page instead.
        """
        owned = select(PersonRow.root_entry_id)
        slug = slugify(name)
        for attempt in range(1, 100):
            path = f"{PEOPLE_DIR}/{slug}{'' if attempt == 1 else f'-{attempt}'}.md"
            found = await head_of(session, path)
            if found is not None and found[0].id in (await session.scalars(owned)).all():
                continue  # someone else's profile happens to sit at this slug
            if found is not None and found[1].body is not None:
                if notes:
                    raise ValueError(
                        f"A page already exists at {path}; register {name} without "
                        "starter notes to make it their profile, then edit it."
                    )
                return found[0]
            entry, _ = await write_page(
                session,
                path,
                self._renderer.person_root(
                    name, handle="/".join(handle) if handle else None, notes=notes
                ),
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
