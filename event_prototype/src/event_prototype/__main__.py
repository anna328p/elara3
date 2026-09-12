"""Command line for the prototype: seed the queue, look at it, triage it, watch
it, and read or write the character's memory as the operator."""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
import textwrap
from datetime import timedelta
from pathlib import Path
from typing import TYPE_CHECKING

from anthropic.lib.tools import ToolError

from .agents import LOOP_ROLES, AgentRole
from .config import DEFAULT_CONFIG_PATH, Config
from .contexts import Block
from .fixtures import seed
from .heartbeats import Beat
from .memory import ROOT
from .queue import EventQueue
from .render import PromptRenderer
from .store import utcnow

if TYPE_CHECKING:
    from .runner import PassResult
    from .subagent import Spend


def main() -> None:
    parser = argparse.ArgumentParser(prog="event_prototype", description=__doc__)
    parser.add_argument(
        "--config", type=Path, default=DEFAULT_CONFIG_PATH, help="path to config.toml"
    )
    sub = parser.add_subparsers(dest="command", required=True)

    seed_cmd = sub.add_parser("seed", help="populate the queue with synthetic events")
    seed_cmd.add_argument(
        "--fresh", action="store_true", help="delete the existing database first"
    )

    list_cmd = sub.add_parser("list", help="show the queue")
    list_cmd.add_argument(
        "--all", action="store_true", help="include completed and archived events"
    )

    sub.add_parser("streams", help="show the ongoing loci events belong to")

    context_cmd = sub.add_parser("context", help="show the conversation a stream routes to")
    context_cmd.add_argument("stream_id", type=int, help="the stream's id, from `streams`")

    for name, help_text in (
        ("triage", "decide what the pending events need"),
        ("sweep", "reconsider the deferred backlog"),
    ):
        pass_cmd = sub.add_parser(name, help=help_text)
        pass_cmd.add_argument(
            "--dry-run",
            action="store_true",
            help="print the prompt without calling the API",
        )

    sub.add_parser(
        "watch",
        help="run each role's pass on its heartbeat, and triage on a nudge or an active arrival, until ^C",
    )
    sub.add_parser("heartbeats", help="show the live schedules and when each next fires")

    memory_cmd = sub.add_parser("memory", help="read and write the character's memory")
    memory_sub = memory_cmd.add_subparsers(dest="memory_command", required=True)
    ls_cmd = memory_sub.add_parser("ls", help="list a directory, as the memory tool shows it")
    ls_cmd.add_argument("prefix", nargs="?", default=ROOT, help=f"a directory under {ROOT}")
    cat_cmd = memory_sub.add_parser("cat", help="print a page")
    cat_cmd.add_argument("path")
    log_cmd = memory_sub.add_parser("log", help="every version of a page, oldest first")
    log_cmd.add_argument("path")
    put_cmd = memory_sub.add_parser(
        "put", help="write a whole page from a file or stdin, as the operator"
    )
    put_cmd.add_argument("path")
    put_cmd.add_argument("file", nargs="?", type=Path, help="defaults to stdin")

    sub.add_parser("people", help="everyone the character knows, with their handles")

    register_cmd = sub.add_parser("register", help="bring a new person into memory")
    register_cmd.add_argument("name", help="the person, as their profile will be titled")
    register_cmd.add_argument("--notes", default="", help="what is known so far")
    register_cmd.add_argument(
        "--handle", metavar="VENUE/USERNAME", help='a handle of theirs: "discord/ines"'
    )

    link_cmd = sub.add_parser("link", help="say which handle belongs to whom")
    link_cmd.add_argument("venue", help='the medium, as events spell it: "discord"')
    link_cmd.add_argument("username", help="the handle, as events spell the sender")
    link_cmd.add_argument("name", help="the person, as their profile is titled")

    args = parser.parse_args()
    config = Config.load(args.config)
    # The MCP server turns on INFO logging, which makes httpx narrate every
    # request over the report we are trying to print.
    logging.getLogger("httpx").setLevel(logging.WARNING)
    try:
        asyncio.run(_dispatch(args, config))
    except KeyboardInterrupt:
        # ^C cancels the main task and everything unwinds before this is
        # re-raised; nothing inside catches CancelledError.
        print("\nStopped.")
    except (ToolError, ValueError) as exc:
        # The memory tool's errors are written for the model; they read fine here too.
        raise SystemExit(f"error: {exc}") from None


async def _dispatch(args: argparse.Namespace, config: Config) -> None:
    match args.command:
        case "seed":
            await _seed(config, fresh=args.fresh)
        case "list":
            await _list(config, show_all=args.all)
        case "streams":
            await _streams(config)
        case "context":
            await _context(config, args.stream_id)
        case "triage" | "sweep" as which:
            await _pass(config, which, dry_run=args.dry_run)
        case "watch":
            await _watch(config)
        case "memory":
            await _memory(config, args)
        case "people":
            await _people(config)
        case "register":
            await _register(config, args.name, args.notes, args.handle)
        case "link":
            await _link(config, args.venue, args.username, args.name)
        case "heartbeats":
            await _heartbeats(config)
        case unknown:  # argparse rejects anything else first
            raise AssertionError(f"unhandled command: {unknown}")


async def _seed(config: Config, *, fresh: bool) -> None:
    if fresh:
        config.db_path.unlink(missing_ok=True)
    async with await EventQueue.open(config.db_path) as queue:
        ids = await seed(queue)
    print(f"Queued {len(ids)} events in {config.db_path}: {ids[0]}–{ids[-1]}")


async def _list(config: Config, *, show_all: bool) -> None:
    async with await EventQueue.open(config.db_path) as queue:
        rows = await queue.list_events(active_only=not show_all)
        # One query for the whole slice of actions, not one per event.
        history = await queue.history_for([row.id for row in rows])

    if not rows:
        print("Queue is empty." if show_all else "No active events.")
        return

    for row in rows:
        stream = row.stream.title if row.stream else "—"
        print(
            f"[{row.id:>3}] {row.priority.name.lower():<10} {row.kind:<9} "
            f"{row.status.value:<9} {stream:<22} {row.description}"
        )
        for entry in history.get(row.id, []):
            stamp = entry.timestamp.isoformat(timespec="seconds")
            print(f"        {stamp} {entry.action.value} by {entry.agent.label}")
            print(textwrap.indent(textwrap.fill(entry.detail, 80), " " * 12))


async def _streams(config: Config) -> None:
    async with await EventQueue.open(config.db_path) as queue:
        summaries = await queue.list_streams()

    if not summaries:
        print("No streams yet.")
        return

    for summary in summaries:
        last = summary.last_event_at
        seen = last.isoformat(timespec="seconds") if last else "never"
        context = summary.stream.context_id
        routed = (
            f"context {context} ({_plural(summary.turns, 'turn')})"
            if context is not None
            else "no context"
        )
        print(
            f"[{summary.stream.id:>3}] {summary.stream.kind.value:<8} "
            f"{summary.stream.title:<26} {summary.active:>2} active   last {seen}   {routed}"
        )


async def _context(config: Config, stream_id: int) -> None:
    async with await EventQueue.open(config.db_path) as queue:
        stream = await queue.get_stream(stream_id)
        if stream.context_id is None:
            print(f"{stream.title} has no context yet: nothing has been assigned in it.")
            return
        turns = await queue.transcript(stream.context_id)

    print(f"{stream.title} → context {stream.context_id}, {_plural(len(turns), 'turn')}\n")
    for turn in turns:
        stamp = turn.timestamp.isoformat(timespec="seconds")
        about = f"  (event {turn.event_id})" if turn.event_id is not None else ""
        print(f"--- {turn.role.value} {stamp}{about}")
        for block in turn.content:
            print(textwrap.indent(_show_block(block), "    "))
        print()


async def _memory(config: Config, args: argparse.Namespace) -> None:
    """Memory as the operator sees it. Writes carry no agent: NULL is the operator."""
    async with await EventQueue.open(config.db_path) as queue:
        match args.memory_command:
            case "ls":
                print(await queue.memory.listing(args.prefix))
            case "cat":
                body = await queue.memory.read(args.path)
                if body is None:
                    raise SystemExit(f"error: no page at {args.path}")
                print(body, end="" if body.endswith("\n") else "\n")
            case "log":
                versions = await queue.memory.history(args.path)
                if not versions:
                    raise SystemExit(f"error: nothing has ever been at {args.path}")
                for version in versions:
                    row = version.row
                    stamp = row.timestamp.isoformat(timespec="seconds")
                    state = "tombstone" if row.body is None else f"{len(row.body)} chars"
                    what = json.dumps(row.edit_metadata, ensure_ascii=False)
                    print(f"[{row.id:>4}] {stamp}  {version.actor:<18} {state:<12} {what}")
            case "put":
                body = args.file.read_text() if args.file else sys.stdin.read()
                print(await queue.memory.put(args.path, body, None))
            case unknown:
                raise AssertionError(f"unhandled memory command: {unknown}")


async def _people(config: Config) -> None:
    async with await EventQueue.open(config.db_path) as queue:
        people = await queue.people.people()
    if not people:
        print("Nobody known yet.")
        return
    for person, handles in people:
        print(
            f"[{person.id:>3}] {person.name:<20} {person.root.path:<36} "
            f"{', '.join(handles) or '—'}"
        )


async def _register(config: Config, name: str, notes: str, handle: str | None) -> None:
    venue, _, username = handle.partition("/") if handle else ("", "", "")
    if handle and not (venue and username):
        raise SystemExit(f"error: a handle is VENUE/USERNAME, not {handle!r}")
    async with await EventQueue.open(config.db_path) as queue:
        profile = await queue.people.register(
            name, notes, (venue, username) if handle else None, None
        )
    print(f"{profile.name} is person {profile.id}; profile at {profile.path}")


async def _link(config: Config, venue: str, username: str, name: str) -> None:
    async with await EventQueue.open(config.db_path) as queue:
        profile = await queue.people.link(venue, username, name, None)
    print(f"{venue}/{username} is {profile.name}; profile at {profile.path}")


def _show_block(block: Block) -> str:
    """A content block as a reader wants it: text in full, the rest by shape."""
    match block:
        case {"type": "text", "text": str(text)}:
            return text
        case {"type": "thinking", "thinking": str(thinking)}:
            return f"[thinking, {len(thinking.split())} words]"
        case {"type": "tool_use", "name": str(name), "id": str(call_id)}:
            return f"[tool_use {name} #{call_id}]"
        case {"type": "tool_result", "tool_use_id": str(call_id)}:
            return f"[tool_result for #{call_id}]"
        case {"type": str(kind)}:
            return f"[{kind}]"
        case _:
            return "[unknown block]"


async def _pass(config: Config, which: str, *, dry_run: bool) -> None:
    """Run a triage or sweep pass, or just show the prompt it would send."""
    renderer = PromptRenderer()
    empty = f"Nothing to {which}."

    async with await EventQueue.open(config.db_path) as queue:
        if dry_run:
            # Fetched exactly as the pass itself does, so this really is the
            # prompt the model would receive.
            if which == "triage":
                view = await queue.triage_view()
                prompt = renderer.triage(view.pending, view.deferred) if view.pending else ""
            else:
                sweep = await queue.sweep_view()
                prompt = renderer.sweep(sweep.rows, sweep.history) if sweep.rows else ""
            print(prompt or empty)
            return

        # Imported here so the dry run and the other commands need no API key.
        from anthropic import AsyncAnthropic
        from dotenv import find_dotenv, load_dotenv

        from .sweep import run_sweep
        from .triage import run_triage

        load_dotenv(find_dotenv(usecwd=True))
        run = run_triage if which == "triage" else run_sweep

        async with AsyncAnthropic() as client:
            result = await run(client, config, queue, renderer)

    if not result.considered:
        print(empty)
        return
    _report(result)


async def _watch(config: Config) -> None:
    """Run each role's pass on its heartbeat, and triage's on an immediate arrival."""
    renderer = PromptRenderer()

    # Imported here for the same reason as in `_pass`.
    from anthropic import AsyncAnthropic
    from dotenv import find_dotenv, load_dotenv

    from .scheduler import OnDue, watch
    from .sweep import run_sweep
    from .triage import run_triage

    load_dotenv(find_dotenv(usecwd=True))

    async with await EventQueue.open(config.db_path) as queue, AsyncAnthropic() as client:

        async def triage_pass(beat: Beat) -> None:
            _report(await run_triage(client, config, queue, renderer, beat=beat))

        async def sweep_pass(beat: Beat) -> None:
            _report(await run_sweep(client, config, queue, renderer, beat=beat))

        passes: dict[AgentRole, OnDue] = {
            AgentRole.TRIAGE: triage_pass,
            AgentRole.SWEEP: sweep_pass,
        }
        on_due: dict[AgentRole, OnDue] = {}
        for role in LOOP_ROLES:
            if role not in config.heartbeat_seconds:
                continue
            await queue.ensure_standing(role, config.heartbeat_seconds[role])
            on_due[role] = passes[role]

        rhythm = ", ".join(
            f"{role.value} every {config.heartbeat_seconds[role]:g}s" for role in on_due
        )
        print(
            f"Watching {config.db_path}: {rhythm}; triage on a nudge or an active "
            "arrival. ^C to stop."
        )
        await watch(queue, on_due=on_due)


async def _heartbeats(config: Config) -> None:
    async with await EventQueue.open(config.db_path) as queue:
        summaries = await queue.heartbeats()

    if not summaries:
        print("Nothing scheduled; `watch` makes the standing schedules when it starts.")
        return

    now = utcnow()
    for summary in summaries:
        schedule = summary.schedule
        if schedule.interval_seconds is None:
            rhythm = "once"
        else:
            rhythm = f"every {_duration(schedule.interval_seconds)}"
            if schedule.expires_at is not None:
                rhythm += f" until {schedule.expires_at.isoformat(timespec='seconds')}"
        by = schedule.agent.label if schedule.agent else "operator"
        fired = f"fired {summary.fired}×" if summary.fired else "never fired"
        print(
            f"[{schedule.id:>3}] {schedule.role.value:<7} {rhythm:<36} "
            f"next {summary.next_due.isoformat(timespec='seconds')} "
            f"({_in(summary.next_due - now)})   {fired}   by {by}"
        )
        if schedule.message:
            print(textwrap.indent(textwrap.fill(schedule.message, 80), " " * 12))


def _duration(seconds: float) -> str:
    """Seconds as a person says them: 45s, 5m, 1.5h."""
    if seconds < 60:
        return f"{seconds:g}s"
    if seconds < 3600:
        return f"{seconds / 60:g}m"
    return f"{seconds / 3600:g}h"


def _in(delta: timedelta) -> str:
    seconds = delta.total_seconds()
    if seconds <= 0:
        return "due"
    return f"in {_duration(seconds)}"


def _plural(count: int, noun: str) -> str:
    return f"{count} {noun}" if count == 1 else f"{count} {noun}s"


def _show_usage(spend: Spend) -> str:
    """Input tokens with how many the cache served, since that is the point."""
    return (
        f"{spend.total_input} in: {spend.cache_read} cached, "
        f"{spend.cache_written} newly cached; {spend.output} out"
    )


def _report(result: PassResult) -> None:
    considered, dispositions = result.considered, result.dispositions
    if not considered and not result.check_ins:
        return
    print(
        f"\n{result.agent.label} considered {_plural(len(considered), 'event')} "
        f"and reached {_plural(len(dispositions), 'disposition')}.\n"
    )

    if result.check_ins:
        print(f"Woken with {_plural(len(result.check_ins), 'check-in')}:")
        for note in result.check_ins:
            by = note.left_by.label if note.left_by else "operator"
            due = note.due_at.isoformat(timespec="seconds")
            print(f"    [{by}, due {due}] {note.message}")
        print()

    for disposition in dispositions:
        listed = ", ".join(str(i) for i in disposition.event_ids)
        actor = f"  [{disposition.agent.label}]" if disposition.agent else ""
        # More than one stream means the assignment crossed contexts.
        spans = f"  {' → '.join(disposition.streams)}" if disposition.streams else ""
        print(f"{disposition.action.value}({listed}){spans}{actor}")
        print(textwrap.indent(textwrap.fill(disposition.detail, 84), "    "))
        if disposition.error:
            print(textwrap.indent(f"FAILED: {disposition.error}", "    "))
        elif disposition.report:
            print(textwrap.indent(textwrap.fill(disposition.report, 84), "  > "))
        if disposition.usage:
            where = f"context {disposition.context_id}, " if disposition.context_id else ""
            print(f"    [{where}{_show_usage(disposition.usage)}]")
        print()

    if missed := [row.id for row in considered if row.id not in result.dispatched_ids]:
        print(f"Left without a disposition: {missed}")

    for summary in result.scheduled:
        schedule = summary.schedule
        when = summary.next_due.isoformat(timespec="seconds")
        if schedule.interval_seconds is None:
            print(f"Scheduled a check-in for {schedule.role.value} at {when}:")
        else:
            until = schedule.expires_at.isoformat(timespec="seconds") if schedule.expires_at else "ever"
            print(
                f"Scheduled {schedule.role.value} every "
                f"{_duration(schedule.interval_seconds)} until {until}, first at {when}:"
            )
        print(textwrap.indent(textwrap.fill(schedule.message or "", 80), "    "))


if __name__ == "__main__":
    main()
