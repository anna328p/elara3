"""Command line for the prototype: seed the queue, look at it, triage it."""

from __future__ import annotations

import argparse
import asyncio
import logging
import textwrap
from pathlib import Path
from typing import TYPE_CHECKING

from .config import DEFAULT_CONFIG_PATH, Config
from .fixtures import seed
from .queue import EventQueue
from .render import PromptRenderer

if TYPE_CHECKING:
    from .runner import PassResult


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

    args = parser.parse_args()
    config = Config.load(args.config)
    # The MCP server turns on INFO logging, which makes httpx narrate every
    # request over the report we are trying to print.
    logging.getLogger("httpx").setLevel(logging.WARNING)
    asyncio.run(_dispatch(args, config))


async def _dispatch(args: argparse.Namespace, config: Config) -> None:
    match args.command:
        case "seed":
            await _seed(config, fresh=args.fresh)
        case "list":
            await _list(config, show_all=args.all)
        case "triage" | "sweep" as which:
            await _pass(config, which, dry_run=args.dry_run)
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
        # One query for the whole log slice, not one per event.
        history = await queue.history_for([row.id for row in rows])

    if not rows:
        print("Queue is empty." if show_all else "No active events.")
        return

    for row in rows:
        print(
            f"[{row.id:>3}] {row.priority.name.lower():<10} {row.kind:<9} "
            f"{row.status.value:<9} {row.description}"
        )
        for entry in history.get(row.id, []):
            stamp = entry.timestamp.isoformat(timespec="seconds")
            print(f"        {stamp} {entry.action.value} by {entry.agent.label}")
            print(textwrap.indent(textwrap.fill(entry.detail, 80), " " * 12))


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


def _plural(count: int, noun: str) -> str:
    return f"{count} {noun}" if count == 1 else f"{count} {noun}s"


def _report(result: PassResult) -> None:
    considered, dispositions = result.considered, result.dispositions
    print(
        f"\n{result.agent.label} considered {_plural(len(considered), 'event')} "
        f"and reached {_plural(len(dispositions), 'disposition')}.\n"
    )

    for disposition in dispositions:
        listed = ", ".join(str(i) for i in disposition.event_ids)
        actor = f"  [{disposition.agent.label}]" if disposition.agent else ""
        print(f"{disposition.action.value}({listed}){actor}")
        print(textwrap.indent(textwrap.fill(disposition.detail, 84), "    "))
        if disposition.error:
            print(textwrap.indent(f"FAILED: {disposition.error}", "    "))
        elif disposition.report:
            print(textwrap.indent(textwrap.fill(disposition.report, 84), "  > "))
        print()

    if missed := [row.id for row in considered if row.id not in result.dispatched_ids]:
        print(f"Left without a disposition: {missed}")


if __name__ == "__main__":
    main()
