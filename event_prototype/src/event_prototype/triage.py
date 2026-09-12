"""One triage pass: what needs attention now.

The triage model sees pending events in full and the deferred backlog as a list
of one-liners, so a live event stays legible however far the backlog has grown.
Its whole vocabulary is `handle_one_event`, `handle_event_sequence`, and
`defer_event`, plus the two tools that act on its own schedule.
"""

from __future__ import annotations

from anthropic import AsyncAnthropic

from .agents import AgentRole
from .config import Config
from .heartbeats import Beat
from .queue import EventQueue
from .render import PromptRenderer
from .runner import PassResult, run_pass
from .tools import Dispatcher, build_triage_server


async def run_triage(
    client: AsyncAnthropic,
    config: Config,
    queue: EventQueue,
    renderer: PromptRenderer | None = None,
    *,
    beat: Beat | None = None,
) -> PassResult:
    """Triage every pending event, then wait for the subagents it spawned.

    `beat` is what the scheduler found on triage's schedule when it woke this
    pass: check-ins to show, and when the next pass is due. A manual pass has
    none, and sees neither.
    """
    renderer = renderer or PromptRenderer()
    check_ins = beat.check_ins if beat else ()
    agent = await queue.spawn(AgentRole.TRIAGE)

    view = await queue.triage_view()
    if not view.pending and not check_ins:
        return PassResult(prompt="", considered=[], dispositions=[], agent=agent)

    prompt = renderer.triage(
        view.pending,
        view.deferred,
        check_ins=check_ins,
        next_due=beat.next_due if beat else None,
    )
    dispatcher = Dispatcher(queue, client, config, renderer, agent)
    await run_pass(
        client,
        dispatcher,
        build_triage_server(dispatcher),
        model=config.triage_model,
        prompt=prompt,
    )
    return PassResult(
        prompt=prompt,
        considered=view.pending,
        dispositions=dispatcher.dispositions,
        agent=agent,
        check_ins=check_ins,
        scheduled=tuple(dispatcher.scheduled),
    )
