"""One backlog sweep: what the deferred events are actually worth.

Where triage is fast and sees only what is live, the sweep is slow, runs on
its own heartbeat, and sees the deferred backlog with every action taken on
each event so far. It is the only pass that can archive.
"""

from __future__ import annotations

from anthropic import AsyncAnthropic

from .agents import AgentRole
from .config import Config
from .heartbeats import Beat
from .queue import EventQueue
from .render import PromptRenderer
from .runner import PassResult, run_pass
from .tools import Dispatcher, build_sweep_server


async def run_sweep(
    client: AsyncAnthropic,
    config: Config,
    queue: EventQueue,
    renderer: PromptRenderer | None = None,
    *,
    beat: Beat | None = None,
) -> PassResult:
    """Reconsider every deferred event, then wait for anything it dispatched.

    `beat` is what the scheduler found on the sweep's schedule when it woke
    this pass; a manual pass has none.
    """
    renderer = renderer or PromptRenderer()
    check_ins = beat.check_ins if beat else ()
    agent = await queue.spawn(AgentRole.SWEEP)

    view = await queue.sweep_view()
    if not view.rows and not check_ins:
        return PassResult(prompt="", considered=[], dispositions=[], agent=agent)

    prompt = renderer.sweep(
        view.rows,
        view.history,
        check_ins=check_ins,
        next_due=beat.next_due if beat else None,
    )
    dispatcher = Dispatcher(queue, client, config, renderer, agent)
    await run_pass(
        client,
        dispatcher,
        build_sweep_server(dispatcher),
        model=config.sweep_model,
        prompt=prompt,
        # Nobody is waiting on this pass, and its calls are the consequential
        # ones — archiving is not reversible.
        effort=config.sweep_effort,
    )
    return PassResult(
        prompt=prompt,
        considered=view.rows,
        dispositions=dispatcher.dispositions,
        agent=agent,
        check_ins=check_ins,
        scheduled=tuple(dispatcher.scheduled),
    )
