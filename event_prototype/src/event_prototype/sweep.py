"""One backlog sweep: what the deferred events are actually worth.

Where triage is fast and sees only what is live, the sweep is slow, runs on a
schedule or in idle time, and sees the deferred backlog with the full log of
what has been decided about each event. It is the only pass that can archive.
"""

from __future__ import annotations

from anthropic import AsyncAnthropic

from .agents import Agent, AgentRole
from .config import Config
from .queue import EventQueue
from .render import PromptRenderer
from .runner import PassResult, run_pass
from .tools import Dispatcher, build_sweep_server


async def run_sweep(
    client: AsyncAnthropic,
    config: Config,
    queue: EventQueue,
    renderer: PromptRenderer | None = None,
) -> PassResult:
    """Reconsider every deferred event, then wait for anything it dispatched."""
    renderer = renderer or PromptRenderer()
    agent = Agent.spawn(AgentRole.SWEEP)

    view = await queue.sweep_view()
    if not view.rows:
        return PassResult(prompt="", considered=[], dispositions=[], agent=agent)

    prompt = renderer.sweep(view.rows, view.history)
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
    )
