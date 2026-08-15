"""One triage pass: what needs attention now.

The triage model sees pending events in full and the deferred backlog as a list
of one-liners, so a live event stays legible however far the backlog has grown.
Its whole vocabulary is `handle_one_event`, `handle_event_sequence`, and
`defer_event`.
"""

from __future__ import annotations

from anthropic import AsyncAnthropic

from .agents import Agent, AgentRole
from .config import Config
from .queue import EventQueue
from .render import PromptRenderer
from .runner import PassResult, run_pass
from .tools import Dispatcher, build_triage_server


async def run_triage(
    client: AsyncAnthropic,
    config: Config,
    queue: EventQueue,
    renderer: PromptRenderer | None = None,
) -> PassResult:
    """Triage every pending event, then wait for the subagents it spawned."""
    renderer = renderer or PromptRenderer()
    agent = Agent.spawn(AgentRole.TRIAGE)

    view = await queue.triage_view()
    if not view.pending:
        return PassResult(prompt="", considered=[], dispositions=[], agent=agent)

    prompt = renderer.triage(view.pending, view.deferred)
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
    )
