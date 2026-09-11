"""The subagent side: one larger model continuing the conversation it was handed."""

from __future__ import annotations

from collections.abc import Iterable

from anthropic import AsyncAnthropic, omit
from anthropic.types import Message

from .config import Config
from .contexts import Turn, to_api

MAX_TOKENS = 16000


async def run_subagent(
    client: AsyncAnthropic,
    config: Config,
    system: str,
    turns: Iterable[Turn],
    *,
    persistent: bool,
) -> Message:
    """One completion over `turns`, returned whole.

    Recording the reply and the outcome is the dispatcher's job; this only does
    the work. In the real framework this would be a full agent loop with tools;
    here it is a single call, which is enough to exercise the dispatch path.

    A persistent context asks for caching. The request-level marker puts the
    breakpoint on the last block sent, and the API looks back from there for the
    prefix the previous call wrote — so each call in a stream reads the last one
    and caches through its own. A one-shot conversation has no next call, so it
    does not pay the write.
    """
    return await client.messages.create(
        model=config.subagent_model,
        max_tokens=MAX_TOKENS,
        thinking={"type": "adaptive"},
        output_config={"effort": config.subagent_effort},
        system=system,
        messages=to_api(turns),
        cache_control={"type": "ephemeral"} if persistent else omit,
    )


def report_of(response: Message) -> str:
    """What the subagent said it did, as `event_actions` records it."""
    if response.stop_reason == "refusal":
        return "[subagent declined to handle this event]"
    return Turn.of(response).text
