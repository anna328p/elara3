"""The subagent side: one larger model handling the events triage assigned it."""

from __future__ import annotations

from collections.abc import Sequence

from anthropic import AsyncAnthropic
from anthropic.types import Message

from .config import Config
from .render import PromptRenderer
from .store import EventRow

MAX_TOKENS = 16000


async def run_subagent(
    client: AsyncAnthropic,
    config: Config,
    renderer: PromptRenderer,
    rows: Sequence[EventRow],
    instructions: str,
) -> str:
    """Hand `rows` to a subagent and return its account of what it did.

    Recording the outcome is the dispatcher's job; this only does the work.

    In the real framework this would be a full agent with its own event loop and
    tools; here it is a single call, which is enough to exercise the dispatch path.
    """
    response = await client.messages.create(
        model=config.subagent_model,
        max_tokens=MAX_TOKENS,
        thinking={"type": "adaptive"},
        output_config={"effort": config.subagent_effort},
        messages=[{"role": "user", "content": renderer.subagent(rows, instructions)}],
    )
    return _text_of(response)


def _text_of(response: Message) -> str:
    """The visible text of a response, ignoring thinking blocks."""
    if response.stop_reason == "refusal":
        return "[subagent declined to handle this event]"
    return "\n".join(block.text for block in response.content if block.type == "text").strip()
