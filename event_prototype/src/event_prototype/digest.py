"""Writing the one-liner that represents a deferred event back to triage.

Triage cannot afford to read the backlog in full, but the alternative — showing
it whatever `description` the event arrived with — describes the event as it was
filed rather than as it stands. A small model writes something better for a few
hundred tokens.
"""

from __future__ import annotations

from anthropic import AsyncAnthropic

from .config import Config
from .render import PromptRenderer
from .store import EventRow

MAX_TOKENS = 100


async def summarize(
    client: AsyncAnthropic,
    config: Config,
    renderer: PromptRenderer,
    row: EventRow,
    reason: str,
) -> str:
    """One line for `row`, for triage to match against what is live."""
    response = await client.messages.create(
        model=config.digest_model,
        max_tokens=MAX_TOKENS,
        messages=[{"role": "user", "content": renderer.digest(row, reason)}],
    )
    text = "\n".join(
        block.text for block in response.content if block.type == "text"
    )
    # However the model chose to lay it out, this has to be one line.
    return " ".join(text.split())
