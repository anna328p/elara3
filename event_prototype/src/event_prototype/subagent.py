"""The subagent side: a larger model continuing the conversation it was handed,
with its memory and whatever else the dispatcher gives it, one call at a time
until it stops asking for tools."""

from __future__ import annotations

from collections.abc import AsyncIterator, Iterable, Sequence
from dataclasses import dataclass
from typing import Any, cast

from anthropic import AsyncAnthropic, omit
from anthropic.lib.tools import BetaAsyncBuiltinFunctionTool, BetaAsyncFunctionTool
from anthropic.types.beta import BetaMessage, BetaMessageParam, BetaUsage

from .config import Config
from .contexts import Turn, to_api

MAX_TOKENS = 16000

#: A tool the runner can call on the subagent's behalf: an MCP tool wrapped for
#: it, or a built-in such as the memory tool.
type SubagentTool = BetaAsyncFunctionTool[Any] | BetaAsyncBuiltinFunctionTool


@dataclass(frozen=True, slots=True)
class Spend:
    """Tokens over a whole assignment, cache reads and writes told apart.

    One assignment is several calls now, so the disposition carries a sum
    rather than the last call's `usage`.
    """

    input: int = 0
    cache_read: int = 0
    cache_written: int = 0
    output: int = 0

    def plus(self, usage: BetaUsage) -> Spend:
        return Spend(
            self.input + usage.input_tokens,
            self.cache_read + (usage.cache_read_input_tokens or 0),
            self.cache_written + (usage.cache_creation_input_tokens or 0),
            self.output + usage.output_tokens,
        )

    @property
    def total_input(self) -> int:
        return self.input + self.cache_read + self.cache_written


@dataclass(frozen=True, slots=True)
class Step:
    """One message of the loop: the turn it is and, for the model's own, the completion."""

    turn: Turn
    message: BetaMessage | None = None


async def run_subagent(
    client: AsyncAnthropic,
    config: Config,
    system: str,
    turns: Iterable[Turn],
    tools: Sequence[SubagentTool],
    *,
    persistent: bool,
) -> AsyncIterator[Step]:
    """The loop over `turns`, yielded a message at a time as it happens.

    Each of the model's replies is yielded first; then the tools it asked for
    are run, and their results are yielded as the user turn that answers it.
    The runner does both of those itself when it resumes, but asking for the
    tool response here means the results exist — and can be stored — before
    the next request goes out, and the runner reuses them rather than running
    the tools twice. A reply that asks for nothing ends the loop.

    Recording turns and the outcome is the dispatcher's job; this only does
    the work. A persistent context asks for caching: the request-level marker
    puts the breakpoint on the last block sent, and the API looks back from
    there for the prefix the previous call wrote, so each call in a stream
    reads the last one and caches through its own. A one-shot conversation has
    no next call, so it does not pay the write.
    """
    runner = client.beta.messages.tool_runner(
        model=config.subagent_model,
        max_tokens=MAX_TOKENS,
        thinking={"type": "adaptive"},
        output_config={"effort": config.subagent_effort},
        system=system,
        # The same blocks in the same shape; the SDK's beta TypedDicts cannot
        # see that through a JSON column, hence the cast.
        messages=cast(list[BetaMessageParam], to_api(turns)),
        tools=tools,
        max_iterations=config.subagent_max_iterations,
        cache_control={"type": "ephemeral"} if persistent else omit,
    )
    async for message in runner:
        yield Step(Turn.of(message), message)
        response = await runner.generate_tool_call_response()
        if response is None:
            return
        yield Step(Turn.results(response))


def report_of(steps: Sequence[Step]) -> str:
    """What the subagent said it did, as `event_actions` records it."""
    if not steps:
        return "[subagent produced nothing]"
    last = steps[-1]
    if last.message is None:
        return "[subagent stopped at the iteration cap with tool calls outstanding]"
    if last.message.stop_reason == "refusal":
        return "[subagent declined to handle this event]"
    return last.turn.text or "[subagent ended without a report]"
