"""Running one model pass against one tool set.

Triage and the sweep differ in what they are shown and what they may do, not in
how they are driven, so the driving lives here.
"""

from __future__ import annotations

from dataclasses import dataclass

from anthropic import AsyncAnthropic, omit
from mcp.server import MCPServer

from .agents import Agent
from .config import Effort
from .heartbeats import CheckIn, HeartbeatSummary
from .store import EventRow
from .tools import Disposition, Dispatcher, mcp_tools

MAX_TOKENS = 8192


@dataclass(frozen=True, slots=True)
class PassResult:
    prompt: str
    considered: list[EventRow]
    dispositions: list[Disposition]
    agent: Agent
    #: The notes this pass was woken with, and the schedules it wrote.
    check_ins: tuple[CheckIn, ...] = ()
    scheduled: tuple[HeartbeatSummary, ...] = ()

    @property
    def dispatched_ids(self) -> set[int]:
        return {i for d in self.dispositions for i in d.event_ids}


async def run_pass(
    client: AsyncAnthropic,
    dispatcher: Dispatcher,
    server: MCPServer,
    *,
    model: str,
    prompt: str,
    effort: Effort | None = None,
) -> None:
    """Let `model` work through `prompt` with `server`'s tools, then settle up."""
    async with mcp_tools(server) as tools:
        runner = client.beta.messages.tool_runner(
            model=model,
            max_tokens=MAX_TOKENS,
            tools=tools,
            messages=[{"role": "user", "content": prompt}],
            # Omitted unless asked for: a triage model chosen for speed (Haiku,
            # say) supports neither and rejects the request outright.
            thinking={"type": "adaptive"} if effort else omit,
            output_config={"effort": effort} if effort else omit,
        )
        async for _ in runner:
            pass

    await dispatcher.drain()
