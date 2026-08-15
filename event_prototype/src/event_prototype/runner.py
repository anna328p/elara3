"""Running one model pass against one tool set.

Triage and the sweep differ in what they are shown and what they may do, not in
how they are driven, so the driving lives here.
"""

from __future__ import annotations

from dataclasses import dataclass

from anthropic import AsyncAnthropic, omit
from anthropic.lib.tools.mcp import async_mcp_tool
from mcp import Client
from mcp.server import MCPServer

from .agents import Agent
from .config import Effort
from .store import EventRow
from .tools import Disposition, Dispatcher

MAX_TOKENS = 8192


@dataclass(frozen=True, slots=True)
class PassResult:
    prompt: str
    considered: list[EventRow]
    dispositions: list[Disposition]
    agent: Agent

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
    """Let `model` work through `prompt` with `server`'s tools, then settle up.

    The MCP server runs in this process: the client speaks the same protocol to
    it over an in-memory transport, with no subprocess to supervise.
    """
    async with Client(server) as mcp:
        listed = await mcp.list_tools()
        runner = client.beta.messages.tool_runner(
            model=model,
            max_tokens=MAX_TOKENS,
            # `.session` is the underlying protocol session the Anthropic bridge
            # calls tools through.
            tools=[async_mcp_tool(tool, mcp.session) for tool in listed.tools],
            messages=[{"role": "user", "content": prompt}],
            # Omitted unless asked for: a triage model chosen for speed (Haiku,
            # say) supports neither and rejects the request outright.
            thinking={"type": "adaptive"} if effort else omit,
            output_config={"effort": effort} if effort else omit,
        )
        async for _ in runner:
            pass

    await dispatcher.drain()
