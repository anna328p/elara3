"""Who did what.

Every action taken on an event is attributed to an agent, and every context is
some agent's conversation. An agent is a row in `agents`: a UUID and a role,
minted by the store when the agent starts work (`EventQueue.spawn`), so the
identity exists before anything can point at it and the trail survives the
process. `Agent` is that row's identity as it travels through the code — a
value, not the row itself.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum


class AgentRole(StrEnum):
    TRIAGE = "triage"
    SWEEP = "sweep"
    SUBAGENT = "subagent"


@dataclass(frozen=True, slots=True)
class Agent:
    role: AgentRole
    id: str

    @property
    def label(self) -> str:
        """Short form, for prompts and reports where the full UUID is noise."""
        return f"{self.role.value} {self.id[:8]}"
