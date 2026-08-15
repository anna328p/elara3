"""Who did what.

Every row in the handling log is attributed to an agent. In the real framework
an agent is a database row with a UUID and an event loop of its own; here it is
just the identity, minted when the agent starts work and carried through to the
log so the trail survives the process.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from uuid import uuid4


class AgentRole(StrEnum):
    TRIAGE = "triage"
    SWEEP = "sweep"
    SUBAGENT = "subagent"


@dataclass(frozen=True, slots=True)
class Agent:
    role: AgentRole
    id: str

    @classmethod
    def spawn(cls, role: AgentRole) -> Agent:
        return cls(role, str(uuid4()))

    @property
    def label(self) -> str:
        """Short form, for prompts and reports where the full UUID is noise."""
        return f"{self.role.value} {self.id[:8]}"
