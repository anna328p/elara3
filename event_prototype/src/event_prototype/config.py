"""Configuration, loaded from a TOML file and passed explicitly to whoever needs it."""

from __future__ import annotations

import tomllib
from collections.abc import Mapping
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any, Literal, cast, get_args

from .agents import LOOP_ROLES, AgentRole

DEFAULT_CONFIG_PATH = Path("config.toml")

type Effort = Literal["low", "medium", "high", "max"]
EFFORTS: tuple[str, ...] = get_args(Effort.__value__)

DEFAULT_HEARTBEAT_SECONDS: Mapping[AgentRole, float] = {
    AgentRole.TRIAGE: 300.0,
    AgentRole.SWEEP: 3600.0,
}


@dataclass(frozen=True, slots=True)
class Config:
    triage_model: str = "claude-sonnet-5"
    sweep_model: str = "claude-sonnet-5"
    subagent_model: str = "claude-sonnet-5"
    #: Writes the backlog one-liners; small and frequent, so keep it cheap.
    digest_model: str = "claude-haiku-4-5"
    db_path: Path = Path("events.db")
    subagent_effort: Effort = "medium"
    sweep_effort: Effort = "high"
    #: How many model calls a subagent may make in one assignment before it is
    #: stopped. Every call after the first is a round of tool results.
    subagent_max_iterations: int = 8
    #: Each role's standing heartbeat: how often its pass runs when nothing
    #: brings one forward. The keys are the roles `watch` runs.
    heartbeat_seconds: Mapping[AgentRole, float] = field(
        default_factory=lambda: dict(DEFAULT_HEARTBEAT_SECONDS)
    )

    @classmethod
    def load(cls, path: Path = DEFAULT_CONFIG_PATH) -> Config:
        """Read `path`, falling back to defaults for a missing file or absent keys."""
        try:
            raw: dict[str, Any] = tomllib.loads(path.read_text())
        except FileNotFoundError:
            return cls()

        known = {f.name for f in fields(cls)}
        if unknown := raw.keys() - known:
            raise ValueError(f"unknown config keys in {path}: {sorted(unknown)}")

        for key in ("subagent_effort", "sweep_effort"):
            if (effort := raw.get(key)) is not None and effort not in EFFORTS:
                raise ValueError(f"{key} must be one of {EFFORTS}, not {effort!r}")
        if "db_path" in raw:
            raw["db_path"] = Path(raw["db_path"])
        if "heartbeat_seconds" in raw:
            raw["heartbeat_seconds"] = _heartbeat_seconds(raw["heartbeat_seconds"])
        if (cap := raw.get("subagent_max_iterations")) is not None:
            # `bool` is an int to `isinstance`, and `true` is not a count.
            if isinstance(cap, bool) or not isinstance(cap, int) or cap < 1:
                raise ValueError(
                    f"subagent_max_iterations must be a positive integer, not {cap!r}"
                )
        return cls(**raw)


def _heartbeat_seconds(raw: Any) -> dict[AgentRole, float]:
    """The `[heartbeat_seconds]` table, one positive number per loop role,
    over the defaults for any role it leaves out."""
    if not isinstance(raw, dict):
        raise ValueError(
            "heartbeat_seconds is a table keyed by role, one per line under "
            "[heartbeat_seconds]: triage = 300, sweep = 3600"
        )
    table = cast(dict[str, Any], raw)
    allowed = [role.value for role in LOOP_ROLES]
    seconds = dict(DEFAULT_HEARTBEAT_SECONDS)
    for name, value in table.items():
        try:
            role = AgentRole(name)
        except ValueError:
            raise ValueError(
                f"heartbeat_seconds.{name}: no such role; the roles are {allowed}. "
                "Keys written after the [heartbeat_seconds] header belong to it, "
                "so the table goes last in the file."
            ) from None
        if role not in LOOP_ROLES:
            raise ValueError(
                f"heartbeat_seconds.{name}: a {name} has no pass to run on a heartbeat; "
                f"the roles are {allowed}"
            )
        interval = float(value)
        if interval <= 0:
            raise ValueError(f"heartbeat_seconds.{name} must be positive, not {value!r}")
        seconds[role] = interval
    return seconds
