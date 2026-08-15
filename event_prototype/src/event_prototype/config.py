"""Configuration, loaded from a TOML file and passed explicitly to whoever needs it."""

from __future__ import annotations

import tomllib
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any, Literal, get_args

DEFAULT_CONFIG_PATH = Path("config.toml")

type Effort = Literal["low", "medium", "high", "max"]
EFFORTS: tuple[str, ...] = get_args(Effort.__value__)


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
        return cls(**raw)
