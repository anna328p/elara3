"""Config loading: the keys that need more than a string."""

from __future__ import annotations

from pathlib import Path

import pytest

from event_prototype.agents import AgentRole
from event_prototype.config import Config
from event_prototype.events import PRIORITY_NAMES, Priority


def test_priority_names_and_the_enum_cannot_drift() -> None:
    assert set(PRIORITY_NAMES) == {priority.name.lower() for priority in Priority}


def test_an_unknown_priority_name_is_rejected() -> None:
    with pytest.raises(ValueError, match="priority must be one of"):
        Priority.from_name("urgent")


def test_the_heartbeats_are_read_from_the_file(tmp_path: Path) -> None:
    path = tmp_path / "config.toml"
    path.write_text("[heartbeat_seconds]\ntriage = 30\nsweep = 600\n")

    config = Config.load(path)

    assert config.heartbeat_seconds == {AgentRole.TRIAGE: 30.0, AgentRole.SWEEP: 600.0}


def test_an_unnamed_role_keeps_its_default_heartbeat(tmp_path: Path) -> None:
    path = tmp_path / "config.toml"
    path.write_text("[heartbeat_seconds]\ntriage = 30\n")

    config = Config.load(path)

    assert config.heartbeat_seconds[AgentRole.TRIAGE] == 30.0
    assert config.heartbeat_seconds[AgentRole.SWEEP] == Config().heartbeat_seconds[AgentRole.SWEEP]


def test_a_non_positive_heartbeat_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "config.toml"
    path.write_text("[heartbeat_seconds]\ntriage = 0\n")

    with pytest.raises(ValueError, match="heartbeat_seconds.triage must be positive"):
        Config.load(path)


def test_a_flat_heartbeat_is_rejected_with_the_table_form_named(tmp_path: Path) -> None:
    path = tmp_path / "config.toml"
    path.write_text("heartbeat_seconds = 300\n")

    with pytest.raises(ValueError, match=r"\[heartbeat_seconds\]"):
        Config.load(path)


def test_a_heartbeat_for_an_unknown_role_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "config.toml"
    # The classic mistake: a key written after the table header.
    path.write_text("[heartbeat_seconds]\ntriage = 30\nsubagent_max_iterations = 4\n")

    with pytest.raises(ValueError, match="no such role.*goes last"):
        Config.load(path)


def test_a_heartbeat_for_a_role_without_a_pass_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "config.toml"
    path.write_text("[heartbeat_seconds]\nsubagent = 30\n")

    with pytest.raises(ValueError, match="no pass to run on a heartbeat"):
        Config.load(path)
