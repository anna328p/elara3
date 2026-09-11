"""Config loading: the keys that need more than a string."""

from __future__ import annotations

from pathlib import Path

import pytest

from event_prototype.config import Config
from event_prototype.events import PRIORITY_NAMES, Priority


def test_priority_names_and_the_enum_cannot_drift() -> None:
    assert set(PRIORITY_NAMES) == {priority.name.lower() for priority in Priority}


def test_an_unknown_priority_name_is_rejected() -> None:
    with pytest.raises(ValueError, match="priority must be one of"):
        Priority.from_name("urgent")


def test_the_heartbeat_is_read_from_the_file(tmp_path: Path) -> None:
    path = tmp_path / "config.toml"
    path.write_text("heartbeat_seconds = 30\n")

    config = Config.load(path)

    assert config.heartbeat_seconds == 30.0


def test_a_non_positive_heartbeat_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "config.toml"
    path.write_text("heartbeat_seconds = 0\n")

    with pytest.raises(ValueError, match="heartbeat_seconds must be positive"):
        Config.load(path)
