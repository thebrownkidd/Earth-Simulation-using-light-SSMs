"""Shared pytest fixtures."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest
from hydra import compose, initialize_config_dir
from omegaconf import DictConfig

from tinyearth.config.store import register_configs
from tinyearth.utils.paths import configs_dir


@pytest.fixture(scope="session", autouse=True)
def _register_schema() -> None:
    """Register structured configs once for the whole test session."""
    register_configs()


@pytest.fixture
def compose_config() -> Iterator[object]:
    """Return a helper that composes the real config tree with overrides.

    Tests compose the actual ``configs/`` tree rather than a fixture copy, so a
    broken config file fails the test suite instead of passing against a stale
    duplicate.

    Yields:
        A callable taking a list of Hydra override strings.
    """

    def _compose(overrides: list[str] | None = None) -> DictConfig:
        with initialize_config_dir(version_base=None, config_dir=str(configs_dir())):
            return compose(config_name="config", overrides=overrides or [])

    yield _compose


@pytest.fixture
def isolated_outputs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Redirect run outputs into a temporary directory.

    Args:
        tmp_path: Pytest temporary directory.
        monkeypatch: Pytest monkeypatch fixture.

    Returns:
        The temporary output root.
    """
    monkeypatch.setenv("TINYEARTH_TEST_OUTPUTS", str(tmp_path))
    return tmp_path
