"""Tests for project path discovery."""

from __future__ import annotations

from pathlib import Path

import pytest

from tinyearth.utils import paths as paths_module
from tinyearth.utils.paths import (
    ROOT_ENV_VAR,
    ProjectRootNotFoundError,
    cache_dir,
    configs_dir,
    data_dir,
    outputs_dir,
    project_root,
)


@pytest.fixture(autouse=True)
def _clear_root_cache():
    """Clear the memoised root between tests that monkeypatch the environment."""
    project_root.cache_clear()
    yield
    project_root.cache_clear()


def test_project_root_contains_pyproject():
    root = project_root()
    assert (root / "pyproject.toml").is_file()


def test_project_root_contains_the_source_tree():
    assert (project_root() / "src" / "tinyearth" / "__init__.py").is_file()


def test_project_root_is_absolute():
    assert project_root().is_absolute()


def test_derived_directories_sit_under_the_root():
    root = project_root()
    for directory in (configs_dir(), data_dir(), outputs_dir(), cache_dir()):
        assert directory.parent == root


def test_configs_dir_exists_and_holds_the_root_config():
    assert (configs_dir() / "config.yaml").is_file()


def test_env_var_overrides_discovery(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv(ROOT_ENV_VAR, str(tmp_path))
    assert project_root() == tmp_path.resolve()


def test_env_var_pointing_at_a_missing_directory_raises(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setenv(ROOT_ENV_VAR, str(tmp_path / "nope"))
    with pytest.raises(ProjectRootNotFoundError, match="not a directory"):
        project_root()


def test_is_project_root_rejects_a_foreign_pyproject(tmp_path: Path):
    (tmp_path / "pyproject.toml").write_text('[project]\nname = "something-else"\n')
    assert paths_module._is_project_root(tmp_path) is False


def test_is_project_root_rejects_a_directory_without_a_marker(tmp_path: Path):
    assert paths_module._is_project_root(tmp_path) is False
