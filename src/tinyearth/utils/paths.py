"""Project path discovery.

Hydra changes the process working directory when a run starts, so any code that
needs a repository-relative path (configs, data, docs) must resolve it against a
stable anchor rather than :func:`os.getcwd`. This module provides that anchor.

The project root is the nearest ancestor directory containing a ``pyproject.toml``
with a ``[project]`` table naming ``tinyearth``. It may be overridden with the
``TINYEARTH_ROOT`` environment variable, which is the supported escape hatch for
installed (non-editable) deployments where no source tree exists.
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

__all__ = [
    "ROOT_ENV_VAR",
    "cache_dir",
    "configs_dir",
    "data_dir",
    "outputs_dir",
    "project_root",
]

ROOT_ENV_VAR = "TINYEARTH_ROOT"
"""Environment variable that overrides automatic project-root discovery."""

_MARKER = "pyproject.toml"


class ProjectRootNotFoundError(RuntimeError):
    """Raised when the TinyEarth project root cannot be located."""


def _is_project_root(candidate: Path) -> bool:
    """Return whether ``candidate`` holds the TinyEarth ``pyproject.toml``.

    Args:
        candidate: Directory to test.

    Returns:
        ``True`` if the directory contains a ``pyproject.toml`` declaring the
        ``tinyearth`` project.
    """
    marker = candidate / _MARKER
    if not marker.is_file():
        return False
    try:
        text = marker.read_text(encoding="utf-8")
    except OSError:  # pragma: no cover - unreadable marker is treated as absent
        return False
    return 'name = "tinyearth"' in text


@lru_cache(maxsize=1)
def project_root() -> Path:
    """Locate the repository root.

    Resolution order:

    1. ``$TINYEARTH_ROOT`` if set (must exist and be a directory).
    2. The nearest ancestor of this file containing the project ``pyproject.toml``.

    Returns:
        Absolute path to the repository root.

    Raises:
        ProjectRootNotFoundError: If neither strategy succeeds. Set
            ``TINYEARTH_ROOT`` to resolve this.
    """
    override = os.environ.get(ROOT_ENV_VAR)
    if override:
        path = Path(override).expanduser().resolve()
        if not path.is_dir():
            raise ProjectRootNotFoundError(
                f"{ROOT_ENV_VAR} points at {path!s}, which is not a directory."
            )
        return path

    here = Path(__file__).resolve()
    for parent in here.parents:
        if _is_project_root(parent):
            return parent

    raise ProjectRootNotFoundError(
        "Could not locate the TinyEarth project root by walking up from "
        f"{here!s}. This happens when the package is installed non-editable; "
        f"set the {ROOT_ENV_VAR} environment variable to the repository path."
    )


def configs_dir() -> Path:
    """Return the Hydra configuration tree (``<root>/configs``)."""
    return project_root() / "configs"


def data_dir() -> Path:
    """Return the default dataset directory (``<root>/data``).

    Datasets are never committed; this directory is git-ignored and populated by
    the download scripts introduced in Phase 2.
    """
    return project_root() / "data"


def outputs_dir() -> Path:
    """Return the default experiment output directory (``<root>/outputs``)."""
    return project_root() / "outputs"


def cache_dir() -> Path:
    """Return the default cache directory (``<root>/.cache``)."""
    return project_root() / ".cache"
