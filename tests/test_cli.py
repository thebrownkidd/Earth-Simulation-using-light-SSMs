"""End-to-end tests for the console entry points.

These run the commands in a subprocess. That is slower than calling ``main()``
directly, but it is the only way to exercise what a user actually invokes:
argument parsing, Hydra's global state, and the console-script wiring.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

from tinyearth.utils.paths import project_root

pytestmark = pytest.mark.slow


def _run(module: str, *args: str, cwd: Path | None = None) -> subprocess.CompletedProcess[str]:
    """Invoke a CLI module in a subprocess and capture its output."""
    return subprocess.run(
        [sys.executable, "-m", module, *args],
        capture_output=True,
        text=True,
        cwd=cwd or project_root(),
        check=False,
    )


def test_info_command_succeeds():
    result = _run("tinyearth.cli.info")
    assert result.returncode == 0, result.stderr
    assert "TinyEarth" in result.stdout
    assert "torch version" in result.stdout


def test_config_command_succeeds(tmp_path: Path):
    result = _run(
        "tinyearth.cli.inspect_config",
        "logging.rich=false",
        f"paths.outputs={tmp_path.as_posix()}",
        f"paths.cache={(tmp_path / 'cache').as_posix()}",
    )
    assert result.returncode == 0, result.stderr


def test_config_command_writes_the_run_artefacts(tmp_path: Path):
    result = _run(
        "tinyearth.cli.inspect_config",
        "logging.rich=false",
        "run.group=g",
        "run.name=n",
        f"paths.outputs={tmp_path.as_posix()}",
        f"paths.cache={(tmp_path / 'cache').as_posix()}",
    )
    assert result.returncode == 0, result.stderr
    assert (tmp_path / "g" / "n" / "resolved_config.yaml").is_file()
    assert (tmp_path / "g" / "n" / "run.log").is_file()


def test_smoke_experiment_composes(tmp_path: Path):
    result = _run(
        "tinyearth.cli.inspect_config",
        "+experiment=smoke",
        "logging.rich=false",
        f"paths.outputs={tmp_path.as_posix()}",
        f"paths.cache={(tmp_path / 'cache').as_posix()}",
    )
    assert result.returncode == 0, result.stderr
    assert (tmp_path / "smoke" / "phase1" / "resolved_config.yaml").is_file()


def test_dry_run_writes_absolutely_nothing(tmp_path: Path):
    outputs = tmp_path / "outputs"
    result = _run(
        "tinyearth.cli.inspect_config",
        "--dry-run",
        "logging.rich=false",
        f"paths.outputs={outputs.as_posix()}",
        f"paths.cache={(tmp_path / 'cache').as_posix()}",
    )

    assert result.returncode == 0, result.stderr
    assert "nothing written to disk" in result.stdout + result.stderr
    # Neither TinyEarth's run directory nor Hydra's job metadata directory.
    assert not outputs.exists()


def test_invalid_override_fails_loudly():
    result = _run("tinyearth.cli.inspect_config", "--dry-run", "seed.value=not-an-int")
    assert result.returncode != 0
    assert "seed.value" in result.stdout + result.stderr


def test_unknown_key_fails_loudly():
    result = _run("tinyearth.cli.inspect_config", "--dry-run", "seed.nonexistent=1")
    assert result.returncode != 0
