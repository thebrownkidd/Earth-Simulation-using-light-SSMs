"""Execute the example notebooks.

"Visualise samples in a notebook" is a Phase 2 deliverable, and a notebook that
no longer runs is a silent failure -- nothing else in the suite touches it. This
executes it for real against synthetic data.

Skipped when the ``notebooks`` extra is not installed, so the default ``pip
install -e ".[dev]"`` workflow is unaffected.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from tinyearth.utils.paths import project_root

pytestmark = pytest.mark.slow

nbformat = pytest.importorskip("nbformat", reason="requires the 'notebooks' extra")
nbclient = pytest.importorskip("nbclient", reason="requires the 'notebooks' extra")
pytest.importorskip("matplotlib", reason="requires the 'notebooks' extra")

NOTEBOOKS = sorted((project_root() / "notebooks").glob("*.ipynb"))


def test_notebooks_exist():
    assert NOTEBOOKS, "no notebooks found under notebooks/"


@pytest.mark.parametrize("path", NOTEBOOKS, ids=lambda p: p.name)
def test_notebook_executes(path: Path, monkeypatch: pytest.MonkeyPatch):
    """Run every cell; any exception fails the test."""
    monkeypatch.setenv("MPLBACKEND", "Agg")
    os.environ["MPLBACKEND"] = "Agg"

    notebook = nbformat.read(path, as_version=4)
    client = nbclient.NotebookClient(notebook, timeout=600, kernel_name="python3")
    client.execute()


@pytest.mark.parametrize("path", NOTEBOOKS, ids=lambda p: p.name)
def test_notebook_outputs_are_cleared_in_git(path: Path):
    """Committed notebooks carry no outputs.

    Stored outputs bloat diffs and leak absolute paths from whoever ran them
    last.
    """
    notebook = nbformat.read(path, as_version=4)
    for index, cell in enumerate(notebook.cells):
        if cell.cell_type == "code":
            assert not cell.get("outputs"), f"{path.name} cell {index} has stored outputs"
            assert (
                cell.get("execution_count") is None
            ), f"{path.name} cell {index} has an execution count"
