# Contributing

TinyEarth is a research repository intended to be publishable. The bar is that a reader
can reproduce any reported number from the repository alone.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

pip install -e ".[dev]"
pre-commit install
```

## Quality gates

Four commands. All four must pass before anything is merged, and before a phase is
declared complete.

```bash
black src tests scripts        # format
ruff check src tests scripts   # lint
mypy                           # type check (strict)
pytest                         # test
```

`pre-commit install` wires these to run on commit.

For a fast inner loop, `pytest -m "not slow"` skips the subprocess CLI and notebook tests
(~15s instead of ~85s). Run the full suite before pushing.

### Formatting

black is the **only** formatter. Ruff runs as a linter with its formatter-overlapping
rules disabled, so the two cannot disagree. Do not enable `ruff format`.

### Type checking

mypy runs with `strict = true` over `src/` and `tests/`. Type hints are required
throughout `src/`; test functions are exempt from annotation requirements because their
signatures are fixture-driven.

If a third-party library lacks stubs, add it to the `ignore_missing_imports` override in
`pyproject.toml` rather than sprinkling `# type: ignore`. When an ignore *is* necessary,
scope it to a specific error code and comment why:

```python
torch.backends.cudnn.version()  # type: ignore[no-untyped-call]
```

## Phase discipline

Work proceeds in phases. **Do not start a phase before the previous one is verified.**

Each phase ends with:

1. All four quality gates green
2. Documentation in `docs/phase-N.md` covering objective, delivered scope, verification
   output, interpretation, and decisions worth revisiting
3. A commit that stands on its own

A broken previous phase is a blocker, not something to work around.

## Coding standards

**Type hints everywhere.** `from __future__ import annotations` at the top of every module.

**Document every public module, class and function.** Google-style docstrings, enforced by
ruff's `D` rules. Docstrings say *why*, not just *what* — the signature already says what.

**Declare `__all__`** in every module. `tests/test_package.py` enforces this.

**Small functions, no duplicated logic.** If two entry points need the same setup, it goes
in `tinyearth.bootstrap` or `tinyearth.utils`.

**No monolithic training scripts.** Entry points compose a config and delegate. Training
logic lives in `tinyearth.training` where it can be tested without a subprocess.

**Dataclasses for structured data**, especially config schemas and metric records.

**Prefer readable over clever.** A reviewer reproducing a result should not have to
decode anything.

## Explain trade-offs before implementing

If a design choice has real trade-offs, or a better abstraction becomes apparent, say so
before changing the architecture. Record the reasoning in `docs/project-structure.md`.
Decisions are cheap to make and expensive to reverse once experiments depend on them.

Equally: **do not overengineer.** Abstractions earn their place by removing duplication
that actually exists, not duplication that might.

## Adding components

Swappable research components — encoders, temporal backbones, decoders, losses — register
themselves so configs can select them by name:

```python
from tinyearth.models.temporal import TEMPORAL_BACKBONES   # Phase 4

@TEMPORAL_BACKBONES.register()
class MambaBackbone(TemporalBackbone):
    """Selective state space temporal backbone."""
```

Registered as `mamba_backbone` (snake_case of the class name), or pass an explicit key.
Duplicate keys raise rather than silently replacing — a silent replacement would change
which architecture an experiment trains.

## Adding a loss

New losses should be a new module plus a config entry, never an edit to the training loop.
If adding a loss requires touching the trainer, the loss interface is wrong — fix that
instead.

## Tests

- Compose the **real** `configs/` tree, not a fixture copy. A broken config should fail
  the suite rather than pass against a stale duplicate.
- Mark GPU-only tests `@pytest.mark.gpu` and skip on CPU.
- Mark tests over a few seconds `@pytest.mark.slow`.
- Test behaviour and error messages, not implementation details.

## Experiments

Every experiment config in `configs/experiment/` documents four things in its header:
objective, configuration, expected output, interpretation. See `smoke.yaml`.

An experiment that cannot state what result would *disconfirm* it is not ready to run.

## Datasets

**Never commit data, and never redistribute a dataset.** `data/` is git-ignored. Provide
download scripts and instructions instead.

Every dataset yields the contract in `tinyearth.datasets.types`. Adding a data source means
writing a reader, not changing the pipeline — see [`docs/datasets.md`](docs/datasets.md).

Tests run against **synthetic cubes written in the real on-disk format**, so they exercise
the production reader rather than a parallel implementation that could drift away from it.
When adding a format detail, add it to `minicube.py` and cover it there.

Two conventions are load-bearing and must not be changed casually:

- **`cldmsk == 1` means cloudy**; validity is `1 - cldmsk`. Inverting this trains the model
  exclusively on cloud, silently — losses still fall.
- **Statistics come from the training split only**, with masked pixels excluded.

## Notebooks

`tests/test_notebooks.py` executes every notebook under `notebooks/`, so a broken example
fails the suite. Clear outputs before committing; the test enforces this.

## Commits

Write commit messages that explain *why*. Keep each phase's work coherent — a reader
should be able to follow the project's development from the log.
