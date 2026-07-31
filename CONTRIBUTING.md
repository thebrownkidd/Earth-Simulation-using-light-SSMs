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
(~30s instead of ~2min). Run the full suite before pushing.

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
from tinyearth.models.base import TemporalBackbone
from tinyearth.models.temporal import TEMPORAL_BACKBONES

@TEMPORAL_BACKBONES.register("mamba")
class MambaBackbone(TemporalBackbone):
    """Selective state space temporal backbone."""

    def forward(self, latents: Tensor, horizon: int) -> Tensor:
        """[B, T, D, h, w] -> [B, K, D, h, w]."""
```

Registered under the given key, or the snake_case class name if omitted. Duplicate keys
raise rather than silently replacing — a silent replacement would change which architecture
an experiment trains.

**Then add the name to `BACKBONE_NAMES` in `tests/test_models.py`.** The parametrised
`TestBackboneContract` applies the full interface suite automatically, and
`TestControlledComparison` checks the fixed components stay identical.

## The controlled comparison

The project's central claim is that **only the temporal backbone changes** between
experiments. Two rules protect it:

- **Never add capacity to the encoder or decoder for one backbone's benefit.** If they
  differ between runs, a difference in results is not attributable.
- **Never give a backbone spatial modelling.** The encoder and decoder own that. A backbone
  that also attends spatially has capacity the others lack.

`tests/test_models.py::TestControlledComparison` and
`tests/test_cli.py::test_swapping_the_backbone_changes_only_the_backbone` enforce both.

## Adding a loss

A new module plus a config entry, never an edit to the training loop. If adding a loss
requires touching the trainer, the loss interface is wrong — fix that instead.

Implement `ForecastLoss`, register in `LOSSES`, and use `masked_mean` for the reduction
rather than reimplementing it. Getting the masked reduction wrong — dividing by the total
pixel count instead of the valid one — silently scales the loss down with cloudiness.

## Adding a metric

Test it against a **closed-form value**, not a plausible-looking one. A metric that is
merely plausible is worse than none: it produces numbers that look publishable and are
wrong. Add a masking test too — corrupting masked pixels must not change the result.

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

## Efficiency measurements

Efficiency is a primary result, not a diagnostic. When reporting one:

- State the determinism setting. Deterministic kernels are not what a user deploys, so
  benchmarking them understates real throughput. Use `seed.deterministic=false`.
- Attach `tinyearth-info` output. A latency number without its hardware is not a result.
- Do not hand-roll timing. `tinyearth.evaluation.efficiency` handles CUDA synchronisation
  and warmup; naive timing overstates speed in two separate ways.

## Notebooks

`tests/test_notebooks.py` executes every notebook under `notebooks/`, so a broken example
fails the suite. Clear outputs before committing; the test enforces this.

## Reporting results

The bar is that a reader can reproduce any reported number from the repository alone.

- **Never report a quality number from synthetic data.** It is meaningless. The default
  dataset is synthetic, so this is an easy mistake to make.
- **Report FLOPs and latency next to parameter count.** They disagree here: the SSM is more
  parameter-efficient and *less* FLOP-efficient. Quoting one alone misleads.
- **State the architectural caveats.** `convlstm` is autoregressive and the other three are
  not; `mamba`'s scan is sequential; the `O(T)` vs `O(T²)` argument does not apply at
  `T ≤ 8`. Each of these makes some architecture look better for reasons unrelated to the
  claim being made.
- **Attach the hardware.** `tinyearth-info` output belongs with any efficiency result.

## Commits

Write commit messages that explain *why*. Keep each phase's work coherent — a reader
should be able to follow the project's development from the log.

Record decisions, reversals and bugs in [`docs/project-log.md`](docs/project-log.md). The
reasons behind a research codebase decay faster than the code does.
