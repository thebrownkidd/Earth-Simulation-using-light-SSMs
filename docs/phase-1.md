# Phase 1 — Repository setup

**Status: complete and verified.** Superseded by [phase-4.md](phase-4.md); see also [project-log.md](project-log.md).

## Objective

Establish a research codebase that later phases can build on without rework: dependency
management, a configuration system, quality tooling, and the reproducibility utilities
every experiment depends on.

Phase 1 contains **no model and no dataset**. That is the intended scope.

## Delivered

| Requirement | Implementation |
| --- | --- |
| Repository structure | `src/` layout, see [project-structure.md](project-structure.md) |
| Dependency management | `pyproject.toml`, extras `dev` / `wandb` / `notebooks` |
| `pip install -e .` | Verified; console scripts registered |
| Formatting | black (line length 100), the sole formatter |
| Linting | ruff, 15 rule families, formatter-overlapping rules disabled |
| Type checking | mypy `strict = true`, clean across 34 files |
| Deterministic utilities | `tinyearth.utils.seed` |
| Logging utilities | `tinyearth.utils.logging` |
| Configuration system | `tinyearth.config` + `configs/`, Hydra structured configs |
| README | [`../README.md`](../README.md) |
| Contributing guide | [`../CONTRIBUTING.md`](../CONTRIBUTING.md) |

Beyond the checklist, three pieces were added because later phases would otherwise
duplicate them:

- **`tinyearth.utils.device`** — device resolution and hardware provenance. Efficiency
  claims are meaningless without the hardware record, and Phase 3 metrics need this.
- **`tinyearth.utils.registry`** — typed component registry. Makes "only the temporal
  backbone changes" a config edit rather than a code edit.
- **`tinyearth.bootstrap`** — shared run initialisation, so entry points stay thin.

## Configuration

```bash
tinyearth-config +experiment=smoke
```

See [`configs/experiment/smoke.yaml`](../configs/experiment/smoke.yaml).

## Expected output

Console: a run header, the fully resolved config, the hardware record, and resolved paths.

On disk:

```
outputs/smoke/phase1/
├── resolved_config.yaml
└── run.log
```

## Verification

All four gates pass on Python 3.13.7 / torch 2.13.0+cpu / Windows 11:

```bash
$ pytest -q
137 passed, 1 skipped

$ ruff check src tests
All checks passed!

$ black --check src tests
35 files would be left unchanged.

$ mypy
Success: no issues found in 35 source files
```

The one skip is `test_cuda_resolves_with_an_explicit_index`, which requires CUDA. It is
marked `@pytest.mark.gpu` and will run on GPU hardware.

The subprocess CLI tests take about a minute. For a fast inner loop:

```bash
pytest -q -m "not slow"    # ~8s
```

Test coverage by area:

| Area | Tests | Notable cases |
| --- | --- | --- |
| `test_seed.py` | 12 | reproducibility, RNG isolation, per-worker seeding |
| `test_registry.py` | 11 | duplicate rejection, error messages, kwarg forwarding |
| `test_config.py` | 20 | real config tree composition, type rejection, fingerprints |
| `test_device.py` | 13 | `auto` resolution, loud failure on missing CUDA |
| `test_logging.py` | 13 | handler idempotency, file capture below console level |
| `test_paths.py` | 9 | root discovery, env-var override |
| `test_bootstrap.py` | 7 | run dir creation, config persistence, dry run |
| `test_cli.py` | 7 | subprocess end-to-end, `--dry-run` leaves nothing behind |
| `test_package.py` | 46 | every module imports, is documented, declares `__all__` |

The CLI tests earned their cost immediately: they caught that `--dry-run` still wrote
Hydra job metadata, and then that the fix broke on Windows short paths (`ARPITG~1`),
because `~` is a metacharacter in Hydra's override grammar. Neither was visible from
in-process testing.

## Interpretation

A green suite means the scaffold is sound: configs compose and validate, seeding works,
paths resolve under Hydra's directory changes, and the package installs cleanly.

It says **nothing** about forecasting quality. No model exists.

## Decisions worth revisiting later

1. **`data`/`model`/`training` are typed `Any`.** Phase 2 should replace `data` with a
   real dataclass and add a composition test, rather than extending the placeholder.
2. **Registry vs. Hydra `_target_`.** The registry is chosen for swappable research
   components; `_target_` remains available for one-off objects like optimisers. If
   Phase 4 finds the split confusing, consolidate then — not now.
3. **mypy runs at `python_version = "3.12"`, not the 3.11 floor.** numpy's bundled stubs
   use `type` statements that only parse under 3.12+. Ruff's `target-version = "py311"`
   enforces the language floor instead. If numpy's stubs change, revisit.
4. **No CI matrix across Python versions yet.** `.github/workflows/ci.yml` tests 3.11 and
   3.13. Add intermediate versions only if a failure appears.

## Next: Phase 2

Dataset pipeline for EarthNet2021 — download instructions (no redistribution), a
`Dataset` returning `{"images": Tensor[T,C,H,W], "target": Tensor[C,H,W], "metadata": ...}`,
dataloaders, temporal sequence generation, normalisation, optional cloud masking,
train/val/test splits, a visualisation notebook, and unit tests.

Phase 2 must not begin until this page's verification block reproduces.
