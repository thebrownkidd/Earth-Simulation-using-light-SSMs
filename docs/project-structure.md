# Project structure

This document records the layout and, more importantly, the reasoning behind the
decisions that were not obvious. Layout choices are cheap to make and expensive to
reverse once experiments depend on them.

## Layout

```
Earth Simulation using light SSMs/
├── configs/                    Hydra configuration tree
│   ├── config.yaml             root config
│   ├── experiment/             complete named experiments
│   ├── data/                   Phase 2
│   ├── model/                  Phases 3-4
│   └── training/               Phase 3
├── docs/                       design notes, per-phase documentation
├── experiments/                experiment definitions and result tables
├── notebooks/                  exploratory analysis
├── scripts/                    dataset download and maintenance scripts
├── src/tinyearth/              the installable package
│   ├── __init__.py
│   ├── bootstrap.py            shared run initialisation
│   ├── cli/                    console entry points
│   ├── config/                 schemas, ConfigStore registration, resolution
│   ├── datasets/               Phase 2
│   ├── evaluation/             Phase 3
│   ├── models/                 Phases 3-4
│   │   ├── encoders/
│   │   ├── temporal/           the only component that varies across experiments
│   │   ├── decoders/
│   │   └── losses/
│   ├── training/               Phase 3
│   └── utils/                  determinism, logging, devices, paths, registry
├── tests/                      pytest suite
├── pyproject.toml
└── .pre-commit-config.yaml
```

## Decisions

### `src/` layout rather than a top-level `tinyearth/` package

The original sketch placed `configs/`, `docs/`, `tests/` and `notebooks/` as siblings of
the model code inside one `tinyearth/` directory. Those four are not importable Python
and should not ship inside the wheel, so the package moved under `src/`.

The concrete benefit: with a flat layout, `import tinyearth` from the repository root
picks up the *source directory* rather than the installed package. That difference hides
packaging bugs — a missing `__init__.py` or an unlisted subpackage — until someone
installs the project somewhere else. With `src/`, the only importable copy is the
installed one, so tests exercise what users get.

Cost: one extra directory level.

### `configs/` lives outside the package

Hydra configs could ship as package data. They do not, because in a research repo configs
are *edited constantly* — they are the experiment record, not code assets. Keeping them at
the repository root means they are versioned, diffable and editable without reinstalling.

This has one consequence worth knowing about: `@hydra.main(config_path="../../configs")`
does **not** work under `pip install -e .`, because Hydra resolves relative config paths as
importable module paths. Every entry point therefore goes through
`tinyearth.cli._hydra.run_with_hydra`, which passes the absolute directory found by
`tinyearth.utils.paths.project_root()`. That works identically for source checkouts,
editable installs and wheels.

### Path discovery via a project-root anchor

Hydra changes the process working directory at run start, so `Path("configs")` is
unreliable. `tinyearth.utils.paths.project_root()` walks up from the package until it
finds the `pyproject.toml` declaring `name = "tinyearth"`, and is overridable with the
`TINYEARTH_ROOT` environment variable for non-editable installs. Everything
repository-relative resolves through it.

### A component registry

`tinyearth.utils.registry.Registry` maps names to classes, one registry per component
family. The central experimental claim — *only the temporal backbone changes between
runs* — is only enforceable if swapping a backbone is a config edit rather than a code
edit.

Hydra's `_target_` instantiation does something similar, and the two compose fine. The
registry is preferred for the small set of swappable research components for two reasons:

1. `_target_` puts fully-qualified import paths into configs, so renaming a module
   invalidates every published experiment config.
2. A registry can be **enumerated**. `list(TEMPORAL_BACKBONES)` is how the scaling sweep
   and the documentation discover what exists.

### Console logging is separate from metric tracking

`tinyearth.utils.logging` handles human-readable diagnostics only. Experiment metric
tracking (TensorBoard, optional W&B) is a separate interface introduced in Phase 3 under
`tinyearth.training`. Keeping them apart means a metric backend can be swapped without
touching diagnostics, and library code can log freely without importing a tracking
dependency.

### Entry points orchestrate, they do not implement

Every console command composes a config, calls `tinyearth.bootstrap.initialise_run`, and
delegates to library code. No training logic lives in a script. This keeps the training
loop testable without spawning a subprocess, and avoids the monolithic-script failure mode
that research repos drift into.

### Phase 2-4 packages exist but are empty

`datasets/`, `models/` and `training/` contain only a documented `__init__.py` stating what
the phase will add. This keeps the import graph and the documentation honest: the structure
is visible, and nothing pretends to work that does not.

Correspondingly, `TinyEarthConfig.data`, `.model` and `.training` are typed `Any` rather
than given invented schemas. An honest hole beats a fabricated one that has to be rewritten.
