# TinyEarth

**How small can a State Space Model become while remaining competitive at Earth observation forecasting?**

TinyEarth is a research codebase investigating that single question. The objective is
**efficiency, not state-of-the-art accuracy**. Results are reported along five axes:

| Axis | Measured by |
| --- | --- |
| Parameter efficiency | parameter count vs. forecast quality |
| Memory | peak GPU memory during training and inference |
| Inference speed | latency, throughput |
| Training speed | wall-clock time per epoch |
| Scaling | quality as a function of model size |

The experimental design holds everything fixed except one component:

```
image encoder  ->  temporal backbone  ->  decoder  ->  forecast
                   ^^^^^^^^^^^^^^^^^
                   the only thing that varies
```

---

## Status

| Phase | Scope | State |
| --- | --- | --- |
| **1** | Repository scaffold, config system, tooling, utilities | **Complete** |
| **2** | EarthNet2021 dataset pipeline | **Complete** |
| 3 | ConvLSTM and temporal-transformer baselines, training loop, metrics | Not started |
| 4 | Tiny State Space Model backbones, scaling study | Not started |

Phases 1–2 deliver a runnable, tested, type-checked scaffold and a full data pipeline.
There is **no model yet** — that is by design. The default dataset is synthetic, so
everything runs without the ~100 GB EarthNet2021 download.
See [`docs/phase-1.md`](docs/phase-1.md) and [`docs/phase-2.md`](docs/phase-2.md).

---

## Installation

Requires Python 3.11+.

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

pip install -e ".[dev]"
```

Optional extras:

```bash
pip install -e ".[dev,wandb]"      # Weights & Biases tracking
pip install -e ".[dev,notebooks]"  # JupyterLab + matplotlib
```

For a CPU-only PyTorch build:

```bash
pip install -e ".[dev]" --extra-index-url https://download.pytorch.org/whl/cpu
```

### Verify the install

```bash
tinyearth-info                          # environment and hardware report
tinyearth-config +experiment=smoke      # compose and validate a config end to end
tinyearth-data   +experiment=data_smoke # build the data pipeline and report on it
pytest                                  # 430 tests
pytest -m "not slow"                    # ~15s, skips subprocess and notebook tests
```

---

## Usage

Every experiment is reproducible from a single Hydra config.

```bash
# defaults
tinyearth-config

# a named experiment
tinyearth-config +experiment=smoke

# ad-hoc overrides
tinyearth-config run.name=ablation seed.value=7 run.device=cpu

# inspect without writing anything
tinyearth-config --dry-run

# the data pipeline
tinyearth-data                                   # synthetic (default, no download)
tinyearth-data data=earthnet2021                 # the real dataset
tinyearth-data data.history_length=8 data.horizon=4
```

Each run writes to `outputs/<run.group>/<run.name>/`:

```
outputs/smoke/phase1/
├── resolved_config.yaml   # fully resolved config -- the reproducibility record
└── run.log                # complete DEBUG-level log
```

To re-run an experiment exactly, point Hydra at its saved `resolved_config.yaml`.

---

## Repository layout

```
configs/                 Hydra configuration tree
├── config.yaml          root config; composes the structured schema
├── experiment/          complete named experiments
├── data/                synthetic (default) and earthnet2021
└── model|training/      populated in Phases 3-4

src/tinyearth/
├── bootstrap.py         shared run initialisation for every entry point
├── config/              structured schemas, ConfigStore registration, resolution
├── utils/               determinism, logging, devices, paths, registry
├── datasets/            EarthNet2021 pipeline, windowing, masking, splits
├── models/              Phase 3-4
│   ├── encoders/  temporal/  decoders/  losses/
├── training/            Phase 3
├── evaluation/          Phase 3
└── cli/                 console entry points

docs/                    design notes and per-phase documentation
tests/                   pytest suite
scripts/                 dataset download and maintenance scripts (Phase 2)
experiments/             experiment definitions and result tables
notebooks/               exploratory analysis
```

`configs/` deliberately sits **outside** the installable package: research configs are
edited constantly and should not be buried in a wheel. See
[`docs/project-structure.md`](docs/project-structure.md) for the reasoning.

---

## Documentation

| Document | Contents |
| --- | --- |
| [`docs/project-structure.md`](docs/project-structure.md) | Layout and the design decisions behind it |
| [`docs/configuration.md`](docs/configuration.md) | How the Hydra config system works |
| [`docs/reproducibility.md`](docs/reproducibility.md) | Seeding, determinism, and its cost |
| [`docs/datasets.md`](docs/datasets.md) | Dataset format, splits, masking, normalisation |
| [`docs/phase-1.md`](docs/phase-1.md) | What Phase 1 delivers and how to verify it |
| [`docs/phase-2.md`](docs/phase-2.md) | What Phase 2 delivers and how to verify it |
| [`CONTRIBUTING.md`](CONTRIBUTING.md) | Development workflow and coding standards |

---

## Development

```bash
black src tests scripts        # format (black is the only formatter)
ruff check src tests scripts   # lint
mypy                           # type check (strict)
pytest                         # test
pre-commit install             # run all of the above on commit
```

All four must pass before a phase is considered complete.

---

## Datasets

TinyEarth **does not redistribute any dataset**. The `data/` directory is git-ignored.

You do not need EarthNet2021 to run this repository — the default dataset is synthetic:

```bash
tinyearth-data +experiment=data_smoke
```

For the real thing:

```bash
pip install -e ".[data]"
python scripts/download_earthnet2021.py --root data/earthnet2021
```

See [`docs/datasets.md`](docs/datasets.md) for the format, the mask polarity convention,
and how the validation split is derived.

---

## License

MIT — see [LICENSE](LICENSE).
