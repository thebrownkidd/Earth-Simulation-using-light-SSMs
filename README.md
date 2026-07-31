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
| 2 | EarthNet2021 dataset pipeline | Not started |
| 3 | ConvLSTM and temporal-transformer baselines, training loop, metrics | Not started |
| 4 | Tiny State Space Model backbones, scaling study | Not started |

Phase 1 delivers a runnable, tested, type-checked scaffold. There is **no model and no
data yet** — that is by design. See [`docs/phase-1.md`](docs/phase-1.md).

---

## Installation

Requires Python 3.10+.

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
tinyearth-info                     # environment and hardware report
tinyearth-config +experiment=smoke # compose and validate a config end to end
pytest                             # 137 tests
pytest -m "not slow"               # ~8s, skips the subprocess CLI tests
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
└── data|model|training/ populated in Phases 2-4

src/tinyearth/
├── bootstrap.py         shared run initialisation for every entry point
├── config/              structured schemas, ConfigStore registration, resolution
├── utils/               determinism, logging, devices, paths, registry
├── datasets/            Phase 2
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
| [`docs/phase-1.md`](docs/phase-1.md) | What Phase 1 delivers and how to verify it |
| [`CONTRIBUTING.md`](CONTRIBUTING.md) | Development workflow and coding standards |

---

## Development

```bash
black src tests          # format (black is the only formatter)
ruff check src tests     # lint
mypy                     # type check (strict)
pytest                   # test
pre-commit install       # run all of the above on commit
```

All four must pass before a phase is considered complete.

---

## Datasets

TinyEarth **does not redistribute any dataset**. Phase 2 adds download instructions and
scripts for EarthNet2021. The `data/` directory is git-ignored.

---

## License

MIT — see [LICENSE](LICENSE).
