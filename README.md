# TinyEarth

**How small can a State Space Model become while remaining competitive at Earth observation
forecasting?**

TinyEarth is a research codebase built to answer one question carefully. The objective is
**efficiency, not state-of-the-art accuracy** — and the infrastructure is designed so that
every efficiency claim comes with the hardware, the configuration, and the caveats attached.

```
image encoder  ->  temporal backbone  ->  decoder  ->  forecast
                   ^^^^^^^^^^^^^^^^^
                   the only thing that varies
```

Swapping the backbone is a single config override, and the encoder and decoder keep
byte-identical parameter counts across the swap — which is what makes any difference in
results *attributable*. That invariant is enforced by tests, not by convention.

---

## Status

| Phase | Scope | State |
| --- | --- | --- |
| **1** | Repository scaffold, config system, tooling, utilities | **Complete** |
| **2** | EarthNet2021 dataset pipeline | **Complete** |
| **3** | ConvLSTM and temporal-transformer baselines, training, metrics | **Complete** |
| **4** | State Space Models, size tiers, experiment sweeps | **Complete** |

**The infrastructure is finished. The experiments have not been run.**

No model has been trained to convergence on real EarthNet2021 data, so **this repository
contains no forecast-quality result**. Every quality number here was produced on synthetic
data over a handful of optimiser steps and is meaningless as science. What *is* established
is cost: parameter counts, FLOPs and latency at matched budgets, measured and reproducible.

See [`docs/project-log.md`](docs/project-log.md) for the full development history and honest
limits.

---

## Quick start

Requires Python 3.11+. **No dataset download needed** — the default data is synthetic.

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -e ".[dev]"

tinyearth-train +experiment=ssm_smoke      # trains an SSM end to end, in seconds
```

That single command exercises the whole stack: config composition, data generation, model
assembly, a masked loss, the training loop, five forecast metrics, five efficiency metrics,
checkpointing and metric tracking.

---

## Installation

```bash
pip install -e ".[dev]"                            # core + quality tooling
pip install -e ".[dev,notebooks]"                  # + JupyterLab, matplotlib
pip install -e ".[dev,wandb]"                      # + Weights & Biases
pip install -e ".[dev,data]"                       # + EarthNet2021 downloader

# CPU-only PyTorch
pip install -e ".[dev]" --extra-index-url https://download.pytorch.org/whl/cpu
```

### Verify

```bash
tinyearth-info                              # environment and hardware report
tinyearth-config +experiment=smoke          # compose and validate a config
tinyearth-data   +experiment=data_smoke     # build the data pipeline and report
tinyearth-train  +experiment=ssm_smoke      # train end to end
tinyearth-model  --compare                  # model size and cost, without training

pytest                                      # 809 tests, ~3 min
pytest -m "not slow"                        # 784 tests, ~28 s
```

---

## The four backbones

| Key | Architecture | Mixes over time by | Autoregressive |
| --- | --- | --- | --- |
| `s4d` | Diagonal SSM (Gu et al., 2022) | FFT convolution — parallel in time | No |
| `mamba` | Selective SSM (Gu and Dao, 2023) | Input-dependent scan — sequential | No |
| `convlstm` | ConvLSTM (Shi et al., 2015) | Gated recurrence — sequential | **Yes** |
| `transformer` | Temporal transformer | Attention over `T` — parallel | No |

```bash
tinyearth-train model=s4d
tinyearth-train model=mamba
tinyearth-train model=convlstm
tinyearth-train model=transformer
```

All four are available at four calibrated size tiers — `tiny` ~2M, `small` ~5M, `base` ~10M,
`large` ~20M — so comparisons happen at **matched parameter budgets**:

```bash
tinyearth-train model=s4d model.backbone.size=base
```

---

## What the cost measurements already show

Measured at matched budgets, 64×64 input, `T=4 → K=2`, CPU. Full table in
[`docs/phase-4.md`](docs/phase-4.md).

| At the 2M tier | `hidden_dim` | GFLOPs/sample | Latency |
| --- | ---: | ---: | ---: |
| `transformer` | 128 | 2.1 | 76 ms |
| `convlstm` | 80 | 7.2 | 76 ms |
| `s4d` | 272 | 5.8 | 104 ms |
| `mamba` | 256 | 6.5 | 610 ms |

**Parameter efficiency is not FLOP efficiency.** The SSM's cheap temporal mixing lets it
afford 2.1× the transformer's width at the same parameter budget — but FLOPs scale with
width squared in the channel-mixing layers every architecture shares. The SSM spends its
parameter saving on width, and pays for that width in compute.

So *"how small can an SSM be?"* has two different answers depending on whether "small" means
parameters or compute, and here they disagree. This repository reports both by default,
because a result quoting only parameter counts would be telling half the story.

**A caveat that matters more than it looks:** the asymptotic argument for SSMs — `O(T)`
against attention's `O(T²)` — does not apply at these sequence lengths. The history sweep
tops out at `T=8`, where that is 8 operations against 64. TinyEarth's sequences are short by
nature, which may simply make this an unfavourable setting for the architecture. That is
worth finding out, and it is what the sweeps are for.

---

## Running experiments

Every experiment is reproducible from a single Hydra config.

```bash
# Scaling: four backbones x four budgets
tinyearth-train --multirun +experiment=scaling \
    model=s4d,mamba,convlstm,transformer \
    model.backbone.size=tiny,small,base,large

# How much context the forecast needs
tinyearth-train --multirun +experiment=history_length data.history_length=2,4,6,8

# How quality decays with forecast distance
tinyearth-train --multirun +experiment=horizon data.horizon=1,2,4,8

# Width scaling within one architecture
tinyearth-train --multirun +experiment=hidden_dim \
    model.backbone.kwargs.hidden_dim=64,128,256,512

# The SSM-specific axis: state size is the CHEAP capacity knob
tinyearth-train --multirun +experiment=state_dim \
    model.backbone.kwargs.state_dim=8,16,32,64,128

# Collate any sweep into a table
python scripts/collate_results.py --group scaling
```

Each run writes to `outputs/<group>/<name>/`:

```
summary.json          # flat headline numbers — scriptable across a sweep
metrics.jsonl         # per-step and per-epoch history
resolved_config.yaml  # the reproducibility record
run.log               # full DEBUG log
best.ckpt, last.ckpt  # weights, optimiser state, RNG state, config
tensorboard/
```

Every experiment config documents four things in its header: **objective, configuration,
expected output, interpretation** — including what the result does *not* show.

---

## Repository layout

```
configs/                 Hydra configuration tree
├── config.yaml          root config; composes the structured schema
├── data/                synthetic (default) and earthnet2021
├── model/               s4d, mamba, convlstm, transformer
├── training/            optimiser, schedule, checkpointing, metrics
└── experiment/          complete named experiments and sweeps

src/tinyearth/
├── bootstrap.py         shared run initialisation for every entry point
├── config/              structured schemas, ConfigStore, resolution
├── datasets/            EarthNet2021 pipeline, windowing, masking, splits
├── models/
│   ├── base.py          Encoder / TemporalBackbone / Decoder interfaces
│   ├── sizes.py         calibrated parameter tiers
│   ├── encoders/        CNN encoder (held fixed)
│   ├── temporal/        THE COMPONENT UNDER STUDY
│   ├── decoders/        CNN decoder (held fixed)
│   └── losses/          L1, L2, Charbonnier; registry-backed
├── training/            trainer, optimiser, schedules, metric tracking
├── evaluation/          forecast-quality and efficiency metrics
├── utils/               determinism, logging, devices, paths, registry
└── cli/                 console entry points

docs/  tests/  scripts/  experiments/  notebooks/
```

`configs/` sits **outside** the installable package on purpose: research configs are the
experiment record and are edited constantly, so burying them in a wheel is wrong. See
[`docs/project-structure.md`](docs/project-structure.md).

---

## Documentation

| Document | Contents |
| --- | --- |
| [`docs/project-log.md`](docs/project-log.md) | **Development history, decisions, and what went wrong** |
| [`docs/project-structure.md`](docs/project-structure.md) | Layout and the reasoning behind it |
| [`docs/configuration.md`](docs/configuration.md) | How the Hydra config system works |
| [`docs/datasets.md`](docs/datasets.md) | Dataset format, splits, masking, normalisation |
| [`docs/models.md`](docs/models.md) | Architecture, backbones, measured parameter counts |
| [`docs/evaluation.md`](docs/evaluation.md) | Metrics and how they are measured |
| [`docs/reproducibility.md`](docs/reproducibility.md) | Seeding, determinism, and its cost |
| [`docs/phase-1.md`](docs/phase-1.md) … [`phase-4.md`](docs/phase-4.md) | Per-phase deliverables and verification |
| [`CONTRIBUTING.md`](CONTRIBUTING.md) | Development workflow and coding standards |

---

## Development

Four gates. All four must pass before anything is merged.

```bash
black src tests scripts        # format (black is the only formatter)
ruff check src tests scripts   # lint
mypy                           # type check (strict)
pytest                         # test
pre-commit install             # run all of the above on commit
```

### Adding a backbone

1. Subclass `TemporalBackbone`, implementing
   `forward(latents: [B,T,D,h,w], horizon: int) -> [B,K,D,h,w]`.
2. Decorate with `@TEMPORAL_BACKBONES.register("name")`.
3. Add `configs/model/name.yaml`.
4. Run `python scripts/calibrate_sizes.py` and add a row to `SIZE_TIERS`.
5. Add the name to `BACKBONE_NAMES` in `tests/test_models.py`.

The parametrised interface suite then applies automatically. Nothing else changes — that is
the design working.

---

## Datasets

TinyEarth **does not redistribute any dataset**. `data/` is git-ignored.

You do not need EarthNet2021 to run this repository. For the real thing:

```bash
pip install -e ".[data]"
python scripts/download_earthnet2021.py --root data/earthnet2021
tinyearth-train data=earthnet2021 model=s4d model.backbone.size=base
```

Two conventions in [`docs/datasets.md`](docs/datasets.md) are load-bearing and easy to get
wrong: **`cldmsk == 1` means cloudy** (inverting it trains the model exclusively on cloud,
and losses still fall), and normalisation statistics must come from the training split only.

---

## Citation

If you use this code, please also cite the dataset and the architectures:

```bibtex
@inproceedings{requena2021earthnet2021,
  title     = {EarthNet2021: A Large-Scale Dataset and Challenge for Earth Surface
               Forecasting as a Guided Video Prediction Task},
  author    = {Requena-Mesa, Christian and Benson, Vitus and Reichstein, Markus and
               Runge, Jakob and Denzler, Joachim},
  booktitle = {CVPR Workshops},
  year      = {2021},
}

@inproceedings{gu2022s4d,
  title     = {On the Parameterization and Initialization of Diagonal State Space Models},
  author    = {Gu, Albert and Gupta, Ankit and Goel, Karan and R{\'e}, Christopher},
  booktitle = {NeurIPS},
  year      = {2022},
}

@article{gu2023mamba,
  title   = {Mamba: Linear-Time Sequence Modeling with Selective State Spaces},
  author  = {Gu, Albert and Dao, Tri},
  journal = {arXiv:2312.00752},
  year    = {2023},
}

@inproceedings{shi2015convlstm,
  title     = {Convolutional LSTM Network: A Machine Learning Approach for
               Precipitation Nowcasting},
  author    = {Shi, Xingjian and Chen, Zhourong and Wang, Hao and Yeung, Dit-Yan and
               Wong, Wai-Kin and Woo, Wang-chun},
  booktitle = {NeurIPS},
  year      = {2015},
}
```

---

## License

MIT — see [LICENSE](LICENSE).
