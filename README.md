# TinyEarth

**A controlled benchmark for measuring how small a State Space Model can get before it stops
forecasting the Earth's surface competitively.**

Four temporal architectures — a diagonal SSM, a selective SSM, a ConvLSTM and a temporal
transformer — swappable by a single config flag, compared at **matched parameter budgets**,
with quality and efficiency measured together on every run.

```
image encoder  ─→  temporal backbone  ─→  decoder  ─→  forecast
                   ▲
                   the only thing that changes between experiments
```

The encoder and decoder keep **byte-identical parameter counts** across every backbone
swap. That invariant is what makes a difference in results attributable to the architecture
rather than to accidental capacity — and it is enforced by tests, not by convention.

```bash
pip install -e ".[dev]"
tinyearth-train +experiment=ssm_smoke     # trains end to end in seconds, no dataset needed
```

---

## What's actually in here

| | |
| --- | --- |
| **4 temporal backbones** | `s4d`, `mamba`, `convlstm`, `transformer` — one interface, one config flag |
| **Full data pipeline** | EarthNet2021 reader, temporal windowing, cloud masking, deterministic splits, caching |
| **Training stack** | Trainer, optimiser/schedule construction, checkpointing, TensorBoard + JSONL + optional W&B |
| **10 metrics** | MAE, RMSE, PSNR, SSIM, SAM (all mask-aware) + parameters, FLOPs, peak memory, latency, throughput |
| **5 experiment sweeps** | scaling, history length, forecast horizon, hidden dimension, state dimension |
| **Calibrated size tiers** | ~2M / ~5M / ~10M / ~20M, measured per architecture |
| **809 tests** | strict mypy, ruff, black, CI on Linux + Windows |
| **Zero-download demo** | Synthetic data in the real on-disk format — everything runs without the 100 GB dataset |

---

## Three things that make it a *benchmark* rather than a model zoo

### 1. The comparison is controlled, and the control is tested

Adding a backbone changes nothing else. The interface is:

```python
encoder:  [B, T, C, H, W] → [B, T, D, h, w]
backbone: [B, T, D, h, w] → [B, K, D, h, w]     # the component under study
decoder:  [B, K, D, h, w] → [B, K, C, H, W]
```

Encoder and decoder hold **no temporal parameters at all** — frames are folded into the
batch dimension, so every unit of sequence-modelling capacity belongs to the backbone.
`test_frames_are_encoded_independently` pins this down by reversing frame order and
requiring the output reverse exactly.

### 2. Efficiency is a first-class result, not a footnote

Every run reports parameters, FLOPs, peak memory, latency and throughput **by default** —
no flag to remember. Timing handles CUDA's asynchrony and warmup properly, and reports a
median rather than a mean.

The efficiency numbers are why the project exists, so they can't be optional.

### 3. Correctness is verified against references, not against convergence

The S4D kernel is checked against a literal step-by-step recurrence to **1e-7**:

```
x_k = Ā·x_{k-1} + B̄·u_k    ←→    FFT convolution with K_k = C·Ā^k·B̄
```

This matters more than it sounds. An SSM with a subtly wrong kernel still trains, still
shows a falling loss, and still produces publishable-looking numbers. Metrics get the same
treatment — every one is tested against a closed-form value, not a plausible-looking one.

---

## A result the benchmark already produced

Measured at matched parameter budgets, 64×64 input, `T=4 → K=2`, CPU:

| At the 2M tier | `hidden_dim` | GFLOPs/sample | Latency |
| --- | ---: | ---: | ---: |
| `transformer` | 128 | **2.1** | **76 ms** |
| `convlstm` | 80 | 7.2 | 76 ms |
| `s4d` | 272 | 5.8 | 104 ms |
| `mamba` | 256 | 6.5 | 610 ms |

**Parameter efficiency is not FLOP efficiency.**

A diagonal SSM's temporal mixing costs `~4HN` parameters against attention's `~4H²`, so at
the same budget it affords 2.1× the width. But FLOPs scale with *width squared* in the
channel-mixing layers every architecture shares — so the SSM spends its parameter saving on
width and pays for that width in compute. The transformer is cheapest on both counts, at
every tier.

"How small can an SSM be?" turns out to have two different answers depending on whether
*small* means parameters or compute, and here they disagree. That is why this codebase
reports both by default.

> **Caveat worth stating loudly:** the usual asymptotic argument for SSMs — `O(T)` versus
> attention's `O(T²)` — does not apply here. This project's sequences top out at `T=8`,
> where that is 8 operations against 64. Earth observation minicubes are short by nature,
> which may simply make this an unfavourable setting for the architecture. Finding that out
> is a legitimate result.

---

## Honest status

**The platform is complete. The experiments have not been run.**

No model has been trained to convergence on real EarthNet2021 data, so **there is no
forecast-quality result in this repository**. Every quality number here came from synthetic
data over a handful of optimiser steps and is meaningless as science.

What *is* established and reproducible: the cost side — parameter counts, FLOPs and latency
at matched budgets — and that the whole stack is correct.

To produce real results you need the ~100 GB download and a GPU:

```bash
pip install -e ".[data]"
python scripts/download_earthnet2021.py --root data/earthnet2021

tinyearth-train --multirun +experiment=scaling data=earthnet2021 \
    model=s4d,mamba,convlstm,transformer \
    model.backbone.size=tiny,small,base,large

python scripts/collate_results.py --group scaling
```

Also outstanding: the frozen foundation-encoder comparison, GPU verification of mixed
precision and peak-memory measurement, and validating the EarthNet2021 format constants
against a real download rather than against documentation.

---

## The backbones

| Key | Architecture | Mixes over time by | Parallel in time | Autoregressive |
| --- | --- | --- | --- | --- |
| `s4d` | Diagonal SSM (Gu et al., 2022) | FFT convolution | ✅ | No |
| `mamba` | Selective SSM (Gu & Dao, 2023) | Input-dependent scan | ❌ | No |
| `convlstm` | ConvLSTM (Shi et al., 2015) | Gated recurrence | ❌ | **Yes** |
| `transformer` | Temporal transformer | Attention over `T` | ✅ | No |

`convlstm` is the only autoregressive one — worth stating whenever latency is compared,
since that difference is not about the mixing mechanism.

**Adding a fifth backbone:** subclass `TemporalBackbone`, register it, add a config, run
`scripts/calibrate_sizes.py`, add the name to one list in the tests. The parametrised
interface suite then applies to it automatically. Nothing else changes.

---

## Quick reference

```bash
tinyearth-info                              # environment and hardware report
tinyearth-config +experiment=smoke          # compose and validate a config
tinyearth-data   +experiment=data_smoke     # build the data pipeline and report on it
tinyearth-train  +experiment=ssm_smoke      # train end to end
tinyearth-model  --compare                  # size and cost of every backbone, no training

pytest                                      # 809 tests, ~3 min
pytest -m "not slow"                        # 784 tests, ~28 s
```

Every run writes to `outputs/<group>/<name>/`:

```
summary.json          flat headline numbers — scriptable across a sweep
metrics.jsonl         per-step and per-epoch history
resolved_config.yaml  the reproducibility record
run.log               full DEBUG log
best.ckpt last.ckpt   weights, optimiser state, RNG state, config
tensorboard/
```

---

## Layout

```
configs/                 Hydra tree: data / model / training / experiment
src/tinyearth/
├── models/
│   ├── base.py          Encoder / TemporalBackbone / Decoder interfaces
│   ├── temporal/        ◀── THE COMPONENT UNDER STUDY
│   ├── encoders/        CNN encoder (held fixed)
│   ├── decoders/        CNN decoder (held fixed)
│   ├── losses/          L1, L2, Charbonnier — registry-backed
│   └── sizes.py         calibrated parameter tiers
├── datasets/            EarthNet2021 pipeline, windowing, masking, splits
├── training/            trainer, optimisation, metric tracking
├── evaluation/          forecast-quality and efficiency metrics
├── config/  utils/  cli/
docs/  tests/  scripts/  experiments/  notebooks/
```

`configs/` sits outside the installable package deliberately — research configs are the
experiment record and are edited constantly, so burying them in a wheel is wrong.

---

## Documentation

| Document | Contents |
| --- | --- |
| [`docs/project-log.md`](docs/project-log.md) | **Development history, decisions, and every bug found** |
| [`docs/models.md`](docs/models.md) | Architecture, backbones, measured parameter counts |
| [`docs/evaluation.md`](docs/evaluation.md) | Metrics and how they are measured |
| [`docs/datasets.md`](docs/datasets.md) | Dataset format, splits, masking, normalisation |
| [`docs/reproducibility.md`](docs/reproducibility.md) | Seeding, determinism, and its cost |
| [`docs/configuration.md`](docs/configuration.md) | How the Hydra config system works |
| [`docs/project-structure.md`](docs/project-structure.md) | Layout and the reasoning behind it |
| [`docs/phase-1.md`](docs/phase-1.md) … [`phase-4.md`](docs/phase-4.md) | Per-phase deliverables and verification |
| [`CONTRIBUTING.md`](CONTRIBUTING.md) | Development workflow and coding standards |

---

## Engineering

```bash
black src tests scripts        # format (black is the only formatter)
ruff check src tests scripts   # lint, 17 rule families
mypy                           # strict, 80 source files clean
pytest                         # 809 tests
pre-commit install
```

Two conventions in the data pipeline are load-bearing and easy to get wrong, so both are
tested aggressively: **`cldmsk == 1` means cloudy** (inverting it trains the model
exclusively on cloud, and the loss still falls), and normalisation statistics must come from
the training split only.

---

## Datasets

TinyEarth **does not redistribute any dataset**. `data/` is git-ignored, and the default
dataset is synthetic so the repository runs out of the box.

## License

MIT — see [LICENSE](LICENSE).

<details>
<summary>Citations</summary>

```bibtex
@inproceedings{requena2021earthnet2021,
  title     = {EarthNet2021: A Large-Scale Dataset and Challenge for Earth Surface
               Forecasting as a Guided Video Prediction Task},
  author    = {Requena-Mesa, Christian and Benson, Vitus and Reichstein, Markus and
               Runge, Jakob and Denzler, Joachim},
  booktitle = {CVPR Workshops}, year = {2021},
}
@inproceedings{gu2022s4d,
  title     = {On the Parameterization and Initialization of Diagonal State Space Models},
  author    = {Gu, Albert and Gupta, Ankit and Goel, Karan and R{\'e}, Christopher},
  booktitle = {NeurIPS}, year = {2022},
}
@article{gu2023mamba,
  title   = {Mamba: Linear-Time Sequence Modeling with Selective State Spaces},
  author  = {Gu, Albert and Dao, Tri}, journal = {arXiv:2312.00752}, year = {2023},
}
@inproceedings{shi2015convlstm,
  title     = {Convolutional LSTM Network: A Machine Learning Approach for
               Precipitation Nowcasting},
  author    = {Shi, Xingjian and Chen, Zhourong and Wang, Hao and Yeung, Dit-Yan and
               Wong, Wai-Kin and Woo, Wang-chun},
  booktitle = {NeurIPS}, year = {2015},
}
```
</details>
