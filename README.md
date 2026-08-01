# TinyEarth

**A controlled benchmark for measuring how small a State Space Model can get before it stops
forecasting the Earth's surface competitively.**

Four temporal architectures — a diagonal SSM, a selective SSM, a ConvLSTM and a temporal
transformer — swappable by one config flag, compared at **matched parameter budgets**, with
quality and efficiency measured on every run.

```
image encoder  ─→  temporal backbone  ─→  decoder  ─→  forecast
                   ▲
                   the only thing that changes between experiments
```

The encoder and decoder keep **byte-identical parameter counts** across every backbone swap.
That invariant is what makes a difference attributable to the architecture rather than to
accidental capacity — and it is enforced by tests, not by convention.

```bash
pip install -e ".[dev]"
tinyearth-train +experiment=ssm_smoke      # trains end to end in seconds, no dataset needed
```

---

## Findings

Two families of result. The **quality** findings are trained on real EarthNet2021 data; the
**cost** findings are dataset-free and reproducible in minutes on any CPU.

---

### 1. On real data at a matched budget, the four architectures are nearly tied — and only just beat doing nothing

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/figures/quality-dark.png">
  <img alt="Left: forecast error against lead time, showing persistence best at short range and worst at long range. Right: error against inference latency, with S4D lowest-error but slowest." src="docs/figures/quality-light.png">
</picture>

Official EarthNet2021 protocol — 10 Sentinel-2 frames (50 days) in, 20 frames (100 days)
out — on 161 held-out validation windows scored at full 128×128 resolution.

| | val MAE ↓ | RMSE ↓ | SSIM ↑ | latency | parameters |
| --- | --- | --- | --- | --- | --- |
| **S4D (SSM)** | **0.0282** | **0.0421** | 0.739 | 611 ms | 2.08M |
| Transformer | 0.0282 | 0.0424 | 0.762 | 384 ms | 2.11M |
| ConvLSTM | 0.0294 | 0.0439 | 0.740 | 367 ms | 2.22M |
| *Persistence* | *0.0299* | *0.0445* | ***0.803*** | *free* | *0* |
| *Climatology* | *0.0352* | *0.0495* | *0.782* | *free* | *0* |

**Mamba is missing from this table, and the reason is the result.** Its selective scan is a
sequential Python loop over `T+K` steps with no fused kernel, which makes it **~5× slower to
train than S4D** — 11.3 s per optimiser step against 2.2 s, measured, and flat in batch size
(0.64, 0.65, 0.71 samples/s at batch 2, 4, 8), so no batching trick recovers it. A single
Mamba epoch costs more than one epoch of the other three architectures combined. At a fixed
compute budget on a CPU the second SSM is simply not affordable — a practical finding about
selective SSMs without CUDA kernels, not an omission.

**S4D and the transformer are indistinguishable** — 0.02815 against 0.02819, a 0.1% gap on
161 windows. Both beat persistence, but by **5.8%**. That margin is the honest headline: a
2M-parameter model, trained for six epochs on a laptop CPU, is barely better than repeating
the last picture you saw.

**Persistence wins on SSIM**, and beats every learned model on it. The learned forecasts are
blurry — visible directly below — and structural similarity punishes blur where mean absolute
error rewards it. Reporting only MAE would have hidden that completely.

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/figures/forecast-0-dark.png">
  <img alt="Filmstrip: 10 observed context frames, then ground truth, S4D prediction and absolute error at six lead times out to 100 days." src="docs/figures/forecast-0-light.png">
</picture>

The model reproduces the field boundaries and the scene's layout, and then holds them almost
static across 100 days — the characteristic failure of an L1-trained video predictor at a
small budget. Cloud-covered frames are labelled rather than silently rendered black, and
error is not scored where the truth was never observed.

Every panel shares one contrast stretch, computed from the observed frames. Restretching each
panel to its own range is the standard way to make a forecast look better than it is, and it
would have hidden exactly the flatness that is the most honest thing in this figure.

### 2. But the learned models predict the *season*, and the references cannot

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/figures/ndvi-0-dark.png">
  <img alt="NDVI maps and mean-greenness trajectory. The learned models track the seasonal rise in vegetation; persistence and climatology stay flat." src="docs/figures/ndvi-0-light.png">
</picture>

The aggregate MAE hides where the difference actually lives. Split the error by lead time
(left panel of the quality figure) and the two regimes separate cleanly:

- **Short range (< ~35 days): persistence is the best forecaster available.** Nothing has
  changed yet, so echoing the last observation is close to optimal, and every learned model
  is worse than it.
- **Long range (> ~35 days): persistence collapses** as the scene drifts away from its last
  observation, while the learned models stay roughly flat. By day 100 persistence has
  roughly doubled its error and the learned models have not.

The NDVI trajectory shows why. Vegetation greens up over the forecast window, and the
learned models track that rise; persistence and climatology are flat lines by construction.
**Getting the average brightness of a landscape right is easy — getting the greening right
is the actual task**, and it is the only place these models earn their parameters.

#### And here is the same figure on a scene where they fail

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/figures/ndvi-1-dark.png">
  <img alt="NDVI figure for a dry scene. The observed vegetation stays near 0.30 while every learned model predicts 0.53 to 0.62; persistence sits almost exactly on the truth." src="docs/figures/ndvi-1-light.png">
</picture>

This scene is dry — bare and senescent ground, observed NDVI hovering around **0.30**. All
three models predict a green landscape at **0.53–0.62**, roughly double the truth, and stay
there for the full 100 days. Persistence, at 0.285, is nearly exact.

That is regression to the mean, seen directly: trained on a corpus where most scenes are
vegetated, the models have learned an average landscape and apply it to a place that is not
average. It is the same bias that makes the filmstrip look flat, and it is why the aggregate
5.8% margin over persistence is thinner than it sounds — the models win on average by
tracking the seasonal trend, and lose badly on scenes that depart from it.

Both scenes come from the same run and the same script; nothing was selected to make either
point. Scene 0 is the most cloud-free window in the validation split and scene 1 the second.

### 3. The cheapest mixer is the slowest model

The right panel is the project's original question, asked of trained models. At a matched 2M
budget S4D reaches the lowest error but costs **611 ms against the transformer's 384 ms** —
1.6× the latency for a 0.1% quality difference.

That is the same trade the cost benchmarks below predict from first principles: a diagonal
SSM's temporal mixing is cheap in parameters, so it spends the saving on width, and width is
expensive everywhere else. **On this task, at this scale, the SSM's parameter efficiency does
not convert into either a speed win or a quality win.**

---

### 4. In isolation, S4D's temporal mixing beats attention at every sequence length

The cost findings below are dataset-free and reproducible via
`scripts/benchmark_efficiency.py` and `scripts/benchmark_scaling.py`.

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/figures/mixing-dark.png">
  <img alt="Latency of the temporal backbone alone against sequence length. S4D is the cheapest at every T, with the gap over the transformer widening from 1.2x to 1.9x." src="docs/figures/mixing-light.png">
</picture>

Measuring the backbone **alone** — the whole point of holding the encoder and decoder fixed —
S4D is the cheapest mixer at every `T`, and the gap over attention widens from **1.2× at
T=8 to 1.9× at T=512**. Fitted latency exponents: S4D `k=0.94`, ConvLSTM `0.98`, Mamba
`1.02`, transformer `1.04`.

Two things worth noticing:

- **Nothing here is quadratic**, not even attention. At `hidden_dim=128` the feed-forward
  term dominates until roughly `T ≈ 4H`; the transformer's FLOPs double at 1.98× per
  doubling of `T` even at 512. The quadratic term shows up first in *wall-clock*
  (2.85× on the last doubling, against S4D's 1.97×) as the attention matrix goes
  memory-bound — before it ever shows up in the FLOP count.
- **Mamba is not in the winning group.** It is an SSM and it is 3× slower than S4D.
  Selectivity without a fused kernel costs more than the architecture class buys.

### 5. But at matched parameters the *whole model* flips the ranking

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/figures/efficiency-dark.png">
  <img alt="Compute and latency against parameter count for four backbones at matched budgets. The SSMs sit above the transformer on both axes." src="docs/figures/efficiency-light.png">
</picture>

A diagonal SSM's temporal mixing costs `~4HN` parameters against attention's `~4H²`, so at a
2M budget it affords `hidden_dim=272` where the transformer gets 128. But FLOPs scale with
*width squared* in the channel-mixing layers **every** architecture shares. The SSM spends
its parameter saving on width and pays for that width everywhere else: **2.8× the FLOPs and
1.4× the latency** of the transformer at the same parameter count.

So the cheaper mixer loses the full-model comparison. **Parameter efficiency and compute
efficiency are different questions, and here they have opposite answers** — which means
"how small can an SSM be?" does too.

### 6. The SSM's state dimension is cheap in parameters and expensive in compute

`state_dim` is the SSM's own capacity axis, with no ConvLSTM or transformer counterpart.
Growing S4D's state 8→256 (32×) adds only **+1.08M parameters**; reaching the same parameter
increase through width takes just 128→256. But per parameter added, state costs **~4× more
latency than width** (S4D), and **~40× more** for Mamba, whose scan is `O(N)` per step.

The same theme a third time: the two ledgers disagree. `state_dim` is the cheap axis only if
you are counting parameters.

> **Caveat that governs all of this:** TinyEarth's sequences top out at `T=8`, marked on the
> figure. Every architecture is firmly in its linear regime there, and the asymptotic
> argument for SSMs — the usual headline reason to reach for one — is simply absent. Earth
> observation minicubes are short by nature, which may make this an unfavourable setting for
> the architecture. Establishing that is a legitimate result.

---

## What's in here

| | |
| --- | --- |
| **4 temporal backbones** | `s4d`, `mamba`, `convlstm`, `transformer` — one interface, one config flag |
| **2 reference forecasts** | Persistence and climatology, parameter-free — the axis a learned MAE is read against |
| **Full data pipeline** | EarthNet2021 reader, partial download, temporal windowing, spatial cropping, cloud masking, deterministic splits, caching |
| **Training stack** | Trainer, optimiser/schedule construction, checkpointing, TensorBoard + JSONL + optional W&B |
| **10 metrics** | MAE, RMSE, PSNR, SSIM, SAM (all mask-aware) + parameters, FLOPs, peak memory, latency, throughput |
| **6 experiment sweeps** | EarthNet comparison, scaling, history length, forecast horizon, hidden dimension, state dimension |
| **Calibrated size tiers** | ~2M / ~5M / ~10M / ~20M, measured per architecture |
| **Qualitative figures** | Forecast filmstrips and NDVI trajectories, rendered on whole scenes from crop-trained checkpoints |
| **891 tests** | strict mypy over `src`, `tests` *and* `scripts`; ruff, black, CI on Linux + Windows |
| **Zero-download demo** | Synthetic data in the real on-disk format — everything runs without the dataset |

---

## Why it's a benchmark, not a model zoo

**The comparison is controlled, and the control is tested.** The interface is

```python
encoder:  [B, T, C, H, W] → [B, T, D, h, w]
backbone: [B, T, D, h, w] → [B, K, D, h, w]     # the component under study
decoder:  [B, K, D, h, w] → [B, K, C, H, W]
```

and encoder/decoder hold **no temporal parameters at all** — frames fold into the batch
dimension, so every unit of sequence-modelling capacity belongs to the backbone.

**Efficiency is a first-class result.** Every run reports parameters, FLOPs, peak memory,
latency and throughput by default, with proper CUDA synchronisation and warmup, reported as
a median. No flag to remember — the efficiency numbers are why the project exists.

**Correctness is verified against references, not convergence.** The S4D kernel is checked
against a literal step-by-step recurrence to **1e-7**:

```
x_k = Ā·x_{k-1} + B̄·u_k    ←→    FFT convolution with K_k = C·Ā^k·B̄
```

An SSM with a subtly wrong kernel still trains, still shows a falling loss, and still
produces publishable-looking numbers. Every metric gets the same treatment — tested against
a closed-form value, not a plausible-looking one.

---

## Status

The platform is complete and the quality experiment has been run on real data. **The numbers
are real; they are also budget-limited, and the limits matter more than the numbers.**

Reproduce the whole thing on a laptop, no GPU required:

```bash
pip install -e ".[data]"
python scripts/download_earthnet2021.py --splits train --max-tarballs 8 --stride 20
python scripts/run_earthnet_study.py
python scripts/evaluate_earthnet.py
python scripts/visualize_forecasts.py
```

**What these numbers do not support.** Read them as a controlled comparison of four
architectures under one small fixed budget — not as any architecture's best.

- **Nothing is trained to convergence.** Six epochs is a compute budget set by the slowest
  architecture and applied equally to all four so the comparison stays controlled.
- **1,650 cubes across 5 Sentinel-2 tiles**, and one tile is 73% of them. Geographic
  diversity is thin, and the validation cubes are often neighbours of training ones.
- **32×32 training crops** cover 640 m, not the full 2.56 km scene. Evaluation and figures
  use whole scenes, which the fully convolutional architecture allows for free.
- **The official test tracks are unused.** Their cubes ship as separate context and target
  files, which the reader does not yet join, so no EarthNetScore comparison is implied.
- Metrics come from the held-out validation partition. No hyperparameter search was run, so
  the only selection pressure is the choice of best epoch.

**Outstanding.** The frozen foundation-encoder comparison; GPU verification of mixed precision
and peak-memory measurement; joining the official test tracks; and **residual forecasting** —
predicting the change from the last observed frame rather than the absolute image, which
would start every model at persistence instead of asking it to rediscover the scene, and is
the single most promising fix for the blur.

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

**Adding a fifth:** subclass `TemporalBackbone`, register it, add a config, run
`scripts/calibrate_sizes.py`, add the name to one list in the tests. The parametrised
interface suite then applies automatically. Nothing else changes.

---

## Commands

```bash
tinyearth-train  +experiment=ssm_smoke      # train end to end
tinyearth-model  --compare                  # size and cost of every backbone, no training
tinyearth-data   +experiment=data_smoke     # build the data pipeline and report on it
tinyearth-info                              # environment and hardware report

python scripts/benchmark_efficiency.py            # cost at matched budgets
python scripts/benchmark_scaling.py --sweep mixing   # cost against sequence length
python scripts/benchmark_scaling.py --sweep state    # cost against SSM state size
python scripts/plot_results.py                       # render the figures above

pytest                                      # 891 tests, ~4 min
pytest -m "not slow"                        # 864 tests, ~30 s
```

The real-data study, end to end:

```bash
pip install -e ".[data]"
python scripts/download_earthnet2021.py --splits train --max-tarballs 8 --stride 20
python scripts/run_earthnet_study.py        # trains all four, concurrently
python scripts/evaluate_earthnet.py         # scores them and the references
python scripts/visualize_forecasts.py       # filmstrips and NDVI figures
python scripts/plot_results.py              # quality figure
```

Sweeps (add `data=earthnet2021` for real results):

```bash
tinyearth-train --multirun +experiment=scaling        model.backbone.size=tiny,small,base,large
tinyearth-train --multirun +experiment=history_length data.history_length=2,4,6,8
tinyearth-train --multirun +experiment=horizon        data.horizon=1,2,4,8
tinyearth-train --multirun +experiment=hidden_dim     model.backbone.kwargs.hidden_dim=64,128,256,512
tinyearth-train --multirun +experiment=state_dim      model.backbone.kwargs.state_dim=8,16,32,64,128
```

Each run writes `summary.json` (flat, scriptable), `metrics.jsonl`, `resolved_config.yaml`,
`run.log`, checkpoints and TensorBoard events to `outputs/<group>/<name>/`.

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
| [`docs/phase-1.md`](docs/phase-1.md) … [`phase-4.md`](docs/phase-4.md) | Per-phase deliverables and verification |
| [`CONTRIBUTING.md`](CONTRIBUTING.md) | Development workflow and coding standards |

---

## Engineering

```bash
black src tests scripts        # format (black is the only formatter)
ruff check src tests scripts   # lint, 17 rule families
mypy                           # strict, 80 source files clean
pytest                         # 809 tests
```

Two data-pipeline conventions are load-bearing and easy to get wrong, so both are tested
aggressively: **`cldmsk == 1` means cloudy** (inverting it trains the model exclusively on
cloud, and the loss still falls), and normalisation statistics must come from the training
split only.

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
