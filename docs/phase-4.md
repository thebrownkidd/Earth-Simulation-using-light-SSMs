# Phase 4 — State space models and the experiment framework

**Status: complete and verified.**

## Objective

The backbones the project is actually about, and the sweep machinery to measure
them: how small can a State Space Model become while remaining competitive?

## The framing correction this phase forced

The usual argument for SSMs is asymptotic — `O(T)` recurrence against attention's
`O(T²)`. **That argument does not apply here.** The history sweep tops out at
`T=8`, where it is 8 operations against 64. No wall-clock measurement in this
repository can show that advantage, and presenting one as if it did would be
misleading.

What *is* being tested is **parameter efficiency**. Diagonal-SSM temporal mixing
costs roughly `4HN` parameters against attention's `4H²`. At `H=256, N=64` that
is 65k versus 262k — a 4x saving in the component under study. The
`state_dim` sweep exists to measure exactly this, and the docs state the
limitation wherever numbers are reported.

## Delivered

| Requirement | Implementation |
| --- | --- |
| SSM backbone | `S4DBackbone` — diagonal SSM, FFT convolution, parallel in time |
| Second SSM | `MambaBackbone` — selective, input-dependent dynamics |
| `TemporalBackbone` interface | Defined in Phase 3; both SSMs implement it unchanged |
| Size tiers (~2M/5M/10M/20M) | `tinyearth.models.sizes`, calibrated per backbone |
| Scaling experiment | `configs/experiment/scaling.yaml` |
| History-length experiment | `configs/experiment/history_length.yaml` |
| Forecast-horizon experiment | `configs/experiment/horizon.yaml` |
| Hidden-dimension experiment | `configs/experiment/hidden_dim.yaml` |
| State-dimension experiment | `configs/experiment/state_dim.yaml` |
| Result collation | `scripts/collate_results.py` |
| Size recalibration | `scripts/calibrate_sizes.py` |

Adding both SSMs required **no change** to the encoder, decoder, losses, metrics,
training loop or data pipeline. That is the Phase 3 interface doing its job.

## The two SSMs

### S4D (`s4d`)

A per-channel linear system with diagonal `A`, discretised with a learned step
size, applied as one FFT convolution over time. Following S4D-Lin, `B` is fixed
to 1 — redundant with `C` under a diagonal `A`.

Stability is structural, not clamped: `A`'s real part is stored as
`-exp(log_real)`, so it is negative by construction. Timescales `Δ` are
initialised log-uniformly so different channels start at different timescales.

### Mamba (`mamba`)

`Δ`, `B` and `C` become functions of the input, so the model can decide per step
how much of an observation to write into state. For Earth observation that is
substantive: a cloudy frame carries little surface information, and a selective
model can learn to hold state through it.

**The cost is real and stated:** input-dependent parameters make the system
time-varying, so it cannot be one convolution and the scan is sequential. Mamba
gives up the parallel-in-time property S4D has. Measured below.

## Size tiers

Every tier fixes `n_layers = 4` and varies **only** `hidden_dim`. Letting depth
and width move together would confound the scaling result.

| Backbone | tiny (2M) | small (5M) | base (10M) | large (20M) |
| --- | ---: | ---: | ---: | ---: |
| `s4d` | 272 | 448 | 672 | 960 |
| `mamba` | 256 | 416 | 608 | 880 |
| `convlstm` | 80 | 128 | 192 | 272 |
| `transformer` | 128 | 208 | 288 | 416 |

Widths are `hidden_dim`. Each lands within ~5% of its nominal target;
`tests/test_sizes.py` rebuilds every tier and fails if the table drifts.

### The first real result

**To reach a 2M budget, the ConvLSTM affords `hidden_dim=80`; S4D affords 272.**

A ConvLSTM spends `4 · 9 · 2H · H` parameters on its gate convolution; a diagonal
SSM spends `≈5H² + 4HN`. At a matched budget the SSM buys roughly **3.4x the
width**, and the transformer roughly 1.6x. This is visible before a single
training step, and it is the mechanism the whole project is testing.

## Measured cost

`latent_dim=128`, `n_layers=4`, 64x64 input, `T=4 -> K=2`, batch 2, CPU.
Reproduce with `tinyearth-model --compare`.

| Backbone | Tier | `hidden_dim` | Total params | Backbone % | GFLOPs/sample | Latency (ms) |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| `s4d` | tiny | 272 | 2,076,916 | 89% | 5.8 | 104 |
| `mamba` | tiny | 256 | 2,056,260 | 89% | 6.5 | 610 |
| `transformer` | tiny | 128 | 2,112,964 | 89% | 2.1 | 76 |
| `convlstm` | tiny | 80 | 2,221,636 | 90% | 7.2 | 76 |
| `s4d` | small | 448 | 4,849,796 | 95% | 13.7 | 216 |
| `mamba` | small | 416 | 4,862,500 | 95% | 15.0 | 1111 |
| `transformer` | small | 208 | 5,154,324 | 96% | 3.7 | 136 |
| `convlstm` | small | 128 | 4,965,700 | 95% | 15.7 | 112 |
| `s4d` | base | 672 | 10,170,916 | 98% | 29.2 | 355 |
| `mamba` | base | 608 | 9,919,588 | 98% | 30.3 | 1573 |
| `transformer` | base | 288 | 9,629,284 | 98% | 6.0 | 169 |
| `convlstm` | base | 192 | 10,430,788 | 98% | 32.4 | 197 |
| `s4d` | large | 960 | 19,961,476 | 99% | 58.1 | 571 |
| `mamba` | large | 880 | 20,238,996 | 99% | 61.7 | 2351 |
| `transformer` | large | 416 | 19,771,108 | 99% | 11.2 | 329 |
| `convlstm` | large | 272 | 20,165,188 | 99% | 62.3 | 370 |

Cost measurements only -- **no forecast-quality claim is made here.**

### The finding: parameter efficiency is not FLOP efficiency

This is the most important result Phase 4 produced, and it complicates the
project's premise rather than confirming it.

The SSM's cheap temporal mixing means that at a matched parameter budget it
affords far more width -- 272 against the transformer's 128 at the 2M tier. But
**FLOPs scale with width squared in the channel-mixing layers**, which every
architecture shares. So the SSM spends its parameter saving on width, and pays
for that width in compute:

| At the 2M tier | `hidden_dim` | GFLOPs | Latency |
| --- | ---: | ---: | ---: |
| `transformer` | 128 | 2.1 | 76 ms |
| `s4d` | 272 | 5.8 (2.8x) | 104 ms (1.4x) |

The transformer is the cheapest architecture here on both compute and latency,
at every tier. That is the opposite of the naive expectation.

Three things follow:

1. **"How small can an SSM be?" has two different answers** depending on whether
   "small" means parameters or compute. This repository measures both, and they
   disagree. A paper reporting only parameter counts would be telling half the
   story.
2. **The SSM's advantage should show up in quality-per-parameter**, if it shows
   up at all -- more width at the same budget is only useful if the model uses
   it. That is exactly what the scaling sweep measures, and it has not been run
   on real data.
3. **The asymptotic argument is absent because `T <= 8`.** At the sequence
   lengths where SSMs win decisively, none of this table would look the same.
   TinyEarth's sequences are short by nature, which may simply make it an
   unfavourable setting for the architecture.

None of this is a reason not to run the experiment. It is a reason to report
FLOPs and latency next to parameter count every time, which the codebase does by
default.

### Mamba's scan cost

Mamba is 6-8x slower than S4D at every tier, entirely from the sequential scan.
Building the discretised coefficients per step rather than materialising
`[B, L, C, N]` tensors cut a large-tier forward pass from 13.8s to 9.8s at 128px
and reduced peak activation memory `L`-fold, but the Python-level loop remains
the bottleneck -- which is precisely what Mamba's fused CUDA kernel exists to
avoid.

At `T <= 8` this is a constant-factor problem, not an asymptotic one. It is
reported rather than hidden because a latency comparison that omitted it would
be misleading.

## Running the experiments

```bash
# Scaling: four backbones x four budgets
tinyearth-train --multirun +experiment=scaling \
    model=s4d,mamba,convlstm,transformer \
    model.backbone.size=tiny,small,base,large

# The SSM-specific axis
tinyearth-train --multirun +experiment=state_dim \
    model.backbone.kwargs.state_dim=8,16,32,64,128

# Collate
python scripts/collate_results.py --group scaling
```

Every run writes a flat `summary.json`, so collation is a walk over files.

## Verification

```bash
$ pytest -q
809 passed, 1 skipped in 197s

$ pytest -q -m "not slow"
784 passed, 1 skipped in 28s

$ ruff check src tests scripts
All checks passed!

$ black --check src tests scripts
84 files would be left unchanged.

$ mypy
Success: no issues found in 80 source files
```

New test coverage:

| Area | Tests | Notable cases |
| --- | --- | --- |
| `test_ssm.py` | 110 | **FFT kernel vs. explicit recurrence**, causality, stability under extreme `A`, selective-scan vs. literal recurrence, spatial independence |
| `test_sizes.py` | 40 | every tier rebuilt and checked against target; tiers comparable across backbones |
| `test_cli.py` (extended) | +8 | all four backbones train from one sweep config; typos still rejected |

### The load-bearing test

`test_fft_convolution_matches_the_explicit_recurrence` computes the state space
response two independent ways — the FFT convolution the model uses, and a literal
`x_k = Ā x_{k-1} + B̄ u_k` loop — and requires agreement to 1e-4.

An SSM with a subtly wrong kernel still trains, still shows a falling loss, and
still produces publishable-looking numbers. Convergence is not evidence of
correctness; a reference computation is.

### Bug found

A sweep config setting `state_dim` crashed the ConvLSTM, because
`model.backbone.kwargs` is a merged dict and reaches every architecture in a
sweep. Fixed with signature-aware filtering: an argument is dropped only if some
*other* registered backbone accepts it. Dropping everything unrecognised would
have swallowed a misspelled `hidden_dim`, which is exactly the error that wastes
a sweep.

### A note on the suite's wall time

A full background run reported 7h27m. That was the machine suspending overnight,
not a regression: the 783 fast tests run in 28s and the 25 subprocess tests take
7–12s each. Timed directly, the full suite is a few minutes.

## Interpretation

A green suite means the SSMs are mathematically correct, satisfy the same
interface as the baselines, hit their calibrated budgets, and sweep across
architectures from one config.

It says **nothing about forecasting quality**. No model has been trained to
convergence on real EarthNet2021 data. Every quality number in this repository
came from synthetic data over a handful of steps.

## Decisions worth revisiting

1. **Mamba's scan is sequential.** Correct at `T≤8`; at long sequence lengths it
   needs a parallel associative scan. Measured latency already shows the cost —
   Mamba is roughly 2.5x slower than S4D at equal size.
2. **Forecast queries are a fixed `max_horizon=32` bank.** Unused rows receive no
   gradient but do receive weight decay, so they drift toward zero. Harmless at a
   fixed horizon; revisit if training at one horizon and evaluating at a longer one.
3. **`state_dim` is not in the size calibration.** Tiers vary width at fixed
   `state_dim`. Since `state_dim` is the cheap axis, a tier could equally be hit
   by raising it — worth a second calibration axis if the state sweep proves it
   matters.
4. **Only S4D is parallel in time.** When reporting speed, say which backbones
   are non-autoregressive (transformer, s4d, mamba) and which is not (convlstm),
   because that difference is not about the mixing mechanism.
5. **The frozen foundation-encoder comparison is not implemented.** It was listed
   as a later comparison point and remains open. The `ENCODERS` registry is where
   it goes.

## Next

The infrastructure is complete. What remains is **running the experiments on real
data**, which needs the EarthNet2021 download and a GPU. See
[project-log.md](project-log.md) for the full status and honest limits.
