# Phase 3 — Baselines, training and metrics

**Status: complete and verified.** Superseded by [phase-4.md](phase-4.md).

## Objective

Reproducible baselines, a training loop, and the metric infrastructure that every later
result depends on. The goal is **correctness**, not benchmark numbers.

## Scope adjustment

The plan put the `TemporalBackbone` interface in Phase 4. But ConvLSTM and the temporal
transformer *are* temporal backbones — if they do not share the interface the SSM will use,
Phase 4 cannot swap them and "only the temporal backbone changes" stops being true.

The interface is therefore defined **now**, with both baselines implementing it. Phase 4
adds one class and changes nothing else. This is the design working as intended rather than
a scope expansion.

## Delivered

| Requirement | Implementation |
| --- | --- |
| ConvLSTM baseline | `tinyearth.models.temporal.ConvLSTMBackbone` |
| Temporal transformer baseline | `tinyearth.models.temporal.TemporalTransformerBackbone` |
| Parameter counts | `ParameterBreakdown`, `tinyearth-model`, tabulated in [models.md](models.md) |
| `TemporalBackbone` interface | `tinyearth.models.base` |
| Encoder / decoder | `CNNEncoder`, `CNNDecoder` — held fixed across backbones |
| L1 loss, extensible loss system | `tinyearth.models.losses` + `LOSSES` registry |
| Forecast metrics (MAE, RMSE, SSIM, PSNR, SAM) | `tinyearth.evaluation.metrics`, all mask-aware |
| Efficiency metrics | `tinyearth.evaluation.efficiency` |
| Metrics appear after every experiment | On by default; `summary.json` per run |
| Training loop, no monolithic script | `tinyearth.training.Trainer`; CLI orchestrates only |
| Experiment tracking | TensorBoard + JSONL, optional W&B |

## Configuration

```bash
tinyearth-train +experiment=baseline_smoke
tinyearth-train +experiment=baseline_smoke model=transformer   # the only change needed
```

See [`configs/experiment/baseline_smoke.yaml`](../configs/experiment/baseline_smoke.yaml).

## Expected output

```
--- Model ----------------------------------------------
  component          params    share
  encoder            25,920   20.2%
  backbone           74,912   58.3%
  decoder            27,652   21.5%
  total             128,484  100.0%
--- Training -------------------------------------------
  epoch 1/2 | train loss 0.29553 | val loss 0.27159 | 0.5s | lr 5.00e-04
  epoch 2/2 | train loss 0.27526 | val loss 0.26881 | 0.3s | lr 1.00e-06
--- Efficiency -----------------------------------------
  parameters                      128,484
  GFLOPs / sample                   0.130
  latency (ms)                       6.99
  throughput (samp/s)               586.7
--- Results --------------------------------------------
  val/mae 0.268813   val/rmse 0.290836   val/psnr 10.931582
  val/ssim 0.340885  val/sam 12.412944
```

Plus `summary.json`, `metrics.jsonl`, `last.ckpt` and `best.ckpt` in the run directory.

## Verification

```bash
$ pytest -q
687 passed, 1 skipped

$ ruff check src tests scripts
All checks passed!

$ black --check src tests scripts
77 files would be left unchanged.

$ mypy
Success: no issues found in 75 source files
```

New test coverage:

| Area | Tests | Notable cases |
| --- | --- | --- |
| `test_models.py` | 87 | backbone interface contract (parametrised over both), **encoder/decoder sizes identical across backbones**, transformer spatial independence, gradient reachability |
| `test_losses.py` | 47 | masked pixels contribute nothing, division by valid count, Charbonnier smoothness |
| `test_metrics.py` | 47 | every metric against a **closed-form value**, SAM brightness invariance, accumulator weighting |
| `test_training.py` | 39 | weight updates, early stopping, checkpoint contents, no-decay grouping, tracker robustness |
| `test_cli.py` (extended) | 12 | end-to-end training; **backbone swap changes only the backbone** |

### The bug this phase surfaced

**All library logging was silently suppressed in every CLI run — including Phase 2's.**

Hydra's `job_logging: disabled` profile sets `disable_existing_loggers: true`, and
`logging.config.dictConfig` applies that to every logger that already exists. Module-level
`get_logger(__name__)` calls run at *import* time, so all of them were switched off before
the first log line. No error, no warning — the trainer's per-epoch progress and the entire
efficiency table simply never appeared, in console or file.

Fixed at the root (`hydra/job_logging: none`, which sets the flag to `false`) and
defensively in `setup_logging`, which now re-enables the `tinyearth` hierarchy regardless of
what else has touched the logging system. `tests/test_logging.py::TestDisabledLoggerRecovery`
guards it.

Two smaller ones: SAM scored a spurious ~0.006° for identical spectra due to `acos`
instability in float32, now using the stable half-angle form; and `Trainer` assumed its run
directory already existed, which broke standalone use outside the CLI.

## Interpretation

A green suite means the stack is correct: models assemble, gradients reach every component,
masks reach the loss and the metrics, the loop trains, checkpoints round-trip, and metrics
are computed against closed-form values.

It says **nothing about forecasting quality**. No baseline has been trained to convergence
on real data. The measured numbers in [models.md](models.md) are *cost* measurements.

## Decisions worth revisiting later

1. **The transformer is non-autoregressive; the ConvLSTM is not.** A genuine architectural
   difference, and favourable to the transformer on latency. When reporting the comparison,
   say so — do not attribute the gap to attention alone.
2. **No size tiers yet.** The tiers (tiny ~2M / small ~5M / base ~10M / large ~20M) belong
   to Phase 4, where scaling is the object of study. `tinyearth-model` is the tool for
   hitting a budget; measured counts are in [models.md](models.md).
3. **`plateau` scheduling is accepted but not wired.** `build_scheduler` returns `None` for
   it and the trainer does not yet step it on a validation metric. Either implement it or
   remove the option in Phase 4.
4. **AMP is untested on GPU.** The code path exists and is CUDA-gated, but this machine has
   no GPU. Verify before relying on it for a memory measurement.
5. **Efficiency is measured on the forward pass only.** Training-step cost (backward +
   optimiser) is not profiled separately. Add it if training speed becomes a headline claim.
6. **No multi-scale or perceptual losses yet.** SSIM and SAM exist as *metrics*; wrapping
   them as losses is a small module each, by design.

## Next: Phase 4

Tiny State Space Model backbones. Register an SSM implementing `TemporalBackbone`, add its
config, add its name to `BACKBONE_NAMES` in `tests/test_models.py`, and the scaling,
history-length, horizon and hidden-dimension sweeps all become config edits.

Phase 4 must not begin until this page's verification block reproduces.
