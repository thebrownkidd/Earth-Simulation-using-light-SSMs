# Evaluation

Both metric families are reported **automatically at the end of every run**. Efficiency is
this project's primary result, so it should never depend on remembering a flag.

They land in three places:

```
outputs/<group>/<name>/
├── summary.json      # headline numbers, one flat mapping
├── metrics.jsonl     # per-step and per-epoch history
├── metrics.json      # final values
└── tensorboard/
```

`summary.json` is deliberately flat and scriptable, so collating a sweep is a script rather
than a UI task.

---

## Forecast quality

All five are **mask-aware**. On EarthNet2021 a large share of pixels are cloud-contaminated,
and scoring them measures cloud prediction rather than vegetation forecasting. An unmasked
PSNR on this data can be several dB off, in whichever direction the cloud falls.

| Metric | Direction | Measures |
| --- | --- | --- |
| `mae` | lower | Mean absolute reflectance error |
| `rmse` | lower | Penalises large errors more than MAE |
| `psnr` | higher | Signal-to-noise, in dB |
| `ssim` | higher | Structural similarity, Wang et al. (2004) |
| `sam` | lower | Spectral angle, in degrees |

### Why these are implemented here rather than taken from `torchmetrics`

Masked variants. Applying a library SSIM to a zero-filled image scores the fill value as if
it were data, and the result is not comparable across samples with different cloud cover.

### Two implementation details that matter

**SSIM masks the map, not the inputs.** Masking the inputs first would let the fill value
bleed into neighbouring windows through the Gaussian, corrupting scores for pixels that are
themselves perfectly valid.

**SAM uses the stable half-angle form**, `2·atan2(|â−b̂|, |â+b̂|)`, not `acos(â·b̂)`. The two
are equivalent in exact arithmetic, but `acos` has unbounded derivative at 1, so in float32
an *identical* pair scores a spurious ~0.006° instead of 0. Small — but it is noise in a
reported metric, and it sits exactly where a good model lives.

SAM is invariant to brightness scaling by construction, which is what makes it useful here:
it separates a genuinely correct land-cover prediction from one that merely has the right
average illumination.

### Accumulation

`MetricAccumulator` weights by sample count and divides once at the end, so the reported
value is the mean over *pixels*, not the mean of per-batch means. Those differ whenever
batches have unequal valid-pixel counts — which, with variable cloud cover, is always.
Infinite PSNR (a perfect match) is excluded rather than propagated.

---

## Efficiency

| Metric | Notes |
| --- | --- |
| `efficiency/parameters` | Trainable count, plus a per-component breakdown |
| `efficiency/gflops_per_sample` | Forward FLOPs via `torch.utils.flop_counter` |
| `efficiency/peak_memory_mb` | Peak allocated during a forward pass; CUDA only |
| `efficiency/latency_ms` | **Median** single-batch forward latency |
| `efficiency/throughput_samples_per_s` | Samples per second at the measured batch size |
| `params/backbone_fraction` | Share of parameters in the component under study |

### Measurement discipline

Naive GPU timing is wrong in two ways, and both *overstate* speed:

- **CUDA is asynchronous.** A kernel launch returns before the work finishes, so timing
  without `torch.cuda.synchronize` measures launch overhead.
- **Early iterations are unrepresentative.** cuDNN autotuning, memory-pool growth and lazy
  initialisation all land on the first passes.

`measure_latency` and `measure_throughput` handle both, with warmup and explicit
synchronisation. Latency is a **median**, which is far less sensitive to scheduler noise
than a mean. Throughput is timed as one block so per-call overhead does not inflate it.

### Caveats

- FLOPs count multiply-accumulate as 2 and do not see elementwise work. Treat the figure as
  a comparable index across architectures, not an absolute cost.
- Peak memory is `None` on CPU; PyTorch does not track allocation there.
- **Report the determinism setting.** Deterministic kernels are not the ones a user would
  deploy, so benchmarking them understates real throughput. Efficiency runs should use
  `seed.deterministic=false`. See [reproducibility.md](reproducibility.md).
- Profiling never takes down a completed run: a failure is logged and reported as `n/a`.

---

## Reading a result

```bash
tinyearth-train +experiment=baseline_smoke
```

```
--- Efficiency -----------------------------------------
  parameters                      128,484
  GFLOPs / sample                   0.130
  peak memory (MiB)                   n/a
  latency (ms)                       6.99
  throughput (samp/s)               586.7
--- Results --------------------------------------------
  val/mae                0.268813
  val/psnr              10.931582
  val/ssim               0.340885
  val/sam               12.412944
```

An efficiency number is only interpretable against the hardware that produced it. Attach
the output of `tinyearth-info` to any reported result.

---

## Comparing models

```bash
tinyearth-model --compare                 # every backbone, side by side
tinyearth-model model.backbone.kwargs.hidden_dim=256
```

This is how the Phase 4 size tiers (tiny ~2M, small ~5M, base ~10M, large ~20M) are reached:
sweep `hidden_dim` against measured counts rather than guessing.

Measured baseline numbers are tabulated in [models.md](models.md).

---

## Adding a metric

1. Add the function to `src/tinyearth/evaluation/metrics.py`, taking
   `(prediction, target, mask)` and returning a float.
2. Add it to `forecast_metrics`.
3. Test it against a **closed-form value**, not a plausible-looking one. A metric that is
   merely plausible is worse than none — it produces numbers that look publishable and are
   wrong.
4. Add a masking test: corrupting masked pixels must not change the result.
