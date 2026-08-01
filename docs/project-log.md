# Project log

A chronological record of what was built, what was decided, and what went wrong.

Kept because the *reasons* behind a research codebase decay faster than the code.
Six months on, "why is the mask polarity checked so aggressively?" is a question
whose answer is expensive to reconstruct and cheap to have written down.

Each phase links to its own verification page. Decisions that were reversed or
qualified later are marked, rather than quietly edited.

---

## Phase 1 — Repository scaffold

**Delivered:** `src/` layout, Hydra structured configs, determinism and logging
utilities, device provenance, a typed component registry, four quality gates.

**Decisions**

| Decision | Reasoning |
| --- | --- |
| `src/` layout, not a flat package | The only importable copy is the installed one, so packaging bugs surface in tests rather than after release. |
| `configs/` outside the package | Configs are the experiment record and are edited constantly; burying them in a wheel is wrong. |
| Registry over Hydra `_target_` | A registry can be enumerated, and does not put import paths into published configs. |
| Console logging separate from metric tracking | A tracking backend can be swapped without touching diagnostics. |
| Typed schema only for what Phase 1 owned | An honest hole beats fabricated fields that get rewritten. |

**Went wrong**

- `@hydra.main(config_path="../../configs")` breaks under `pip install -e .` —
  Hydra resolves relative config paths as importable module paths. Fixed by
  routing every entry point through `cli/_hydra.py` with an absolute path.
- `--dry-run` still wrote Hydra job metadata; the flag did not do what it said.
- The fix for that then broke on Windows short paths — `~` in `ARPITG~1` is a
  metacharacter in Hydra's override grammar. Both found by subprocess tests, not
  visible in-process.

---

## Phase 2 — Dataset pipeline

**Delivered:** EarthNet2021 reader, temporal windowing, cloud masking,
normalisation, deterministic splits, per-worker cube cache, a synthetic dataset
in the real on-disk format, a visualisation notebook.

**Decisions**

| Decision | Reasoning |
| --- | --- |
| `target` is `[K,C,H,W]`, not `[C,H,W]` | **Deviation from the written spec.** The horizon sweep needs 1/2/4/8; a rank that depends on a config value forces every loss and metric to branch. |
| Hash-based train/val split | Stable when cubes are added or removed. A seeded shuffle reassigns everything when the file count changes, contaminating comparisons against earlier runs. |
| BLAKE2b, not `hash()` | `hash()`'s string seed varies per process. |
| Synthetic data writes real `.npz` files | Tests exercise the production reader, not a parallel implementation that could drift. |
| NaN pixels marked invalid | Deviates from the official toolkit, which only zero-fills. Zero-filled NaN is indistinguishable from genuine dark reflectance. |
| Per-worker LRU cube cache | 26 overlapping windows per cube become one decompression instead of 26. |

**The convention that matters most**

`cldmsk == 1` means **cloudy**; validity is `1 - cldmsk`. Inverting it trains the
model exclusively on cloud and *losses still fall* —
`tests/test_minicube.py::TestMaskPolarity` exists for precisely this.

**Went wrong**

- Hash partitioning assigned zero cubes to validation at small cube counts, and
  the error blamed an incomplete download. Now distinguished with an actionable
  message.
- Lowering synthetic `n_cubes` left stale files that the dataset kept reading.

---

## Phase 3 — Baselines, training, metrics

**Delivered:** `TemporalBackbone` interface, ConvLSTM and temporal-transformer
baselines, fixed CNN encoder/decoder, registry-backed losses, five mask-aware
forecast metrics, five efficiency metrics, the training loop, metric tracking.

**Decisions**

| Decision | Reasoning |
| --- | --- |
| `TemporalBackbone` defined here, not Phase 4 | **Scope adjustment.** The baselines *are* temporal backbones; without a shared interface, Phase 4 could not swap them and the project's central claim would be false. |
| Backbone emits `K` steps, not a summary state | Anything the decoder did to expand one state into `K` frames would be shared machinery whose capacity differs by architecture. |
| Encoder/decoder hold no temporal parameters | All sequence-modelling capacity belongs to the component under study. |
| Transformer attends over time only | Spatial attention would add capacity the others lack, and costs `O((T·h·w)²)`. |
| Metrics written, not taken from `torchmetrics` | Masked variants. A library SSIM on a zero-filled image scores the fill value as data. |
| SAM via the stable half-angle form | `acos` has unbounded derivative at 1; in float32 identical spectra scored a spurious 0.006°. |
| Efficiency profiled by default | It is the primary result; it must not depend on remembering a flag. |

**Went wrong**

- **All library logging was silently suppressed in every CLI run — including
  Phase 2's.** Hydra's `job_logging: disabled` sets
  `disable_existing_loggers: true`, and `dictConfig` applies that to every
  logger created at import time, which is every module-level
  `get_logger(__name__)`. No error, no warning; the trainer's epoch progress and
  the entire efficiency table simply never appeared. Fixed at the root
  (`job_logging: none`) and defensively in `setup_logging`.
- `Trainer` assumed its run directory already existed, breaking standalone use.

---

## Phase 4 — State space models and the experiment framework

**Delivered:** S4D (diagonal) and Mamba (selective) backbones, calibrated size
tiers across all four architectures, five sweep configs, results collation, a
recalibration script.

**Decisions**

| Decision | Reasoning |
| --- | --- |
| S4D as the primary SSM | Parallel in time via an FFT convolution, and simple enough to verify against a literal recurrence. |
| Mamba's scan is sequential | Input-dependent parameters make the system time-varying, so no single kernel exists. At `T≤8` a parallel scan would be machinery with nothing to do. **Stated as a limitation, not hidden.** |
| SSMs are non-autoregressive | Matches the transformer, so the two differ only in how they mix over time. ConvLSTM is the odd one out. |
| Size tiers calibrated in code, not YAML | Different architectures reach 2M at very different widths; the comparison is only meaningful at matched budgets. |
| Tiers fix depth, vary width only | Letting both move would confound the scaling result. |
| Inapplicable backbone kwargs are dropped, typos still raise | A cross-architecture sweep passes `state_dim` to the ConvLSTM. Dropping everything unrecognised would swallow a misspelled `hidden_dim`. |

**The framing correction**

The SSM's headline advantage — `O(T)` versus attention's `O(T²)` — **does not
apply at this project's sequence lengths.** The history sweep tops out at `T=8`,
where that is 8 operations against 64. What is actually being tested is
**parameter efficiency**: diagonal-SSM temporal mixing costs `~4HN` against
attention's `~4H²`. The docs say so wherever numbers are reported, and the
`state_dim` sweep exists to measure it directly.

**The finding that complicated the premise**

Measuring the tiers produced a result that argues *against* the naive
expectation, and it is reported prominently rather than buried.

At a matched 2M budget the SSM affords `hidden_dim=272` against the
transformer's 128 — the parameter saving is real. But FLOPs scale with width
squared in the channel-mixing layers every architecture shares, so the SSM
spends that saving on width and pays for the width in compute: 5.8 GFLOPs
against the transformer's 2.1, and 104 ms against 76 ms. **The transformer is
the cheapest architecture here on both compute and latency, at every tier.**

So "how small can an SSM be?" has two answers depending on whether *small* means
parameters or compute, and here they disagree. The SSM's case now rests entirely
on quality-per-parameter — more width at the same budget helps only if the model
uses it — which is exactly what the scaling sweep measures and what has not been
run.

This is the kind of result that a project wanting a tidy story would omit. It is
in the README.

**Went wrong**

- A sweep config setting `state_dim` crashed the ConvLSTM, since `backbone.kwargs`
  is a merged dict. Fixed by signature-aware filtering that still rejects typos.
- Mamba was pathologically slow — 13.8 s for one large-tier forward pass —
  because the scan materialised three `[B, L, C, N]` tensors up front. Building
  the coefficients per step cut that to 9.8 s and reduced peak activation memory
  `L`-fold. Still slow: the remaining cost is the Python-level loop, which is
  what Mamba's fused CUDA kernel exists to avoid.
- A background test run reported 7 h 27 m, which looked like a catastrophic
  regression. It was the machine suspending overnight — the 783 fast tests run in
  28 s. Worth recording because the instinct to optimise on that number would
  have wasted a day.

---

## Phase 5 — Real data

**Delivered:** partial download support, spatial cropping, parameter-free
reference forecasts, full-scene evaluation, the qualitative figures, and the
first forecast-quality numbers this repository has produced on real data.

**The constraint that shaped everything.** The reference machine is a Ryzen 7
5700U laptop: 8 cores, 15 GB RAM, **no CUDA GPU**, PyTorch CPU build. Measured
before planning anything, a training step at the official 128 px protocol runs
at 0.27–0.50 samples/s. One epoch over a thousand cubes would take an hour, per
model. Every decision below follows from that measurement rather than from
taste.

**Decisions**

| Decision | Reasoning |
| --- | --- |
| Download a *prefix* of the training split | The dataset ships as 160 independently-addressable ~1 GB tarballs. The toolkit's downloader is all-or-nothing, so 155 GB was the smallest thing it could do. `--max-tarballs` takes 8. The binding constraint was never data volume — it was CPU throughput. |
| Wrote our own downloader | Also forced: the toolkit verifies TLS against the platform trust store, which on this machine cannot build a chain to the host and fails outright. `certifi` works. |
| Reduce **space**, never time | A step at 128 px costs ~18× one at 32 px. That cost falls on the encoder and decoder — the components held *fixed* across architectures. Cropping buys compute without touching the independent variable. Shortening the sequence would have attacked the very thing being measured. |
| Crop rather than downsample | Preserves native 20 m texture, and a random crop differs every epoch, so it augments. Each 128×128 scene yields ~9,400 distinct 32×32 crops, which is why 1,200 cubes is not as few as it sounds. |
| Keep the official 10→20 protocol exactly | The temporal axis is the object of study. Halving the horizon would have made the run ~2× faster and the result far less comparable. |
| Train on crops, evaluate on whole scenes | No parameter depends on height or width, so this is free. It also makes the reported numbers answer a *harder* question than training optimised. |
| Run the four models concurrently | Measured: four threads reach ~95% of sixteen-thread throughput (s4d 3.96 vs 4.09 samples/s). The other twelve threads were contention, not work. Four processes at four threads each is close to 4× the useful work. |
| Added persistence and climatology | A learned MAE is uninterpretable alone. Earth surface imagery is mostly static over 100 days, so a model that learned only to echo its input would still post a good number. |

**Bugs and surprises**

- **Mamba is ~6× slower than S4D in training**, and its throughput is *flat* in
  batch size — 0.64, 0.65, 0.71 samples/s at batch 2, 4, 8. There is no batching
  trick that recovers it; the sequential Python scan dominates. It alone set the
  epoch budget for every other model, since a controlled comparison requires
  equal epochs.
- Batch 16 and 32 made Mamba *worse than linearly* (3.1× and 3.9× per doubling).
  Resident memory stayed at 0.5 GB, so this was cache behaviour, not swapping —
  worth recording because the obvious diagnosis was wrong.
- **The first forecast figure was washed out to grey.** The cause was not the
  model: `stretch_limits` computed percentiles over cloud-masked pixels, which
  the mask policy had blanked to zero. On a quarter-cloud scene those zeros own
  the lower percentile and pin the black point at 0.0. Excluding them recovered
  the true dynamic range. The prediction's own statistics were fine all along
  (mean 0.105 against truth's 0.082) — a case where the plot lied and the
  numbers did not.
- **Error panels lit up bright over cloud**, because blanked truth is zero and
  the model does not predict zero. Masked now, and drawn flat grey.
- **The observed NDVI curve plunged to zero several times a season**, inventing a
  dramatic vegetation collapse out of weather. Fully clouded frames have both
  bands zeroed, so NDVI evaluates to exactly 0 — a *plausible* value for bare
  ground, which is what made it dangerous. Such frames are now gaps.
- **The matched-parameter comparison was not matched.** The first launch trained
  at 0.73M, 0.73M, 1.19M and 2.61M parameters — a 3.6x spread — while every
  config said `size: tiny` and every log line said the tier had been resolved.
  Two independent causes, both silent:

  1. Every model config ships a concrete `backbone.kwargs.hidden_dim`, and an
     explicit width beats `size`. The tier was computed, logged, and discarded.
     There was no way to *stand down* a default width, so `size` could never
     take effect from a composed config at all.
  2. The tiers are calibrated at four layers, but the ConvLSTM and transformer
     configs default to two — so even the right width would have hit the wrong
     budget.

  **The test that should have caught this asserted on a log substring.** It ran
  `model.backbone.size=tiny` and checked for `"hidden_dim=272"` in the output —
  and the message warning that the tier had been *ignored* also contains
  `hidden_dim=272`. It passed for the entire life of the size-tier feature while
  the feature did nothing. Fixed by making `hidden_dim: null` mean "let the tier
  decide", raising when a backbone ends up with no width at all, promoting the
  override notice to a warning that says the tier is not in effect, and
  rewriting the test to assert on the *resolution* message and on the absence of
  the override message. Caught only because 2M-parameter models reported
  0.7M — the kind of thing a glance at a log catches and a green test suite does
  not.
- **The first eight tarballs were all one Sentinel-2 tile.** The archives are
  grouped by tile, so a prefix of the training split is one region of the planet
  — 1200 cubes of southwest Iberia — not a sample of EarthNet2021. Nothing
  failed; `ls data/earthnet2021/train/` listing a single directory was the only
  symptom. A model trained *and validated* on that has seen one landscape, and
  the validation cubes are neighbours of the training ones. Fixed by adding
  `--stride` to the downloader and fetching additional tarballs from across the
  manifest. The documentation had explicitly claimed the opposite — that the
  ordering did not correlate with content — which is a reminder that a plausible
  assumption written confidently into docs is worse than no note at all.
- Two config conflicts (`persistent_workers` and `prefetch_factor` against
  `num_workers: 0`) were caught by the schema in seconds, before any run.

**Honest limits of the quality numbers**

- 32×32 training crops cover 640 m, not the full 2.56 km scene.
- **No model is trained to convergence.** Eight epochs is a compute budget set by
  the slowest architecture, applied equally to all four so the comparison stays
  controlled.
- Metrics are reported on the held-out validation partition. No hyperparameter
  search was run, so the only selection pressure is the choice of best epoch.
- The official test tracks are unused. Their cubes are split into separate
  context and target files, which the reader does not yet join.

---

## Cross-cutting lessons

**Subprocess tests earn their cost.** Three bugs — the `--dry-run` leak, the
Windows short-path lexing failure, and the suppressed logging — were invisible
to in-process tests. They only appear when you run what a user runs.

**Silent failures are the expensive ones.** Every bug in this log that took real
effort shared one property: nothing raised. Inverted mask polarity, disabled
loggers, stale cubes, an unbounded decoder against standardised targets. The
tests that matter most are the ones asserting a *value*, not an absence of
exceptions.

**Verify mathematics against a reference, not against convergence.** An SSM with
a subtly wrong kernel still trains and still produces plausible losses. The FFT
convolution is checked against a literal step-by-step recurrence.

**Assert on the outcome, not on the log line.** The size-tier test checked that
`hidden_dim=272` appeared in the output. It did appear — inside the message
saying the tier had been overridden and ignored. A test that reads a log is
testing a string; a test that reads the built model is testing the system. Where
a message and its negation share a substring, the assertion is worse than
absent, because it also reports success.

**State the limitation next to the number.** The transformer is
non-autoregressive and the ConvLSTM is not; Mamba's scan is sequential; the
asymptotic argument does not apply at `T≤8`. Each of these makes some
architecture look better or worse for reasons unrelated to the claim being made.

---

## Status and honest limits

| Phase | State |
| --- | --- |
| 1 — Scaffold | Complete and verified |
| 2 — Data pipeline | Complete and verified |
| 3 — Baselines and training | Complete and verified |
| 4 — SSMs and experiments | Complete and verified |
| 5 — Real data | Complete; quality numbers are budget-limited, see below |

The EarthNet2021 format constants are now validated against a real download
rather than against documentation: 30 frames, `(128, 128, 7, 30)` float16,
mean reflectance 0.109, mean valid fraction 0.76.

**What has not been done.** No model is trained to *convergence* — eight epochs is
a compute budget, not a stopping criterion, and it was set by the slowest
architecture so that all four could be compared at equal cost. The quality
numbers are real and reproducible, but they measure four architectures under a
small fixed budget rather than four architectures at their best.

Also outstanding: the frozen foundation-encoder comparison, GPU verification of
mixed precision and peak-memory measurement, the official test tracks (their
context and target files need joining), and residual forecasting — predicting
the *change* from the last observed frame rather than the absolute image, which
would start every model at the persistence baseline instead of asking it to
rediscover the scene.
