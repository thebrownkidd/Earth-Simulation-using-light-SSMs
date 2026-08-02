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
| ~~Run the four models concurrently~~ **Reversed — see below** | A thread sweep showing four threads at ~95% of sixteen-thread throughput (s4d 3.96 vs 4.09 samples/s) looked like an invitation to run four processes at four threads each. Measured, not inferred: this was wrong. See "Bugs and surprises" below. |
| Added persistence and climatology | A learned MAE is uninterpretable alone. Earth surface imagery is mostly static over 100 days, so a model that learned only to echo its input would still post a good number. |

**Bugs and surprises**

- **Concurrent training was 4.5× slower, not ~4× faster.** The thread sweep above measured one
  process against an idle machine, which says nothing about four processes competing for
  memory bandwidth — and these convolutional workloads are bandwidth-bound on a mobile CPU, not
  thread-bound. Measured directly: a ConvLSTM epoch under four concurrent jobs took 1188 s
  against a 265 s solo estimate, **4.5× slower each**, so the aggregate throughput was *below*
  running one model at a time. `scripts/run_earthnet_study.py --jobs` now defaults to `1`; the
  four-times-faster intuition from the thread sweep was simply wrong, and is kept in this log
  as a lesson rather than quietly edited into a decision that always looked obvious.
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

## Phase 6 — Skip connections, a blur penalty, and resumable training

**Delivered:** encoder-to-decoder skip connections (a second, opt-in
architecture, `v2_skip_gdl`, run alongside v1 rather than replacing it), a
gradient-difference loss term, `--resume` for the training loop, and a
recalibrated size-tier table for the larger encoder/decoder skip connections
produce.

**Why a second architecture rather than editing the first.** Phase 5's
filmstrip figure showed the v1 models holding an almost static forecast for
100 days — a known failure mode of an L1-only decoder with no path back to
full-resolution input detail. Two standard fixes exist: give the decoder a
skip pathway to the input, and penalise blur directly rather than relying on
L1 to discourage it as a side effect. Both are applied here, and the existing
v1 results, config and checkpoints are left untouched precisely so the
comparison between "before" and "after" stays available rather than being
overwritten by "after."

**Decisions**

| Decision | Reasoning |
| --- | --- |
| Skip pathway carries the LAST context frame only, broadcast across all K forecast steps | A plain U-Net skip assumes matched input/output frame counts; this project's decoder emits `horizon=20` frames from `history=10`, so there is no frame-15-of-input to match to frame-15-of-output. The last observed frame is the only "what did this look like most recently" signal that is well-defined regardless of K. |
| Implemented inside `Encoder`/`Decoder`, not any backbone | The controlled comparison requires the fixed components to stay identical across backbones. A per-backbone skip implementation could not be verified identical, and would reopen exactly the failure mode the matched-parameter bug (Phase 5) was fixed to prevent. |
| Selecting the last frame is indexing, not a learned op | `CNNEncoder(skip_connections=True)` adds zero parameters over the plain encoder — confirmed by `test_no_parameters_are_added_by_enabling_skip_connections`. All of the skip pathway's new parameters live in the decoder's fusion convolutions, where the actual mixing happens. |
| GDL added as a term, L1 kept as the base | Task instruction, and independently correct: MSE/L2 is documented to increase blur relative to L1, not reduce it. GDL compares gradient *magnitude* between prediction and target rather than raw pixel differences, so it penalises exactly the failure mode L1 cannot see — a prediction that gets the mean right but has no edges. |
| GDL masked the same way as every other loss | A gradient spans two neighbouring pixels; it is valid only where both are, i.e. the product of the two shifted masks. Getting this wrong would optimise the model against gradients computed across cloud boundaries, which are not real edges in the data. |
| A second, separately-calibrated size-tier table (`SIZE_TIERS_SKIP`) | Skip connections add 43,232 parameters to the fixed encoder/decoder component (228,548 → 271,780 at the standard `base_channels=32, depth=2`). Reusing `SIZE_TIERS` for a skip-connection model would silently target the wrong parameter budget — not by a huge amount, but `convlstm/tiny` at v1's width of 80 overshoots 2M by 13.2% once the skip overhead is added, outside the tolerance the tests enforce. |
| `architecture_version` recorded in the config, every checkpoint, and `summary.json` | So a run's numbers can always be traced to the code that produced them, and so `--resume` has something concrete to check before loading weights into a possibly-incompatible model. |
| `--resume` refuses a checkpoint whose `architecture_version` disagrees, when a caller states which version it expects | A v1 checkpoint has no skip-fusion convolutions for v2 code to load weights into; loading it anyway would either crash on a shape mismatch (best case) or silently succeed with the fusion convs at random initialisation (worst case, and what actually motivated making the check opt-in-strict rather than best-effort). |
| Mamba excluded from v2, explicitly, as a carried-forward decision | Same constraint as v1: Mamba's per-step cost is ~5x S4D's and flat in batch size, because its selective scan has no fused kernel on this platform. v2 adds GDL's extra gradient computation and the skip pathway's extra decoder convolutions on top of every backbone's existing cost, which does not improve Mamba's relative position. Stated here so a future reader does not have to guess whether this was reconsidered and rejected, or simply never revisited. |

**The resume mechanism, and what it does and does not guarantee.** Model
weights, optimiser state, scheduler state, the training loader's own shuffle
generator (a separate RNG stream from the ones `capture_rng_state()` covers —
see `loaders.py`, seeded explicitly so shuffle order does not depend on how
many random draws the model happens to make) and the global RNG state all
round-trip through the checkpoint. Verified end to end: `tests/test_training.py`
trains 2+2 epochs with a checkpoint in between and asserts the final weights
and validation metrics match an uninterrupted 4-epoch run within floating-point
tolerance — not by asserting on each piece individually, but by showing that
if any piece failed to round-trip, the two runs would visibly diverge and the
test would catch it regardless of which piece was at fault.

That test deliberately uses `shuffle=False` loaders, matching the existing
fixture convention in the same file. The loader-generator state is still
captured and restored by `save_checkpoint`/`load_checkpoint` for the general
case (real EarthNet2021 configs use `shuffle=True`), but verifying that path
bit-exactly would require reasoning about a second, independently-advancing
RNG stream inside the round-trip test, which the existing test suite's own
convention already avoids for exactly this reason.

**Went wrong**

- The first version of `CNNDecoder`'s skip-fusion construction crashed
  immediately whenever `skip_channels` was empty (the default, i.e. every
  v1 decoder): `zip(level_channels, reversed([]), strict=True)` raises when
  the two sequences have different lengths, and an empty `skip_channels`
  against a non-empty `level_channels` is exactly that. Every existing test
  in the repository builds a `CNNDecoder` without skip connections, so this
  would have failed the entire suite instantly — caught before it did,
  by building a decoder with no skip connections as the very first sanity
  check after writing the encoder/decoder changes, before writing a single
  test.
- The initial `--step 16` calibration search (matching v1's grid) landed
  `convlstm/tiny` at 13.2% from its 2M target once skip connections' fixed
  overhead was added — outside the 12% tolerance `tests/test_sizes.py`
  enforces. The overhead is a bigger proportional bite out of the smallest
  tier than the larger ones, and the default grid was too coarse to find a
  closer width nearby. Recalibrated at `--step 8`, which found `hidden_dim=72`
  (−4.0% from target) for the same backbone and tier.
- A first draft of the GDL blur-monotonicity test used a checkerboard target.
  Blurring a checkerboard is *not* monotonic at every step: its energy is
  concentrated at the Nyquist frequency, and a discrete Gaussian kernel's
  response there aliases rather than smoothly attenuates, so the loss dipped
  slightly non-monotonically between two blur levels on a small grid. Fixed
  by testing against a single step edge instead, whose blur response is
  well-behaved (monotonic, no periodicity to alias against) — this is a
  property of the *test's* synthetic signal, not of GDL itself.
- **The dataset grew between v1's run and v2's**, from ~1,650 to 3,150 cubes,
  because a separate data-collection task ran in the same window as this one.
  `earthnet_v2.yaml`'s own comments said to avoid exactly this ("run against
  the SAME validation cubes v1 was scored on") and to say so plainly if it
  happened anyway. It happened anyway — training was already committed to the
  larger dataset by the time this was noticed. Every v1-vs-v2 delta in
  `docs/v1-vs-v2.md` therefore mixes an architecture change with a
  dataset-size change; the document is explicit about this rather than
  presenting the comparison as clean. A repeat run against the frozen
  1,650-cube set is the honest next step, not yet done.
- **v2 ConvLSTM's training appeared to take 8h58m**; the real number is under
  40 minutes. Six epochs logged normally at ~533s each, then a 7h22m gap
  opened between the last epoch finishing and the run actually completing
  (during efficiency profiling, which normally takes seconds). This is the
  same false alarm Phase 4 already logged once — the machine suspending while
  idle, not a regression — confirmed here the same way: by reading the
  per-epoch timestamps rather than trusting the wall-clock total. Recorded a
  second time because the instinct to treat a large duration as a bug rather
  than checking the timestamps first is exactly the mistake Phase 4's note
  exists to prevent, and it nearly repeated.

**What "v2" means, now that both runs are measured**

Full results, per-backbone deltas, and a reading against the stated
pre-training hypothesis are in `docs/v1-vs-v2.md`. Briefly: the hypothesis
that shared skip+GDL machinery would **narrow** the backbone-to-backbone
spread was **wrong** — it widened (0.0225 → 0.0284 SSIM between best and
worst backbone) — and that is reported as a miss, not revised after the fact.
What did hold up: SSIM improved for all three backbones (+5.3% to +7.0%),
consistent with GDL targeting blur specifically rather than accuracy
generally, and Transformer under v2 is the first model in either version to
beat the persistence baseline on SSIM. Both readings carry the dataset-size
caveat above.

GDL's weight (`lambda=1.0`) was not swept. The config comment says it is a
config value precisely so it can be swept later without a code change; that
sweep has not been run.

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
| 6 — Skip connections, GDL, resume | Complete; v2 trained and measured, but against a larger dataset than v1 — see `docs/v1-vs-v2.md` |

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
context and target files need joining), residual forecasting — predicting the
*change* from the last observed frame rather than the absolute image, which
would start every model at the persistence baseline instead of asking it to
rediscover the scene — a v2 rerun against the exact 1,650-cube set v1 used
(the current v2 numbers are on 3,150 cubes, so every v1-vs-v2 delta in
`docs/v1-vs-v2.md` mixes an architecture change with a dataset-size change),
and a sweep over GDL's loss weight.
