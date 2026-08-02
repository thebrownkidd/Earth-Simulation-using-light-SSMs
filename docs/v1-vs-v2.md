# v1 vs v2: skip connections + GDL loss

**Status: both v1 and v2 have been trained and evaluated. Read the caveat
below before trusting any number in this document** — the two runs are not on
the same validation set, for a reason explained there.

## What v2 changes, and nothing else

Three independent changes, detailed in `docs/project-log.md` Phase 6 and in
the comments of `configs/experiment/earthnet_v2.yaml`:

1. **Encoder-to-decoder skip connections.** The encoder's last-observed-
   context-frame features are fused into the decoder at every resolution
   level, giving the decoder a full-resolution reference for "what things
   looked like most recently" instead of reconstructing detail purely from
   the backbone's compressed latent. Adds 43,232 parameters to the fixed
   encoder/decoder component (228,548 → 271,780), identically for all four
   backbones — which is why the size tiers below are recalibrated
   (`SIZE_TIERS_SKIP`, not `SIZE_TIERS`) rather than reused.
2. **A gradient-difference (GDL) loss term**, added alongside L1 (weight
   1.0, untuned) rather than replacing it. GDL compares gradient *magnitude*
   between prediction and target rather than raw pixel differences, so it
   penalises a model that reproduces the mean but smooths over real edges —
   L1's blind spot, and the visible failure mode in v1's filmstrip figure.
3. **Resumable training** (`--resume`), which does not change what gets
   measured but is what made a second multi-hour, multi-backbone run
   practical to manage on the same CPU-only machine as v1.

Protocol, crop size, batch size, epoch budget, optimiser, schedule, and the
hash-based train/val split are all **identical to v1**. Architecture is the
only variable this comparison isolates — see "Isolating the architecture
change" in `configs/experiment/earthnet_v2.yaml` for why the same validation
cubes matter here.

**Mamba is excluded from v2**, carrying forward v1's decision rather than
re-litigating it silently: its selective scan has no fused kernel on this
platform, costs ~5x S4D per step, and v2 adds cost (GDL's extra gradient
computation, the skip pathway's extra decoder convolutions) that does not
improve its relative position.

## Stated expectation, before training (kept verbatim, checked below rather than edited after the fact)

Both changes were expected to **narrow, not eliminate**, the quality gap
between backbones on pixel-level and structural metrics. Sharpness now partly
comes from the skip pathway and the GDL term, both identical across every
backbone by construction — so if v2's backbone-to-backbone spread on
`val/ssim` shrinks relative to v1's, that would be the intended effect of
adding shared machinery, not a sign the comparison broke.

**This did not happen.** See "Did the spread shrink?" below — it is reported
as a miss, not edited away now that the numbers exist.

## A caveat that affects every number below

**The two runs are not scored on the same validation cubes.** Between v1's
run and v2's, the dataset grew from ~1,650 to 3,150 cubes (an unrelated
data-collection task run in the same window), and the hash-based split (see
`docs/project-log.md` Phase 2) assigns new cubes to train/val independently of
what existed before — it does not keep the validation *set* fixed when the
pool of cubes grows, only each individual cube's assignment once it exists.
v1 scored 161 windows; v2 scored 312.

`configs/experiment/earthnet_v2.yaml` explicitly said to avoid this ("do not
point this at data downloaded after v1's run") and to say so plainly if it
happened anyway rather than silently comparing numbers that differ along two
dimensions. It happened anyway — the additional download was already in
progress in parallel with this work by the time training started, and
re-running v1 against the frozen 1,650-cube set was not done. **Every v1-vs-v2
delta below therefore mixes an architecture change with a dataset-size
change**, and should be read as suggestive, not as an isolated measurement of
the architecture alone.

One piece of direct evidence the two validation sets differ: the
parameter-free **persistence** baseline itself moved, 0.02987 → 0.03045 MAE
and 0.8031 → 0.8044 SSIM, on cubes nothing was trained on and that use no
learned parameters at all. That shift is dataset composition, not
architecture — and it is the honest reason to read the margin-over-persistence
numbers below as the more trustworthy comparison: they cancel out most of
the shared "this dataset's scenes are somewhat different" effect, since both
the learned model and its reference are scored on the same 312 (or 161)
windows within a single run.

## Results

### v1 (baseline) — 161 held-out validation windows, full 128×128 scenes

| Backbone | val MAE ↓ | RMSE ↓ | PSNR ↑ | SSIM ↑ | SAM ↓ | latency | parameters |
| --- | --- | --- | --- | --- | --- | --- | --- |
| S4D | 0.02815 | 0.04205 | 27.87 | 0.7390 | 7.10 | 611 ms | 2,076,916 |
| Transformer | 0.02819 | 0.04243 | 27.86 | 0.7615 | 6.95 | 384 ms | 2,112,964 |
| ConvLSTM | 0.02937 | 0.04390 | 27.56 | 0.7400 | 7.35 | 367 ms | 2,221,636 |
| Mamba | *(excluded — see project-log.md Phase 5)* | | | | | | |
| *Persistence (free)* | *0.02987* | *0.04447* | *27.50* | *0.8031* | *7.00* | *free* | *0* |
| *Climatology (free)* | *0.03518* | *0.04947* | *26.41* | *0.7822* | *7.93* | *free* | *0* |

### v2 (skip connections + GDL) — 312 held-out validation windows, full 128×128 scenes, larger dataset (see caveat above)

| Backbone | val MAE ↓ | RMSE ↓ | PSNR ↑ | SSIM ↑ | SAM ↓ | latency | parameters |
| --- | --- | --- | --- | --- | --- | --- | --- |
| **Transformer** | **0.02593** | **0.03925** | **28.47** | **0.8081** | **6.06** | 401 ms | 1,930,908 |
| S4D | 0.02799 | 0.04207 | 27.87 | 0.7908 | 6.40 | 642 ms | 2,023,548 |
| ConvLSTM | 0.02943 | 0.04396 | 27.47 | 0.7796 | 6.40 | 553 ms | 1,920,420 |
| Mamba | *(excluded — see below)* | | | | | | |
| *Persistence (free)* | *0.03045* | *0.04661* | *27.08* | *0.8044* | *6.36* | *free* | *0* |
| *Climatology (free)* | *0.03358* | *0.04921* | *26.52* | *0.7932* | *7.08* | *free* | *0* |

### Change, v1 → v2, per backbone (mixes architecture + dataset-size, per the caveat)

| Backbone | ΔMAE | ΔSSIM | Margin over persistence (MAE), v1 → v2 |
| --- | --- | --- | --- |
| S4D | −0.6% | **+7.0%** | 5.8% → 8.1% |
| ConvLSTM | +0.2% | **+5.3%** | 1.7% → 3.4% |
| Transformer | **−8.0%** | **+6.1%** | 5.6% → **14.9%** |

Parameter counts are measured directly from the built models
(`tinyearth-model +experiment=earthnet_v2 model=<name>`), independent of
training or the dataset. Reproduce the quality numbers with:

```bash
python scripts/run_earthnet_study.py --experiment earthnet_v2 --backbones s4d convlstm transformer
python scripts/evaluate_earthnet.py --group earthnet_v2 --output experiments/results/earthnet_v2.json
```

## Reading the actual numbers against the stated criteria

**Did the spread shrink?** No — it widened, from 0.0225 to 0.0284 SSIM
between the best and worst backbone. The stated hypothesis (shared skip+GDL
machinery narrows the backbone-to-backbone gap) is **not supported** by this
run. If anything, the backbone that already led on v1 (Transformer) pulled
further ahead on v2, on every metric. Recorded as a miss rather than quietly
dropped: whatever v1's SSIM gap was measuring, it was not eliminated by
giving every backbone the same decoder-side help.

**Did SSIM specifically improve, more than MAE did?** Yes, clearly, for all
three backbones (+5.3% to +7.0%), while MAE moved in both directions and by
smaller magnitudes (−8.0% to +0.2%). This is consistent with GDL doing what
it was added to do — penalising blur specifically, not accuracy generally —
independent of the dataset-size caveat, since SSIM improved for every
backbone regardless of which direction MAE moved. **Transformer is now the
only backbone (v1 or v2) that beats the persistence baseline on SSIM**
(0.8081 vs 0.8044) — no v1 model did, including v1 Transformer (0.7615).

**Is the ranking preserved?** Mostly. ConvLSTM stayed worst on MAE in both
versions. S4D and ConvLSTM were within noise of each other on v1 SSIM (0.7390
vs 0.7400) and separated more clearly on v2 (0.7908 vs 0.7796, S4D now
ahead) — a near-tie resolving differently is a weaker claim than a genuine
rank flip, and is not read as one here. Transformer led v1 on SSIM and v2 on
every metric, including latency (401 ms, the cheapest of the three, same as
v1). The architecture question and the decoder-improvement question stay
reasonably separable: nothing here suggests v2's improvements are backbone-
specific in a way that would make "which backbone is best" a different
question under v2 than under v1.

**Net reading.** GDL appears to be doing real work — the SSIM gain is
consistent in direction and larger than plausible dataset-composition noise
across all three backbones, and the persistence-relative MAE margin grew for
all three too (most for Transformer, 5.6% → 14.9%). The spread-narrowing
hypothesis specifically was wrong, and that is worth taking seriously rather
than explaining away: it suggests the v1 gap between backbones was not purely
a "who can route detail to the decoder" question that a shared skip pathway
would flatten. **This is not a clean single-variable result** — the dataset
grew between the two runs — and the honest next step, stated in
`docs/project-log.md` Phase 6, is rerunning v2 against the exact 1,650-cube
set v1 used before treating any of these deltas as settled.
