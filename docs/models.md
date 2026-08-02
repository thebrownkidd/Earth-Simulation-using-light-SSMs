# Models

## The architecture is fixed

```
encoder  ->  temporal backbone  ->  decoder  ->  forecast
             ^^^^^^^^^^^^^^^^^
             the only thing that varies
```

| Component | Input | Output |
| --- | --- | --- |
| `Encoder` | `[B, T, C, H, W]` | `[B, T, D, h, w]` |
| `TemporalBackbone` | `[B, T, D, h, w]` | `[B, K, D, h, w]` |
| `Decoder` | `[B, K, D, h, w]` | `[B, K, C, H, W]` |

`T` = history length, `K` = forecast horizon, `D` = latent channels, `h, w` = downsampled
spatial size.

This control is the whole experiment. If the fixed components differ between two runs, a
difference in results is not attributable to the backbone.
`tests/test_models.py::TestControlledComparison` asserts that encoder and decoder parameter
counts are byte-identical across backbones, and
`tests/test_cli.py::test_swapping_the_backbone_changes_only_the_backbone` re-checks it
through the actual CLI.

### Why the backbone emits `K` steps

It is the component under study, so the sequence-to-sequence mapping lives entirely inside
it. If the decoder expanded one summary state into `K` frames, that expansion would be
shared machinery whose capacity differs by architecture — and it would contaminate the
comparison.

### Why encoder and decoder hold no temporal parameters

Frames are folded into the batch dimension and processed independently. Every bit of
temporal modelling capacity belongs to the backbone.
`test_frames_are_encoded_independently` pins this down: reversing the frame order must
exactly reverse the output.

---

## Swapping the backbone

One config override. Nothing else changes.

```bash
tinyearth-train model=convlstm
tinyearth-train model=transformer
tinyearth-train model.backbone.kwargs.hidden_dim=256
```

Backbones register themselves, so configs select them by name:

```python
from tinyearth.models.temporal import TEMPORAL_BACKBONES

@TEMPORAL_BACKBONES.register("mamba")
class MambaBackbone(TemporalBackbone):
    """Selective state space temporal backbone."""

    def forward(self, latents: Tensor, horizon: int) -> Tensor:
        ...
```

`model.backbone.kwargs` is deliberately untyped — the one escape hatch in the schema.
Enumerating every backbone's arguments would force a schema change for each new backbone,
defeating the registry. Unknown keys still fail loudly at construction.

---

## The four backbones

| Key | Architecture | Mixes over time by |
| --- | --- | --- |
| `convlstm` | ConvLSTM (Shi et al., 2015) | Gated recurrence; sequential |
| `transformer` | Temporal transformer | Attention over `T`; parallel |
| `s4d` | Diagonal SSM (Gu et al., 2022) | FFT convolution; parallel |
| `mamba` | Selective SSM (Gu and Dao, 2023) | Input-dependent scan; sequential |

`convlstm` is the only autoregressive one; the other three emit all `K` forecast
steps in a single pass. State that when comparing latency -- it is not a property
of the mixing mechanism.

Measured costs at matched parameter budgets, and the finding that parameter
efficiency does **not** imply FLOP efficiency, are in
[phase-4.md](phase-4.md#the-finding-parameter-efficiency-is-not-flop-efficiency).

## Baselines (Phase 3)

Their purpose is **correctness and a reference point**, not benchmark numbers.

### ConvLSTM

Shi et al., NeurIPS 2015. Convolutions replace the LSTM's matrix multiplications, so the
hidden state keeps its spatial layout. Encoder-decoder (seq2seq): consume the history, then
run `horizon` autoregressive steps.

Cost profile — and this is exactly what an SSM is expected to improve on:

- **Strictly sequential in time.** Step `t` needs step `t-1`; latency grows linearly in
  `T + K` with no temporal parallelism available.
- Each step convolves over the full spatial grid.

### Temporal transformer

Attention over the **time axis only**. Each spatial location is an independent sequence.

Spatiotemporal attention is deliberately avoided for two reasons. It would stop being a
*temporal* backbone — adding spatial capacity the others lack makes differences
unattributable. And it costs `O((T·h·w)²)`: at `T=8` on a 32×32 grid that is 8192 tokens,
~67M attention entries per head per layer, against 64 for temporal-only.

**Non-autoregressive**: all `K` steps emit in one pass from positional queries. That is a
real architectural advantage over the ConvLSTM for latency, so read efficiency comparisons
with it in mind rather than crediting attention alone.

Position encodings are *fixed* sinusoidal, not learned, so a model trained at one history
length can be evaluated at another — directly useful for the history-length sweep.

---

## Measured parameter counts

`latent_dim=128`, encoder/decoder `base_channels=32, depth=2`, `n_layers=2`, 128×128 input,
`T=4 → K=2`, CPU. Reproduce with `tinyearth-model --compare`.

| Backbone | `hidden_dim` | Total | Backbone | Backbone % | GFLOPs/sample | Latency (ms) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| convlstm | 64 | 974,660 | 746,112 | 76.6% | 13.70 | 123.3 |
| convlstm | 128 | 2,605,380 | 2,376,832 | 91.2% | 33.68 | 212.2 |
| convlstm | 192 | 5,120,836 | 4,892,288 | 95.5% | 64.53 | 357.0 |
| transformer | 64 | 478,596 | 250,048 | 52.2% | 5.16 | 117.4 |
| transformer | 128 | 1,187,268 | 958,720 | 80.8% | 6.67 | 170.3 |
| transformer | 192 | 2,354,692 | 2,126,144 | 90.3% | 9.12 | 241.4 |

Fixed encoder + decoder: **228,548 parameters**, identical across every row.

### Two things worth noting

**Scaling really does happen in the backbone.** At `hidden_dim=128` the backbone holds
91% (ConvLSTM) and 81% (transformer) of parameters, so a size sweep varies the component
under study rather than fixed machinery. `ParameterBreakdown.backbone_fraction` reports
this on every run; a low value means a scaling result would not mean what it appears to.

**FLOPs scale very differently.** Tripling `hidden_dim` from 64 to 192 multiplies ConvLSTM
FLOPs by **4.7×** but transformer FLOPs by only **1.8×**. The ConvLSTM's cost is dominated
by convolutions over the full spatial grid at every timestep; the transformer's is
dominated by the fixed-cost projections, with attention itself negligible at `T=4`. This
is a real result and it sets up the Phase 4 question directly.

> These are **cost** measurements, not quality measurements. No baseline has been trained
> to convergence on real data. Nothing here says which architecture forecasts better.

---

## Losses

Adding a loss is a new module plus a config entry — never an edit to the training loop.

```yaml
model:
  loss:
    terms:
      l1: 1.0
      ssim: 0.5     # Phase 3+ once implemented
```

| Loss | Behaviour |
| --- | --- |
| `l1` | Mean absolute error. **The default.** |
| `l2` | Mean squared error. |
| `charbonnier` | `sqrt((x-y)² + ε²)`; L1-like but smooth at zero. |
| `gdl` | Gradient-difference loss — compares gradient **magnitude**, not raw pixel values. Added *alongside* `l1` in v2, not in place of it. |

L1 is the default because on satellite imagery L2 over-penalises the rare large errors
that cloud edges and shadows produce, and the usual result is a model hedging toward the
local mean — visually, a blurred forecast. **Do not reach for L2/MSE to fight blur** — it
is documented to make blur worse, not better; use `gdl` alongside `l1` instead.

### GDL: a blur penalty, not a replacement for L1

L1 rewards a model for getting the mean pixel value right and is blind to whether the
*edges* in a forecast match the edges in the target — a prediction that reproduces the
scene's overall brightness but smooths away every field boundary can still post a good L1
number. GDL closes that blind spot directly: it computes the horizontal and vertical
pixel-to-pixel difference of both prediction and target, takes the **magnitude** of each,
and penalises the difference between those magnitudes —

```
GDL = mean(| |∇ₓ pred| − |∇ₓ target| |) + mean(| |∇_y pred| − |∇_y target| |)
```

— rather than the difference of the gradients themselves, which is what distinguishes it
from simply running L1 on a gradient image. A model that blurs edges the target actually
has now pays for it here even when L1 alone would not notice.

Masked exactly like every other loss, with one subtlety: a gradient spans **two**
neighbouring pixels, so it counts as valid only where both are — the product of the two
shifted masks, not the mask at a single pixel. Getting this wrong would teach the model to
reproduce artefacts at cloud boundaries, which are not real edges in the data.

Reference: Mathieu, Couprie & LeCun, *Deep Multi-Scale Video Prediction Beyond Mean Square
Error*, ICLR 2016.

**All losses are mask-aware.** `masked_mean` divides by the *valid* pixel count, not the
total; dividing by the total would silently scale the loss down in proportion to cloudiness
and make cloudy batches contribute less gradient than they should. A fully-cloudy batch
returns 0, not NaN.

Multi-term objectives go through `CompositeLoss`, which exposes per-term values so a
stalled objective can be distinguished from one where a single term dominates. A
single-term objective uses the same path, so the trainer has one code path.

---

## Normalisation and the decoder output

`decoder.output_activation: sigmoid` bounds predictions to `[0, 1]`, matching reflectance.
Pair it with `data.normalization.kind: identity`.

Under `standardize` the targets are no longer bounded, so set `output_activation: none`.
The mismatch is a silent trap — the loss plateaus for a reason that looks like an
optimisation failure — so `tinyearth-train` warns when it detects the combination.

---

## Skip connections and architecture versions (v2)

Every backbone above trains under two architecture versions, selected independently of which
backbone is chosen: **v1** (this document's baseline) and **v2**
(`model.skip_connections: true`, plus the `gdl` loss term above). `model.architecture_version`
tags which is which — recorded in the resolved config, every checkpoint and `summary.json`, so
a run's numbers can always be traced back to the code that produced them, and so `--resume`
can refuse to load a checkpoint into structurally incompatible code (see
`docs/reproducibility.md`).

### The problem a plain U-Net skip connection can't solve here

A standard encoder-decoder skip connects encoder stage `i`'s output at input frame `t` to
decoder stage `i`'s input at output frame `t` — it assumes matched frame counts. This project's
decoder emits `horizon=20` frames from `history=10`: there is no frame 15 of the input to
match to frame 15 of the output. The fix is to carry a signal that does not depend on frame
alignment at all — **the last observed context frame**, which is well-defined regardless of
`K`.

### What actually happens

`CNNEncoder(skip_connections=True)` exposes, via `forward_with_skips()`, the feature map at
every downsampling stage — but only for the **last** frame in the input sequence, selected by
plain indexing after each stage runs. That selection adds **zero parameters**: the encoder
still holds no temporal-mixing weights, exactly as the controlled comparison requires.

`CNNDecoder(skip_channels=...)` fuses each of those features into the matching resolution
level of its upsampling path. Because the decoder produces `K` steps but the skip only has one
frame's worth of features, the same skip feature is **broadcast across all `K` steps** before
a 1×1 convolution concatenates and projects it back down to the decoder's own channel count —
the skip pathway informs every forecast step identically, since it carries no information about
*which* step is being decoded, only what the scene looked like most recently.

```python
# tinyearth.models.decoders.cnn.CNNDecoder._fuse (simplified)
broadcast = skip.unsqueeze(1).expand(batch, steps, *skip.shape[1:]).reshape(batch * steps, ...)
fused = fusion_conv(torch.cat([decoder_features, broadcast], dim=1))
```

Implemented inside the shared `Encoder`/`Decoder` base classes, not any individual backbone —
the controlled comparison requires the fixed components to stay identical across backbones,
and a per-backbone skip implementation could not be verified identical.

### The parameter cost, and why it needs its own size-tier table

Skip connections add **43,232 parameters** to the fixed encoder/decoder pair — 228,548 (v1) →
271,780 (v2), at the standard `base_channels=32, depth=2`. That is enough to matter at the
"tiny" (~2M) tier: reusing v1's `SIZE_TIERS` widths for a v2 model would overshoot the 2M
target by over 10% for some backbones once the extra fixed parameters are counted. `models.sizes`
therefore ships a second, separately-calibrated table, `SIZE_TIERS_SKIP`, and
`model.skip_connections` selects which table `resolve_hidden_dim` reads from. Recalibrate
either with `scripts/calibrate_sizes.py`, passing `--skip-connections` for the v2 table.

### The expected effect, stated before it was measured

Because the skip pathway and the GDL term are identical across every backbone by construction,
both were expected to **narrow, not eliminate**, the backbone-to-backbone quality gap seen in
v1 — sharpness would now partly come from shared machinery rather than solely from whichever
backbone routed detail to the decoder best. Measuring it found the opposite: the gap widened.
See the "Version 2" findings in the top-level README and `docs/v1-vs-v2.md` for the full result,
reported as a miss against the stated hypothesis rather than revised after the fact.

## Adding a backbone (Phase 4)

1. Subclass `TemporalBackbone` in `src/tinyearth/models/temporal/`.
2. Implement `forward(latents: [B,T,D,h,w], horizon: int) -> [B,K,D,h,w]`.
3. Decorate with `@TEMPORAL_BACKBONES.register("name")`.
4. Add `configs/model/name.yaml`.
5. Add the name to `BACKBONE_NAMES` in `tests/test_models.py` — the parametrised
   `TestBackboneContract` then applies the full interface suite automatically.

Nothing else changes. That is the design working.
