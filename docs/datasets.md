# Datasets

TinyEarth **does not redistribute any dataset**. `data/` is git-ignored.

You do not need EarthNet2021 to run this repository. The default dataset is
synthetic and requires no download:

```bash
tinyearth-data +experiment=data_smoke
```

---

## The sample contract

Every dataset yields the same structure, so swapping a data source never requires
touching a model, a loss or the training loop:

```python
{
    "images":      Tensor[T, C, H, W],   # history frames
    "target":      Tensor[K, C, H, W],   # frames to forecast
    "metadata":    SampleMetadata,       # provenance
    "images_mask": Tensor[T, 1, H, W],   # optional; 1 = cloud-free
    "target_mask": Tensor[K, 1, H, W],   # optional
}
```

Tensors are **time-major, channels second** (`[T, C, H, W]`), matching `Conv3d` and the
`[B, T, C, H, W]` batch layout used throughout. This avoids a permute in the encoder.

### Why `target` is `[K, C, H, W]` and not `[C, H, W]`

The original specification called for `[C, H, W]`, which is the horizon-1 case. But the
experiment plan sweeps forecast horizon over 1, 2, 4, 8. Squeezing the axis at `K == 1`
would make tensor *rank* depend on a config value, forcing every loss, metric and collate
function to branch on horizon. The time axis is therefore always present; a horizon-1
target is `[1, C, H, W]`.

---

## EarthNet2021

[EarthNet2021](https://www.earthnet.tech/) frames Earth surface forecasting as guided
video prediction: predict 20 future Sentinel-2 frames from 10 context frames plus
topography and weather.

### Format

Each minicube is a compressed `.npz`:

| Array | Shape | Contents |
| --- | --- | --- |
| `highresdynamic` | `(128, 128, 7, T)` | Sentinel-2, 20 m, 5-daily |
| `highresstatic` | `(128, 128)` | EU-DEM elevation, 20 m |
| `mesodynamic` | `(80, 80, 5, 5T)` | E-OBS weather, 1.28 km, daily |
| `mesostatic` | `(80, 80)` | Elevation, 1.28 km |

`highresdynamic` channels are `[blue, green, red, nir, cld, scl, cldmsk]`. Reflectance is
nominally in `[0, 1]` and contains NaNs. Training cubes hold 30 frames; the official
protocol forecasts frames 10–29 from frames 0–9.

### Two conventions that must not be got wrong

Both are inherited from the official `earthnet` toolkit and are load-bearing for
comparability with published numbers.

**1. Imagery is the first four channels; the quality mask is the last.** Indexing the mask
as `[-1]` rather than `[6]` is what the toolkit does, and keeps the reader working on test
cubes that carry fewer channels.

**2. `cldmsk == 1` means cloudy.** Validity is `1 - cldmsk`.

> Getting the mask polarity backwards trains the model **exclusively on cloud**, and does
> so silently — losses still decrease. `tests/test_minicube.py::TestMaskPolarity` exists
> specifically to catch this.

### One deliberate deviation

TinyEarth also marks NaN imagery pixels invalid; the toolkit only zero-fills them.
Zero-filled NaN is indistinguishable from genuine zero reflectance, so a model trained
against it learns to predict black where data is missing. Set `nan_is_invalid: false` when
reproducing official EarthNetScore numbers.

### Download

```bash
pip install -e ".[data]"
python scripts/download_earthnet2021.py --root data/earthnet2021
python scripts/download_earthnet2021.py --instructions   # manual route
```

Sizes: train ~100 GB (~23000 cubes), each test track ~5–15 GB (~4000 cubes).

Expected layout:

```
data/earthnet2021/
├── train/
├── iid_test_split/
├── ood_test_split/
├── extreme_test_split/
└── seasonal_test_split/
```

Then:

```bash
tinyearth-data data=earthnet2021
```

---

## Splits

| Split | Source | Tests generalisation to... |
| --- | --- | --- |
| `train` | `train/` minus held-out | — |
| `val` | carved from `train/` | — (model selection) |
| `iid_test` | `iid_test_split/` | held-out cubes, same distribution |
| `ood_test` | `ood_test_split/` | geographically disjoint regions |
| `extreme_test` | `extreme_test_split/` | drought and heatwave events |
| `seasonal_test` | `seasonal_test_split/` | a full seasonal cycle |

### There is no official validation split

One is required regardless — selecting a model on a test track invalidates every number
reported against it. TinyEarth carves validation out of the training cubes.

**The partition is by hash of the cube identifier, not by a seeded shuffle.** Both are
deterministic, but only hashing is stable when cubes are added or removed. A shuffle
reassigns every cube when the file count changes, silently moving previously-validation
cubes into training and contaminating comparisons against earlier runs. Hashing assigns
each cube independently, so existing assignments never change.

BLAKE2b is used rather than `hash()`, whose string seed varies per process.

```python
from tinyearth.datasets import assign_partition, summarise_partition
assign_partition("29SND_2017-06-20_cube_0", val_fraction=0.1)  # -> Split.TRAIN or Split.VAL
```

> **Small-dataset caveat.** Because each cube is assigned independently, a low
> `val_fraction` can round down to an *empty* validation split when there are few cubes.
> With 6 synthetic cubes, `val_fraction: 0.1` yields zero. The synthetic config therefore
> uses `0.25`. The error message for this case is explicit about the cause.

---

## Windowing

A cube is a long sequence; a sample is a short `(history, forecast)` window cut from it.

**Train/val use sliding windows** — every valid start offset, advanced by `stride`. A
30-frame cube with `history=4, horizon=1` yields 26 samples.

**Test tracks use a single anchored window per cube**, whose history ends exactly at the
context boundary (frame 10). This keeps evaluation comparable across runs and matches the
published protocol.

```python
from tinyearth.datasets import WindowSpec, WindowMode, generate_windows

generate_windows(30, WindowSpec(history_length=4, horizon=1))            # 26 windows
generate_windows(30, WindowSpec(history_length=4, horizon=2),
                 mode=WindowMode.ANCHORED, context_length=10)            # 1 window: 6..12
```

---

## Cloud masking

Two independent knobs.

**Fill policy** — what the model sees where data is missing:

| `mask_policy` | Behaviour | Trade-off |
| --- | --- | --- |
| `keep` | leave original values | honest, but cloud injects high-frequency structure |
| `zero` | set invalid to 0 | simple; indistinguishable from dark water (**default**) |
| `mean` | per-frame, per-channel valid mean | stable frame statistics, no OOD values |

**Window filtering** — `min_valid_fraction` rejects windows whose *target* is cloudier than
allowed. Filtering on the target rather than the history is deliberate: filtering on
history would discard exactly the samples where forecasting from partial context is most
valuable.

Masks are always propagated to the sample, so Phase 3 losses can exclude invalid pixels
from the objective regardless of fill policy. That is what actually matters — with a masked
loss the fill value is largely irrelevant for the target, though it still affects the
encoder input.

---

## Normalisation

Sentinel-2 reflectance is already in `[0, 1]`, so normalisation is a modelling choice.

| `kind` | Effect | Pairs with |
| --- | --- | --- |
| `identity` | unchanged (**default**) | sigmoid decoder output; loss reads as reflectance error |
| `standardize` | zero mean, unit variance per channel | unbounded decoder; metrics must invert first |

Statistics come from the **training split only**, with masked pixels excluded — cloud is
near-white and would bias every channel mean upward.

```bash
python scripts/compute_dataset_statistics.py --dataset earthnet2021
```

Then set:

```yaml
normalization:
  kind: standardize
  statistics_path: earthnet2021_channel_statistics.json
```

Every normaliser is invertible, because predictions must return to reflectance space
before being scored or plotted.

---

## Caching

Consecutive windows overlap heavily — with `history=4, horizon=1, stride=1`, 26 samples
from one cube all read the same file. An LRU cache of decoded cubes turns ~26
decompressions into one. This is the highest-value optimisation in the pipeline and is on
by default (`cache_size: 4`).

The cache is **per worker process**, so memory is `num_workers × cache_size × cube_size`.
Raise it if you have RAM; a full 128×128×7×30 float32 cube is ~13 MB decoded.

---

## Synthetic data

`data=synthetic` writes **real `.npz` files in the real layout**, so the production reader
is what gets exercised — not a parallel implementation that could drift.

Content is *plausible*, not realistic: a smooth spatial gradient with a seasonal cycle,
plus drifting cloud blobs. Sufficient to verify shapes, masking, windowing, normalisation
and determinism.

> **Results from synthetic data are meaningless and must never be reported.** The dataset
> logs a warning at construction saying so.

Cubes are written under `paths.cache`, not `paths.data`, because they are derived
artefacts rather than source data.

---

## Reproducibility

The pipeline is deterministic given a seed. `tinyearth-data` verifies this directly: two
independently constructed loaders with the same seed must produce bit-identical first
batches.

Two easily-missed requirements are wired into `build_dataloader` by default:

- **`generator=`** controls shuffle order. Left to the global RNG, epoch order depends on
  how many random numbers the *model* drew — adding a dropout layer silently changes the
  data order.
- **`worker_init_fn=`** reseeds `random` and `numpy` per worker. PyTorch seeds each
  worker's `torch` RNG but leaves the other two identical, so N workers apply the *same*
  augmentations.

See [reproducibility.md](reproducibility.md).

---

## Citation

```bibtex
@inproceedings{requena2021earthnet2021,
  title     = {EarthNet2021: A Large-Scale Dataset and Challenge for Earth Surface
               Forecasting as a Guided Video Prediction Task},
  author    = {Requena-Mesa, Christian and Benson, Vitus and Reichstein, Markus and
               Runge, Jakob and Denzler, Joachim},
  booktitle = {CVPR Workshops},
  year      = {2021},
}
```
