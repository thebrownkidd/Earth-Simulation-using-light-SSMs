# Phase 2 — Dataset pipeline

**Status: complete and verified.**

## Objective

A data pipeline for EarthNet2021 that yields a fixed sample contract, so that Phases 3 and
4 can swap temporal backbones without touching data code.

Phase 2 contains **no model**. That is the intended scope.

## Delivered

| Requirement | Implementation |
| --- | --- |
| Download instructions (no redistribution) | `scripts/download_earthnet2021.py`, [datasets.md](datasets.md) |
| Dataset class | `tinyearth.datasets.EarthNet2021Dataset` |
| Dataloaders | `tinyearth.datasets.build_dataloader` |
| Temporal sequence generation | `tinyearth.datasets.windows` |
| Normalisation | `tinyearth.datasets.normalization` + `scripts/compute_dataset_statistics.py` |
| Optional cloud masking | `tinyearth.datasets.masking` |
| Train/val/test splits | `tinyearth.datasets.splits` |
| Caching | Per-worker LRU cube cache in `EarthNet2021Dataset` |
| Sample visualisation notebook | `notebooks/01_dataset_visualisation.ipynb` |
| Unit tests | 300 new tests across 8 files |

Beyond the checklist:

- **`tinyearth.datasets.synthetic`** — generates real `.npz` files in the real layout, so
  the pipeline, tests, CI and the notebook all run without the ~100 GB download.
- **`tinyearth.datasets.factory`** — the single translation point from `DataConfig` to live
  objects, keeping the dataset classes independent of Hydra.
- **`tinyearth-data`** — a CLI that builds the pipeline and reports on it.
- **`DataConfig`** — replaces the Phase 1 `Any` placeholder with a validated schema.

## The sample contract

```python
{
    "images":      Tensor[T, C, H, W],
    "target":      Tensor[K, C, H, W],
    "metadata":    SampleMetadata,
    "images_mask": Tensor[T, 1, H, W],   # optional
    "target_mask": Tensor[K, 1, H, W],   # optional
}
```

### Deviation from the original specification

The spec called for `"target": Tensor[C, H, W]`. The experiment plan also sweeps forecast
horizon over 1, 2, 4, 8. Both cannot hold unless tensor *rank* depends on a config value,
which forces every loss, metric and collate function downstream to branch on horizon.

`target` therefore always keeps its leading time axis; `[C, H, W]` is the `K == 1` case
written `[1, C, H, W]`. This is the one place Phase 2 departs from the written spec.

## Configuration

```bash
tinyearth-data +experiment=data_smoke
```

See [`configs/experiment/data_smoke.yaml`](../configs/experiment/data_smoke.yaml).

## Expected output

```
--- Splits ---------------------------------------------
  train                 4 cubes     104 windows     26 batches
  val                   2 cubes      52 windows     13 batches
  train/val overlap  0
--- Batch ----------------------------------------------
  images=(4, 4, 4, 16, 16) target=(4, 1, 4, 16, 16) target_valid=0.885
  images             (4, 4, 4, 16, 16)  min=0.0000 max=0.7154 mean=0.1909
  target             (4, 1, 4, 16, 16)  min=0.0000 max=0.7154 mean=0.2144
  images_mask        (4, 4, 1, 16, 16)  min=0.0000 max=1.0000 mean=0.8796
--- Reproducibility ------------------------------------
  same seed -> same batch True
  cube cache         hits=1 misses=3 size=3
```

`target_valid ≈ 0.88` is consistent with `synthetic.cloud_fraction=0.2` plus a 1% NaN rate.

## Verification

```bash
$ pytest -q
430 passed, 1 skipped

$ ruff check src tests scripts
All checks passed!

$ black --check src tests scripts
56 files would be left unchanged.

$ mypy
Success: no issues found in 54 source files
```

New test coverage:

| Area | Tests | Notable cases |
| --- | --- | --- |
| `test_windows.py` | 26 | exhaustive history/target disjointness, anchored protocol, all swept horizons |
| `test_minicube.py` | 26 | **mask polarity**, transpose fidelity, NaN cleaning, 5-channel test cubes |
| `test_masking.py` | 24 | fill policies, per-channel mean, empty-frame fallback |
| `test_normalization.py` | 26 | moment recovery, mask-excluded statistics, invertibility |
| `test_splits.py` | 29 | partition stability under added cubes, cross-process stability |
| `test_dataset.py` | 48 | sample contract, temporal correctness, caching, corrupt-file tolerance |
| `test_loaders.py` | 30 | collation, seeded shuffle, factory root resolution |
| `test_notebooks.py` | 3 | the notebook actually executes, outputs cleared |

### Three bugs the tests caught

1. **Empty validation split.** With 6 cubes and `val_fraction=0.1`, hash partitioning
   assigned zero cubes to validation, and the error message blamed an incomplete download.
   Now distinguished, with an actionable message; the synthetic config uses `0.25`.
2. **Stale synthetic cubes.** Lowering `n_cubes` left the old files on disk, so the dataset
   silently kept reading them. Regeneration now clears its own cubes first.
3. **Windows short-path lexing** (carried over from Phase 1) — `~` in `ARPITG~1` breaks
   Hydra's override grammar.

## Interpretation

A green suite means the pipeline is wired correctly and is reproducible: windows do not
leak future frames, masks have the right polarity, splits are disjoint and stable, and two
loaders with the same seed produce identical batches.

It says **nothing** about forecasting quality — there is no model, and the default data is
synthetic.

## Decisions worth revisiting later

1. **Auxiliary data is not loaded.** `highresstatic` (elevation) and `mesodynamic`
   (weather) are read by `minicube.py` but not returned by the dataset, because TinyEarth's
   architecture is encoder → temporal → decoder over imagery alone. EarthNet2021 is
   fundamentally a *guided* prediction task, so if Phase 4 finds imagery-only forecasting
   too weak a baseline, weather conditioning is the first thing to add.
2. **`nan_is_invalid=True` deviates from the official toolkit**, which only zero-fills.
   Set it to `False` when reproducing published EarthNetScore numbers.
3. **The format constants are verified against documentation, not against real data.**
   The channel layout, mask polarity and shapes come from the EarthNet2021 paper and the
   `earthnet` toolkit source. They are isolated in `minicube.py` and covered by tests, but
   the first real download should be checked against
   `tests/test_minicube.py::TestMaskPolarity`.
4. **No on-disk index cache.** Scanning ~23000 files at startup may prove slow enough to
   warrant caching the discovered index. Measure before adding it.
5. **`min_valid_fraction` defaults to 0.1 for real data.** Untuned. The right value depends
   on the cloud statistics of the actual download.

## Next: Phase 3

Baselines — ConvLSTM and a temporal transformer, the training loop, L1 loss, and the
forecast-quality (MAE, RMSE, SSIM, PSNR, SAM) and efficiency (parameters, FLOPs, peak
memory, throughput, latency) metrics. The sample contract above is what they consume.

Phase 3 must not begin until this page's verification block reproduces.
