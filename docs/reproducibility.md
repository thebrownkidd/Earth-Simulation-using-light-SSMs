# Reproducibility

TinyEarth's results are efficiency claims. An efficiency claim that cannot be reproduced
on the stated hardware is not a result, so seeding and provenance are part of the
infrastructure rather than an afterthought.

## Two distinct levels

Seeding and determinism are often conflated. They are different, and they have different
costs.

### Seeding

`seed_everything(seed)` seeds `random`, `numpy` and `torch` (CPU and all CUDA devices),
and sets `PYTHONHASHSEED`. It makes a run repeatable given identical software, hardware
and kernel selection. It is free, and always applied.

### Deterministic algorithms

`seed_everything(seed, deterministic=True)` additionally:

- calls `torch.use_deterministic_algorithms(True, warn_only=True)`
- sets `torch.backends.cudnn.deterministic = True`
- disables cuDNN autotuning
- sets `CUBLAS_WORKSPACE_CONFIG=:4096:8`, required for deterministic cuBLAS GEMMs

This makes results bit-reproducible across runs on the same device, at a **measurable
throughput cost**.

`warn_only=True` is deliberate: an operation with no deterministic implementation emits a
warning rather than raising. A hard failure mid-sweep over a non-critical op costs more
than the lost determinism.

## Which setting for which run

Because throughput and latency are primary results in this project, the setting matters:

| Run type | `deterministic` | `cudnn_benchmark` | Why |
| --- | --- | --- | --- |
| Correctness, ablations, unit tests | `true` | `false` | Bit-reproducible; small speed cost is irrelevant |
| Efficiency benchmarks | `false` | `true` | Deterministic kernels are not the ones a user would deploy; benchmarking them understates real throughput |

`cudnn_benchmark` is ignored when `deterministic` is `true`, so the combination cannot be
set inconsistently.

**Report which was used.** A latency number from a deterministic run and one from a
benchmarked run are not comparable.

## Beyond seeding

Bit-identical results are not guaranteed across:

- different GPU architectures, or GPU vs. CPU
- different PyTorch, CUDA or cuDNN versions
- different `num_workers` counts, if data loading consumes RNG
- reduction order in multi-GPU training

This is why every run records its hardware. `tinyearth.utils.device.describe_device()`
captures device name, torch/CUDA/cuDNN versions, total memory, platform and Python
version, and the record is logged at run start.

## DataLoader determinism

Two things are needed, and both are easy to forget:

```python
from tinyearth.utils.seed import seeded_generator, worker_init_fn

DataLoader(
    dataset,
    shuffle=True,
    num_workers=4,
    generator=seeded_generator(cfg.seed.value),  # shuffle order
    worker_init_fn=worker_init_fn,               # per-worker augmentation RNG
)
```

`generator=` matters because global-RNG shuffling makes epoch order depend on how many
random numbers the *model* happened to draw — change a dropout layer and the data order
changes with it.

`worker_init_fn` matters because PyTorch seeds each worker's `torch` RNG but leaves
`random` and `numpy` identical across workers. Without it, four workers apply the *same*
augmentations, silently reducing effective augmentation diversity by 4x. This is a common
and near-invisible bug.

## Scoped RNG control

```python
from tinyearth.utils.seed import temporary_seed, capture_rng_state, restore_rng_state

with temporary_seed(0):
    batch = next(iter(fixed_val_loader))   # same batch every epoch
# training RNG stream is untouched here

state = capture_rng_state()   # for checkpoint save/resume
restore_rng_state(state)
```

`temporary_seed` is how evaluation is made independent of the training RNG stream:
without it, evaluating at a different frequency changes the training trajectory.

## The run record

Each run writes to `outputs/<group>/<name>/`:

| Artefact | Purpose |
| --- | --- |
| `resolved_config.yaml` | Fully resolved config, interpolations expanded — the input side |
| `run.log` | Full DEBUG log, including device record and seed |

Plus a config fingerprint logged at start, hashing everything except `run.*`:

```
TinyEarth v0.1.0 | run=smoke/phase1
config fingerprint: a361b176
device: cpu (AMD64 Family 23 Model 104) | seed: 1234 | deterministic: True
```

To reproduce a run, use its `resolved_config.yaml`. Matching fingerprints mean two runs
had identical substantive configuration.

## Checklist for a reported result

- [ ] `resolved_config.yaml` archived with the result
- [ ] Hardware record from `tinyearth-info` attached
- [ ] Determinism setting stated
- [ ] Seed stated, and ideally several seeds run
- [ ] Efficiency numbers taken with `deterministic=false`, and said so
