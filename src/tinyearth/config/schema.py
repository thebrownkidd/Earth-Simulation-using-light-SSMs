"""Hydra structured-config schemas.

These dataclasses are registered with Hydra's :class:`~hydra.core.config_store.ConfigStore`
so that YAML configs are **validated and type-coerced at composition time**. A
typo in a config key, or a string where an int belongs, fails before any model
is constructed rather than three hours into a sweep.

Scope: Phases 1 and 2 own ``run``, ``seed``, ``paths``, ``logging`` and
``data``, which are fully typed here. The ``model`` and ``training`` groups are
typed ``Any`` on purpose -- their schemas are defined by Phases 3 and 4, and
inventing fields now would only guarantee a rewrite. They compose normally in
the meantime, just without static validation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

__all__ = [
    "DataConfig",
    "LoaderConfig",
    "LoggingConfig",
    "NormalizationConfig",
    "PathsConfig",
    "RunConfig",
    "SeedConfig",
    "SyntheticConfig",
    "TensorBoardConfig",
    "TinyEarthConfig",
    "WandbConfig",
]


@dataclass
class RunConfig:
    """Identity and provenance of a single experiment run.

    Attributes:
        name: Short human-readable run name; becomes part of the output path.
        group: Grouping label for related runs, e.g. ``"scaling"`` or
            ``"history_length"``. Used to collate sweep results.
        notes: Free-text description of what the run is testing.
        tags: Labels forwarded to experiment trackers.
        device: Device specification passed to
            :func:`tinyearth.utils.device.resolve_device`.
    """

    name: str = "default"
    group: str = "dev"
    notes: str = ""
    tags: list[str] = field(default_factory=list)
    device: str = "auto"


@dataclass
class SeedConfig:
    """Determinism settings.

    Attributes:
        value: Base RNG seed.
        deterministic: Request deterministic kernels. Costs throughput; leave on
            for correctness and ablation runs, turn off for efficiency benchmarks.
        cudnn_benchmark: Enable cuDNN autotuning. Ignored when ``deterministic``
            is ``True``.
    """

    value: int = 42
    deterministic: bool = True
    cudnn_benchmark: bool = False


@dataclass
class PathsConfig:
    """Filesystem locations.

    Values may be absolute, or relative to the repository root. Resolution is
    performed by :func:`tinyearth.config.resolution.resolve_paths`, never by
    string concatenation at the point of use.

    Attributes:
        data: Dataset root. Git-ignored; populated by the Phase 2 download scripts.
        outputs: Root for run artefacts (checkpoints, configs, metrics).
        cache: Root for derived artefacts that are expensive but reproducible.
    """

    data: str = "data"
    outputs: str = "outputs"
    cache: str = ".cache"


@dataclass
class TensorBoardConfig:
    """TensorBoard tracking settings.

    Attributes:
        enabled: Whether to write TensorBoard event files.
        flush_secs: Seconds between flushes to disk.
    """

    enabled: bool = True
    flush_secs: int = 30


@dataclass
class WandbConfig:
    """Weights & Biases tracking settings.

    W&B is optional throughout the project; every result must be obtainable
    without it. Install via the ``wandb`` extra.

    Attributes:
        enabled: Whether to log to W&B.
        project: W&B project name.
        entity: W&B entity (user or team). ``None`` uses the local default.
        mode: One of ``"online"``, ``"offline"`` or ``"disabled"``.
    """

    enabled: bool = False
    project: str = "tinyearth"
    entity: str | None = None
    mode: str = "online"


@dataclass
class LoggingConfig:
    """Diagnostics and experiment-tracking configuration.

    Attributes:
        level: Console log level name, e.g. ``"INFO"`` or ``"DEBUG"``.
        rich: Use colourised console output. Disable in CI.
        log_file: Log filename written inside the run directory. ``None``
            disables file logging.
        tensorboard: TensorBoard settings.
        wandb: Weights & Biases settings.
    """

    level: str = "INFO"
    rich: bool = True
    log_file: str | None = "run.log"
    tensorboard: TensorBoardConfig = field(default_factory=TensorBoardConfig)
    wandb: WandbConfig = field(default_factory=WandbConfig)


@dataclass
class NormalizationConfig:
    """Imagery normalisation.

    Attributes:
        kind: ``"identity"`` (keep reflectance in ``[0, 1]``) or
            ``"standardize"`` (zero mean, unit variance per channel).
        statistics_path: JSON file of per-channel statistics, required for
            ``"standardize"``. Relative paths resolve against the cache
            directory. Generate with ``scripts/compute_dataset_statistics.py``.
    """

    kind: str = "identity"
    statistics_path: str | None = None


@dataclass
class LoaderConfig:
    """DataLoader settings.

    Attributes:
        batch_size: Samples per batch.
        num_workers: Worker processes. ``0`` loads in the main process.
        shuffle: Shuffle order. Forced off for evaluation splits.
        drop_last: Drop a trailing partial batch.
        pin_memory: Page-lock host memory; only useful with CUDA.
        persistent_workers: Keep workers alive between epochs.
        prefetch_factor: Batches prefetched per worker.
    """

    batch_size: int = 8
    num_workers: int = 0
    shuffle: bool = True
    drop_last: bool = False
    pin_memory: bool = False
    persistent_workers: bool = False
    prefetch_factor: int | None = None


@dataclass
class SyntheticConfig:
    """Synthetic-data generation, used only by the ``synthetic`` dataset.

    Attributes:
        n_cubes: Cubes generated per split.
        n_frames: Timesteps per cube.
        size: Spatial extent of the square cubes.
        cloud_fraction: Approximate share of cloudy pixels.
        nan_fraction: Share of NaN pixels, exercising the cleaning path.
        seed: Generation seed.
    """

    n_cubes: int = 6
    n_frames: int = 30
    size: int = 16
    cloud_fraction: float = 0.2
    nan_fraction: float = 0.01
    seed: int = 0


@dataclass
class DataConfig:
    """Dataset and dataloader configuration.

    Attributes:
        name: Registry key -- ``"earthnet2021"`` or ``"synthetic"``.
        root: Dataset root. Relative paths resolve against ``paths.data``, or
            against ``paths.cache`` for the synthetic dataset, since generated
            cubes are derived artefacts rather than source data.
        split: Split to read. See :class:`~tinyearth.datasets.splits.Split`.
        history_length: Context frames fed to the model.
        horizon: Frames to forecast.
        stride: Step between sliding windows. Ignored for test splits.
        channels: Imagery channels to keep, from ``(blue, green, red, nir)``.
            Empty keeps all four.
        cloud_masking: Emit validity masks alongside imagery.
        mask_policy: How invalid pixels are filled: ``keep``, ``zero``, ``mean``.
        min_valid_fraction: Reject windows whose target is cloudier than this.
        val_fraction: Share of training cubes held out for validation.
        split_salt: Salt for the deterministic train/val partition.
        expected_frames: Frames assumed per cube when building the index.
        context_frames: Context length for anchored (test) windowing.
        cache_size: Decoded cubes cached per worker. ``0`` disables.
        nan_is_invalid: Treat NaN imagery pixels as invalid.
        max_cubes: Cap on cubes read, for smoke tests. ``None`` reads all.
        normalization: Imagery normalisation.
        loader: DataLoader settings.
        synthetic: Generation parameters for the synthetic dataset.
    """

    name: str = "synthetic"
    root: str = "earthnet2021"
    split: str = "train"
    history_length: int = 4
    horizon: int = 1
    stride: int = 1
    channels: list[int] = field(default_factory=list)
    cloud_masking: bool = True
    mask_policy: str = "zero"
    min_valid_fraction: float = 0.0
    val_fraction: float = 0.1
    split_salt: str = "tinyearth"
    expected_frames: int = 30
    context_frames: int = 10
    cache_size: int = 4
    nan_is_invalid: bool = True
    max_cubes: int | None = None
    normalization: NormalizationConfig = field(default_factory=NormalizationConfig)
    loader: LoaderConfig = field(default_factory=LoaderConfig)
    synthetic: SyntheticConfig = field(default_factory=SyntheticConfig)


@dataclass
class TinyEarthConfig:
    """Root configuration object for every TinyEarth entry point.

    Attributes:
        run: Run identity and device selection.
        seed: Determinism settings.
        paths: Filesystem locations.
        logging: Diagnostics and tracking.
        data: Dataset configuration. ``None`` when a command needs no data.
        model: Model configuration. Typed in Phases 3-4.
        training: Optimisation configuration. Typed in Phase 3.
    """

    run: RunConfig = field(default_factory=RunConfig)
    seed: SeedConfig = field(default_factory=SeedConfig)
    paths: PathsConfig = field(default_factory=PathsConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)

    # Optional so that commands which need no data (e.g. `tinyearth-info`)
    # compose without a data group.
    data: DataConfig | None = None

    # Phase 3-4 placeholders. See the module docstring for why these are `Any`.
    model: Any = None
    training: Any = None
