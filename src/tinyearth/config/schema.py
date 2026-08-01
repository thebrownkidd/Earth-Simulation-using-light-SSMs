"""Hydra structured-config schemas.

These dataclasses are registered with Hydra's :class:`~hydra.core.config_store.ConfigStore`
so that YAML configs are **validated and type-coerced at composition time**. A
typo in a config key, or a string where an int belongs, fails before any model
is constructed rather than three hours into a sweep.

Scope: every group is now fully typed. ``model.backbone.kwargs`` is the one
deliberate escape hatch -- it holds backbone-specific arguments, which cannot be
enumerated in advance without the schema having to change every time a new
backbone is added. That would defeat the point of the registry.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

__all__ = [
    "BackboneConfig",
    "CheckpointConfig",
    "DataConfig",
    "DecoderConfig",
    "EncoderConfig",
    "EvaluationConfig",
    "LoaderConfig",
    "LoggingConfig",
    "LossConfig",
    "ModelConfig",
    "NormalizationConfig",
    "OptimizerConfig",
    "PathsConfig",
    "RunConfig",
    "SchedulerConfig",
    "SeedConfig",
    "SyntheticConfig",
    "TensorBoardConfig",
    "TinyEarthConfig",
    "TrainingConfig",
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
        crop_size: Side of a square spatial crop, or ``None`` for whole scenes.
            Training crops at a random origin and so also augments; other splits
            crop centred. The dominant cost lever on CPU -- a step at 128 px
            costs roughly 16x one at 32 px -- and it acts only on the encoder
            and decoder, which are held fixed across the architectures compared.
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
    crop_size: int | None = None
    normalization: NormalizationConfig = field(default_factory=NormalizationConfig)
    loader: LoaderConfig = field(default_factory=LoaderConfig)
    synthetic: SyntheticConfig = field(default_factory=SyntheticConfig)


@dataclass
class EncoderConfig:
    """Image encoder configuration.

    Held fixed across temporal-backbone experiments -- changing it invalidates
    comparisons against previously reported backbone results.

    Attributes:
        name: Registry key, e.g. ``"cnn"``.
        base_channels: Channels after the stem, doubled per downsampling stage.
        depth: Number of stride-2 stages; spatial reduction is ``2 ** depth``.
        norm: ``group``, ``batch``, ``instance`` or ``none``.
        activation: ``relu``, ``gelu``, ``silu`` or ``leaky_relu``.
    """

    name: str = "cnn"
    base_channels: int = 32
    depth: int = 2
    norm: str = "group"
    activation: str = "gelu"


@dataclass
class DecoderConfig:
    """Decoder configuration.

    Attributes:
        name: Registry key, e.g. ``"cnn"``.
        base_channels: Channel count at full resolution.
        depth: Number of upsampling stages; must equal the encoder's depth.
        norm: Normalisation kind.
        activation: Hidden activation.
        output_activation: ``sigmoid`` for reflectance in ``[0, 1]``, or
            ``none`` for unbounded output. Must be ``none`` when
            ``data.normalization.kind`` is ``standardize``.
    """

    name: str = "cnn"
    base_channels: int = 32
    depth: int = 2
    norm: str = "group"
    activation: str = "gelu"
    output_activation: str = "sigmoid"


@dataclass
class BackboneConfig:
    """Temporal backbone configuration -- the component under study.

    Attributes:
        name: Registry key from
            :data:`~tinyearth.models.temporal.TEMPORAL_BACKBONES`.
        size: Optional calibrated size tier -- ``tiny``, ``small``, ``base`` or
            ``large`` -- which sets ``hidden_dim`` to the width that puts this
            backbone at the tier's parameter budget. This is what makes the
            scaling study a single override across architectures. An explicit
            ``kwargs.hidden_dim`` wins over it.
        kwargs: Backbone-specific arguments, e.g. ``hidden_dim``, ``n_layers``.
            Deliberately untyped: enumerating them would force a schema change
            for every new backbone, defeating the registry. Unknown keys still
            fail loudly at construction.
    """

    name: str = "convlstm"
    size: str | None = None
    kwargs: dict[str, Any] = field(default_factory=dict)


@dataclass
class LossConfig:
    """Training objective.

    Attributes:
        terms: Registry key to weight, e.g. ``{l1: 1.0}``. Multiple entries form
            a weighted sum; each term is logged separately.
        kwargs: Per-loss constructor arguments, keyed by the same names.
    """

    terms: dict[str, float] = field(default_factory=lambda: {"l1": 1.0})
    kwargs: dict[str, dict[str, Any]] = field(default_factory=dict)


@dataclass
class ModelConfig:
    """Full model configuration.

    Attributes:
        latent_dim: Latent channel count shared by all three components.
        encoder: Encoder settings.
        backbone: Temporal backbone settings.
        decoder: Decoder settings.
        loss: Training objective.
        skip_connections: Wire the encoder's last-observed-context-frame
            features into the decoder at every resolution level. Off by
            default, so every existing (v1) config and checkpoint is
            untouched; v2 experiment configs turn it on explicitly. Changes
            the encoder/decoder parameter count, so the size tiers calibrated
            with it on differ from :data:`tinyearth.models.sizes.SIZE_TIERS`
            -- see :data:`tinyearth.models.sizes.SIZE_TIERS_SKIP`.
        architecture_version: Free-text tag identifying this model's
            architecture, recorded into the resolved config, every
            checkpoint and ``summary.json``. Exists so that a run's numbers
            can always be traced to the code that produced them, and so
            ``--resume`` can refuse to load a checkpoint into structurally
            different code (e.g. a v1 checkpoint has no skip-fusion
            convolutions for v2 code to load weights into).
    """

    latent_dim: int = 128
    encoder: EncoderConfig = field(default_factory=EncoderConfig)
    backbone: BackboneConfig = field(default_factory=BackboneConfig)
    decoder: DecoderConfig = field(default_factory=DecoderConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    skip_connections: bool = False
    architecture_version: str = "v1_baseline"


@dataclass
class OptimizerConfig:
    """Optimiser settings.

    Attributes:
        name: ``adamw``, ``adam`` or ``sgd``.
        lr: Learning rate.
        weight_decay: Decoupled weight decay for AdamW.
        betas: Adam beta coefficients.
        momentum: SGD momentum; ignored by Adam variants.
    """

    name: str = "adamw"
    lr: float = 1e-3
    weight_decay: float = 1e-4
    betas: tuple[float, float] = (0.9, 0.999)
    momentum: float = 0.9


@dataclass
class SchedulerConfig:
    """Learning-rate schedule.

    Attributes:
        name: ``cosine``, ``step``, ``plateau`` or ``none``.
        warmup_epochs: Linear warmup length. Non-zero matters for the
            transformer baseline, which is unstable without it.
        min_lr: Floor for the cosine schedule.
        step_size: Epochs between decays, for ``step``.
        gamma: Decay factor for ``step``.
        patience: Epochs without improvement before decay, for ``plateau``.
    """

    name: str = "cosine"
    warmup_epochs: int = 0
    min_lr: float = 1e-6
    step_size: int = 10
    gamma: float = 0.1
    patience: int = 5


@dataclass
class CheckpointConfig:
    """Checkpointing behaviour.

    Attributes:
        enabled: Write checkpoints at all.
        save_last: Keep the most recent epoch, for resuming.
        save_best: Keep the best epoch by :attr:`monitor`.
        monitor: Validation metric to select on.
        mode: ``min`` or ``max`` for that metric.
    """

    enabled: bool = True
    save_last: bool = True
    save_best: bool = True
    monitor: str = "val/loss"
    mode: str = "min"


@dataclass
class EvaluationConfig:
    """What is measured, and when.

    Attributes:
        every_n_epochs: Validation frequency.
        efficiency: Profile parameters, FLOPs, memory, latency and throughput at
            the end of every run. On by default -- efficiency is this project's
            primary result, so it should never depend on remembering a flag.
        efficiency_warmup: Untimed passes before timing.
        efficiency_iterations: Timed passes.
        data_range: Peak signal value for PSNR and SSIM.
    """

    every_n_epochs: int = 1
    efficiency: bool = True
    efficiency_warmup: int = 3
    efficiency_iterations: int = 10
    data_range: float = 1.0


@dataclass
class TrainingConfig:
    """Optimisation and the training loop.

    Attributes:
        epochs: Number of training epochs.
        optimizer: Optimiser settings.
        scheduler: Learning-rate schedule.
        checkpoint: Checkpointing behaviour.
        evaluation: Metric settings.
        grad_clip: Gradient-norm clip; ``0`` disables.
        amp: Automatic mixed precision. Ignored on non-CUDA devices.
        log_every_n_steps: Training-metric logging frequency.
        max_steps_per_epoch: Cap on steps per epoch, for smoke tests. ``None``
            runs the full epoch.
        early_stopping_patience: Epochs without improvement before stopping.
            ``None`` disables.
    """

    epochs: int = 10
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    scheduler: SchedulerConfig = field(default_factory=SchedulerConfig)
    checkpoint: CheckpointConfig = field(default_factory=CheckpointConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    grad_clip: float = 1.0
    amp: bool = False
    log_every_n_steps: int = 10
    max_steps_per_epoch: int | None = None
    early_stopping_patience: int | None = None


@dataclass
class TinyEarthConfig:
    """Root configuration object for every TinyEarth entry point.

    Attributes:
        run: Run identity and device selection.
        seed: Determinism settings.
        paths: Filesystem locations.
        logging: Diagnostics and tracking.
        data: Dataset configuration.
        model: Model configuration.
        training: Optimisation configuration.

    The three component groups are optional so that commands needing only a
    subset -- ``tinyearth-info`` needs none of them -- compose without them.
    """

    run: RunConfig = field(default_factory=RunConfig)
    seed: SeedConfig = field(default_factory=SeedConfig)
    paths: PathsConfig = field(default_factory=PathsConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    data: DataConfig | None = None
    model: ModelConfig | None = None
    training: TrainingConfig | None = None
