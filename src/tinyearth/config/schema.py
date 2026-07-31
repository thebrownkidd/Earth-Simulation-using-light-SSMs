"""Hydra structured-config schemas.

These dataclasses are registered with Hydra's :class:`~hydra.core.config_store.ConfigStore`
so that YAML configs are **validated and type-coerced at composition time**. A
typo in a config key, or a string where an int belongs, fails before any model
is constructed rather than three hours into a sweep.

Scope: Phase 1 owns ``run``, ``seed``, ``paths`` and ``logging``, which are
fully typed here. The ``data``, ``model`` and ``training`` groups are typed
``Any`` on purpose -- their schemas are defined by Phases 2, 3 and 4, and
inventing fields now would only guarantee a rewrite. They compose normally in
the meantime, just without static validation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

__all__ = [
    "LoggingConfig",
    "PathsConfig",
    "RunConfig",
    "SeedConfig",
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
class TinyEarthConfig:
    """Root configuration object for every TinyEarth entry point.

    Attributes:
        run: Run identity and device selection.
        seed: Determinism settings.
        paths: Filesystem locations.
        logging: Diagnostics and tracking.
        data: Dataset configuration. Typed in Phase 2.
        model: Model configuration. Typed in Phases 3-4.
        training: Optimisation configuration. Typed in Phase 3.
    """

    run: RunConfig = field(default_factory=RunConfig)
    seed: SeedConfig = field(default_factory=SeedConfig)
    paths: PathsConfig = field(default_factory=PathsConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)

    # Phase 2-4 placeholders. See the module docstring for why these are `Any`.
    data: Any = None
    model: Any = None
    training: Any = None
