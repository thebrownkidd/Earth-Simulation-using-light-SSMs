"""Experiment metric tracking.

Separate from :mod:`tinyearth.utils.logging`, which handles human-readable
diagnostics. The split means a tracking backend can be swapped without touching
diagnostics, and library code can log freely without importing a tracker.

Every backend implements :class:`MetricTracker`, and :class:`MultiTracker` fans
out to several. Weights & Biases is optional throughout the project: **every
result must be obtainable without it**, so a missing ``wandb`` install degrades
to a warning rather than an error.

All backends also write ``metrics.json`` alongside the run, so results are
readable without launching any viewer -- which is what makes a result table
scriptable across a sweep.
"""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

from tinyearth.config.schema import LoggingConfig
from tinyearth.utils.logging import get_logger

__all__ = [
    "JsonTracker",
    "MetricTracker",
    "MultiTracker",
    "NullTracker",
    "TensorBoardTracker",
    "build_tracker",
]

logger = get_logger(__name__)


class MetricTracker(ABC):
    """Records scalar metrics over the course of a run.

    Only :meth:`log_scalars` is abstract. :meth:`log_hyperparameters` and
    :meth:`close` are optional hooks with working no-op defaults, so a backend
    that has nothing to flush -- or nowhere to record hyperparameters -- does not
    have to implement them. They are deliberately non-abstract rather than
    forgotten.
    """

    @abstractmethod
    def log_scalars(self, metrics: dict[str, float], step: int) -> None:
        """Record a group of scalars.

        Args:
            metrics: Metric name to value.
            step: Global step or epoch index.
        """

    def log_hyperparameters(self, params: dict[str, Any]) -> None:  # noqa: B027
        """Record run hyperparameters. Optional hook; does nothing by default.

        Args:
            params: Flat mapping of hyperparameters.
        """

    def close(self) -> None:  # noqa: B027
        """Flush and release resources. Optional hook; does nothing by default."""


class NullTracker(MetricTracker):
    """Discards everything. Used when tracking is disabled."""

    def log_scalars(self, metrics: dict[str, float], step: int) -> None:
        """Discard the metrics."""


class JsonTracker(MetricTracker):
    """Appends metrics to a JSON Lines file, plus a final summary.

    Always enabled. A run's numbers should be readable with ``json.loads`` and
    no viewer, so that collating a sweep is a script rather than a UI task.

    Args:
        path: Destination ``.jsonl`` file. Parents are created.
    """

    def __init__(self, path: Path | str) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._handle = self.path.open("w", encoding="utf-8")
        self._latest: dict[str, float] = {}

    def log_scalars(self, metrics: dict[str, float], step: int) -> None:
        """Append one record."""
        record = {"step": step, **{key: float(value) for key, value in metrics.items()}}
        self._handle.write(json.dumps(record) + "\n")
        self._handle.flush()
        self._latest.update(record)

    def log_hyperparameters(self, params: dict[str, Any]) -> None:
        """Write hyperparameters as the first record."""
        self._handle.write(json.dumps({"hyperparameters": params}) + "\n")
        self._handle.flush()

    def close(self) -> None:
        """Write a summary file and close the handle."""
        if self._handle.closed:
            return
        self._handle.close()
        summary = self.path.with_name("metrics.json")
        summary.write_text(json.dumps(self._latest, indent=2), encoding="utf-8")


class TensorBoardTracker(MetricTracker):
    """Writes TensorBoard event files.

    Args:
        log_dir: Directory for event files.
        flush_secs: Seconds between flushes.
    """

    def __init__(self, log_dir: Path | str, flush_secs: int = 30) -> None:
        from torch.utils.tensorboard import SummaryWriter

        self.writer = SummaryWriter(log_dir=str(log_dir), flush_secs=flush_secs)

    def log_scalars(self, metrics: dict[str, float], step: int) -> None:
        """Write each scalar."""
        for key, value in metrics.items():
            self.writer.add_scalar(key, float(value), step)

    def log_hyperparameters(self, params: dict[str, Any]) -> None:
        """Write hyperparameters as text.

        Uses ``add_text`` rather than ``add_hparams``: the latter creates a
        separate run directory, which fragments the run's own logs.
        """
        rendered = "\n".join(f"- **{key}**: {value}" for key, value in sorted(params.items()))
        self.writer.add_text("hyperparameters", rendered, 0)

    def close(self) -> None:
        """Flush and close the writer."""
        self.writer.flush()
        self.writer.close()


class WandbTracker(MetricTracker):
    """Logs to Weights & Biases.

    Optional throughout the project. Install with the ``wandb`` extra.

    Args:
        project: W&B project name.
        entity: W&B entity, or ``None`` for the local default.
        run_name: Display name for the run.
        mode: ``online``, ``offline`` or ``disabled``.
        config: Configuration recorded with the run.
    """

    def __init__(
        self,
        project: str,
        entity: str | None,
        run_name: str,
        mode: str,
        config: dict[str, Any] | None = None,
    ) -> None:
        import wandb

        self._wandb = wandb
        self.run = wandb.init(
            project=project, entity=entity, name=run_name, mode=mode, config=config or {}
        )

    def log_scalars(self, metrics: dict[str, float], step: int) -> None:
        """Log the metrics."""
        self._wandb.log(dict(metrics), step=step)

    def log_hyperparameters(self, params: dict[str, Any]) -> None:
        """Merge hyperparameters into the run config."""
        self._wandb.config.update(params, allow_val_change=True)

    def close(self) -> None:
        """Finish the run."""
        self._wandb.finish()


class MultiTracker(MetricTracker):
    """Fans out to several trackers.

    Args:
        trackers: Backends to forward to.
    """

    def __init__(self, trackers: list[MetricTracker]) -> None:
        self.trackers = trackers

    def log_scalars(self, metrics: dict[str, float], step: int) -> None:
        """Forward to every backend."""
        for tracker in self.trackers:
            tracker.log_scalars(metrics, step)

    def log_hyperparameters(self, params: dict[str, Any]) -> None:
        """Forward to every backend."""
        for tracker in self.trackers:
            tracker.log_hyperparameters(params)

    def close(self) -> None:
        """Close every backend, tolerating individual failures.

        A tracker failing to close must not lose the results held by the others.
        """
        for tracker in self.trackers:
            try:
                tracker.close()
            except Exception as error:
                logger.warning("failed to close %s: %s", type(tracker).__name__, error)


def build_tracker(
    cfg: LoggingConfig,
    run_dir: Path,
    run_name: str,
    config: dict[str, Any] | None = None,
) -> MetricTracker:
    """Construct the configured trackers.

    Args:
        cfg: Logging configuration.
        run_dir: Directory for run artefacts.
        run_name: Display name for the run.
        config: Configuration to record with the run.

    Returns:
        A tracker; a :class:`MultiTracker` when several are enabled.
    """
    trackers: list[MetricTracker] = [JsonTracker(run_dir / "metrics.jsonl")]

    if cfg.tensorboard.enabled:
        try:
            trackers.append(TensorBoardTracker(run_dir / "tensorboard", cfg.tensorboard.flush_secs))
        except ImportError as error:
            logger.warning("TensorBoard unavailable (%s); skipping", error)

    if cfg.wandb.enabled:
        try:
            trackers.append(
                WandbTracker(
                    project=cfg.wandb.project,
                    entity=cfg.wandb.entity,
                    run_name=run_name,
                    mode=cfg.wandb.mode,
                    config=config,
                )
            )
        except ImportError:
            logger.warning(
                "wandb is enabled but not installed; continuing without it. "
                'Install with: pip install -e ".[wandb]"'
            )

    return MultiTracker(trackers)
