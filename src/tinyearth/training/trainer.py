"""The training loop.

Lives here rather than in a script so it can be tested without spawning a
subprocess, which is the difference between a training loop that is verified and
one that is merely run.

Design notes
------------
**Masks propagate into the loss and the metrics.** Optimising or scoring
cloud-contaminated pixels measures cloud prediction; the masks come from the
dataset and are used wherever a reduction happens.

**Efficiency is profiled at the end of every run by default.** Efficiency is
this project's primary result, so it should never depend on remembering a flag.

**Checkpoints record the resolved config, the RNG state and the metrics.** A
checkpoint that cannot reproduce its own run is a liability rather than an
asset.
"""

from __future__ import annotations

import random
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from torch.utils.data import DataLoader

from tinyearth.config.schema import TrainingConfig
from tinyearth.datasets.types import Batch, Sample
from tinyearth.evaluation.efficiency import EfficiencyReport, profile_model
from tinyearth.evaluation.metrics import MetricAccumulator, forecast_metrics
from tinyearth.models.forecaster import Forecaster
from tinyearth.models.losses.base import CompositeLoss, ForecastLoss
from tinyearth.training.optim import current_lr
from tinyearth.training.tracking import MetricTracker, NullTracker
from tinyearth.utils.logging import get_logger, log_section
from tinyearth.utils.seed import capture_rng_state

__all__ = ["EpochResult", "Trainer", "TrainingResult"]

logger = get_logger(__name__)


@dataclass(frozen=True)
class EpochResult:
    """Metrics from one epoch.

    Attributes:
        epoch: Zero-based epoch index.
        train: Training metrics, prefixed ``train/``.
        validation: Validation metrics, prefixed ``val/``. Empty when validation
            did not run this epoch.
        duration_s: Wall-clock seconds.
        learning_rate: Learning rate at the end of the epoch.
    """

    epoch: int
    train: dict[str, float]
    validation: dict[str, float] = field(default_factory=dict)
    duration_s: float = 0.0
    learning_rate: float = 0.0

    def merged(self) -> dict[str, float]:
        """Return all metrics in one mapping, ready for a tracker."""
        return {
            **self.train,
            **self.validation,
            "epoch/duration_s": self.duration_s,
            "epoch/learning_rate": self.learning_rate,
        }


@dataclass
class TrainingResult:
    """Outcome of a complete training run.

    Attributes:
        epochs: Per-epoch results.
        best_metric: Best observed value of the monitored metric.
        best_epoch: Epoch that achieved it.
        parameters: Parameter breakdown, as a flat mapping.
        efficiency: Efficiency profile, when measured.
        stopped_early: Whether early stopping triggered.
    """

    epochs: list[EpochResult] = field(default_factory=list)
    best_metric: float = float("inf")
    best_epoch: int = -1
    parameters: dict[str, float] = field(default_factory=dict)
    efficiency: EfficiencyReport | None = None
    stopped_early: bool = False

    def summary(self) -> dict[str, float]:
        """Flatten the headline numbers for logging.

        Returns:
            Best-metric, parameter and efficiency values in one mapping.
        """
        payload: dict[str, float] = dict(self.parameters)
        payload["best/metric"] = self.best_metric
        payload["best/epoch"] = float(self.best_epoch)
        if self.efficiency is not None:
            payload.update(self.efficiency.as_dict())
        if self.epochs:
            payload.update(self.epochs[-1].validation)
        return payload


class Trainer:
    """Trains a forecaster and reports quality and efficiency.

    Args:
        model: The forecaster to train.
        loss: Training objective.
        optimizer: Optimiser.
        cfg: Training configuration.
        device: Device to train on.
        train_loader: Training data.
        val_loader: Validation data. ``None`` disables validation, along with
            checkpoint selection and early stopping.
        scheduler: Per-step LR schedule, or ``None``.
        tracker: Metric tracker.
        run_dir: Directory for checkpoints.
        resolved_config: Config recorded into checkpoints for reproducibility.
        architecture_version: Tag identifying this model's architecture,
            recorded into every checkpoint. :meth:`load_checkpoint` compares
            it against a checkpoint's own tag when asked to, refusing to
            resume across an architecture change rather than silently loading
            weights into modules the checkpoint was never trained against.
    """

    def __init__(
        self,
        model: Forecaster,
        loss: ForecastLoss,
        optimizer: Optimizer,
        cfg: TrainingConfig,
        device: torch.device,
        train_loader: DataLoader[Sample],
        val_loader: DataLoader[Sample] | None = None,
        scheduler: LRScheduler | None = None,
        tracker: MetricTracker | None = None,
        run_dir: Path | None = None,
        resolved_config: dict[str, object] | None = None,
        architecture_version: str = "v1_baseline",
    ) -> None:
        self.model = model.to(device)
        self.loss = loss.to(device)
        self.optimizer = optimizer
        self.cfg = cfg
        self.device = device
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.scheduler = scheduler
        self.tracker = tracker or NullTracker()
        self.run_dir = run_dir
        self.resolved_config = resolved_config or {}
        self.architecture_version = architecture_version

        self.global_step = 0
        # AMP is CUDA-only in this project; enabling it on CPU would silently
        # do nothing while suggesting a speedup was measured.
        self.amp_enabled = cfg.amp and device.type == "cuda"
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.amp_enabled)

        if cfg.amp and not self.amp_enabled:
            logger.info("amp requested but device is %s; running in full precision", device.type)

    # -- batch handling -----------------------------------------------------

    def _to_device(self, batch: Batch) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        """Move a batch's tensors to the training device.

        Args:
            batch: A collated batch.

        Returns:
            ``(images, target, target_mask)``; the mask is ``None`` when the
            dataset was built without cloud masking.
        """
        images = batch["images"].to(self.device, non_blocking=True)
        target = batch["target"].to(self.device, non_blocking=True)
        mask = batch.get("target_mask")
        if mask is not None:
            mask = mask.to(self.device, non_blocking=True)
        return images, target, mask

    def _loss_terms(self) -> dict[str, float]:
        """Return per-term loss values from the last forward pass."""
        if isinstance(self.loss, CompositeLoss):
            return {f"train/loss_{name}": value for name, value in self.loss.last_terms.items()}
        return {}

    # -- training -----------------------------------------------------------

    def train_epoch(self) -> dict[str, float]:
        """Run one training epoch.

        Takes no epoch index: progress is tracked by :attr:`global_step`, and a
        second, separately-maintained counter would be one more thing to get out
        of sync.

        Returns:
            Training metrics, prefixed ``train/``.
        """
        self.model.train()
        accumulator = MetricAccumulator()
        total_loss = 0.0
        total_weight = 0.0
        max_steps = self.cfg.max_steps_per_epoch

        for step, batch in enumerate(self.train_loader):
            if max_steps is not None and step >= max_steps:
                break

            images, target, mask = self._to_device(batch)
            batch_size = float(images.shape[0])

            self.optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=self.device.type, enabled=self.amp_enabled):
                prediction = self.model(images, horizon=target.shape[1])
                loss_value = self.loss(prediction, target, mask)

            self.scaler.scale(loss_value).backward()
            if self.cfg.grad_clip > 0:
                # Unscale before clipping, or the clip threshold is applied to
                # scaled gradients and means nothing.
                self.scaler.unscale_(self.optimizer)
                nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.grad_clip)
            self.scaler.step(self.optimizer)
            self.scaler.update()

            if self.scheduler is not None:
                self.scheduler.step()

            total_loss += float(loss_value.detach()) * batch_size
            total_weight += batch_size
            self.global_step += 1

            with torch.no_grad():
                accumulator.update(
                    forecast_metrics(
                        prediction.detach().float(),
                        target,
                        mask,
                        self.cfg.evaluation.data_range,
                    ),
                    batch_size,
                )

            if self.global_step % max(self.cfg.log_every_n_steps, 1) == 0:
                self.tracker.log_scalars(
                    {
                        "train/loss_step": float(loss_value.detach()),
                        "train/lr": current_lr(self.optimizer),
                        **self._loss_terms(),
                    },
                    self.global_step,
                )

        metrics = {f"train/{key}": value for key, value in accumulator.compute().items()}
        metrics["train/loss"] = total_loss / total_weight if total_weight else 0.0
        return metrics

    @torch.no_grad()
    def validate(self) -> dict[str, float]:
        """Evaluate on the validation split.

        Returns:
            Validation metrics, prefixed ``val/``. Empty when there is no
            validation loader.
        """
        if self.val_loader is None:
            return {}

        self.model.eval()
        accumulator = MetricAccumulator()
        total_loss = 0.0
        total_weight = 0.0

        for batch in self.val_loader:
            images, target, mask = self._to_device(batch)
            batch_size = float(images.shape[0])

            prediction = self.model(images, horizon=target.shape[1])
            loss_value = self.loss(prediction, target, mask)

            total_loss += float(loss_value) * batch_size
            total_weight += batch_size
            accumulator.update(
                forecast_metrics(prediction.float(), target, mask, self.cfg.evaluation.data_range),
                batch_size,
            )

        metrics = {f"val/{key}": value for key, value in accumulator.compute().items()}
        metrics["val/loss"] = total_loss / total_weight if total_weight else 0.0
        return metrics

    # -- checkpoints --------------------------------------------------------

    def save_checkpoint(self, name: str, epoch: int, metrics: dict[str, float]) -> Path | None:
        """Write a checkpoint.

        Args:
            name: Filename stem, e.g. ``"best"`` or ``"last"``.
            epoch: Epoch index being saved.
            metrics: Metrics at that epoch.

        Returns:
            The path written, or ``None`` when checkpointing is disabled or no
            run directory was given.
        """
        if not self.cfg.checkpoint.enabled or self.run_dir is None:
            return None

        # The CLI creates the run directory, but the Trainer is also used
        # directly from tests and notebooks, where nothing has yet.
        self.run_dir.mkdir(parents=True, exist_ok=True)
        path = self.run_dir / f"{name}.ckpt"
        rng = capture_rng_state()
        # The train loader's own shuffle generator is a separate RNG stream
        # from the ones capture_rng_state() covers -- see loaders.py, where it
        # is seeded explicitly rather than left to the global torch RNG so
        # that shuffle order does not depend on how many random draws the
        # model happened to make. Saving its state is what lets a resumed run
        # continue the SAME shuffle sequence rather than restarting it.
        generator = getattr(self.train_loader, "generator", None)
        torch.save(
            {
                "epoch": epoch,
                "global_step": self.global_step,
                "architecture_version": self.architecture_version,
                "model": self.model.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "scheduler": self.scheduler.state_dict() if self.scheduler else None,
                "metrics": metrics,
                "config": self.resolved_config,
                "rng": {
                    "python": rng.python_state,
                    "numpy": rng.numpy_state,
                    "torch": rng.torch_state,
                    "cuda": rng.cuda_states,
                },
                "loader_generator": generator.get_state() if generator is not None else None,
            },
            path,
        )
        return path

    def load_checkpoint(self, path: Path, *, architecture_version: str | None = None) -> int:
        """Restore training state from a checkpoint and return the epoch to resume from.

        Restores model weights, optimizer state, scheduler state, the train
        loader's shuffle generator (if any) and the global RNG state. Called
        once, after everything else is built and seeded but before
        :meth:`fit` -- the RNG state this restores overwrites whatever
        :func:`~tinyearth.utils.seed.seed_everything` set, which is the point.

        Args:
            path: A checkpoint written by :meth:`save_checkpoint`.
            architecture_version: Expected architecture-version tag. When
                given and it disagrees with the checkpoint's own tag, the load
                is refused rather than silently proceeding -- e.g. resuming a
                v1 checkpoint into v2 code would try to load weights into
                skip-fusion convolutions v1 never had, or leave v2's
                skip-fusion weights at their random initialisation with no
                indication anything was wrong.

        Returns:
            The epoch index to resume from: one past the epoch the checkpoint
            recorded, since that epoch's training already happened.

        Raises:
            ValueError: If ``architecture_version`` is given and does not
                match the checkpoint's recorded tag.
        """
        payload = torch.load(path, map_location=self.device, weights_only=False)

        checkpoint_version = payload.get("architecture_version")
        if architecture_version is not None and checkpoint_version != architecture_version:
            raise ValueError(
                f"Refusing to resume from {path}: it was trained as architecture_version="
                f"{checkpoint_version!r}, but this run is {architecture_version!r}. Their "
                "encoder/decoder/backbone parameters are not guaranteed compatible."
            )

        self.model.load_state_dict(payload["model"])
        self.optimizer.load_state_dict(payload["optimizer"])
        if self.scheduler is not None and payload.get("scheduler") is not None:
            self.scheduler.load_state_dict(payload["scheduler"])
        self.global_step = payload["global_step"]

        rng = payload["rng"]
        random.setstate(rng["python"])
        if rng.get("numpy") is not None:
            np.random.set_state(rng["numpy"])
        else:  # pragma: no cover - only reachable loading a pre-numpy-capture checkpoint
            logger.warning("checkpoint has no numpy RNG state (older format); not restored")
        torch.set_rng_state(rng["torch"])
        if rng.get("cuda") and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(list(rng["cuda"]))

        generator_state = payload.get("loader_generator")
        loader_generator = getattr(self.train_loader, "generator", None)
        if generator_state is not None and loader_generator is not None:
            loader_generator.set_state(generator_state)

        resume_epoch = int(payload["epoch"]) + 1
        logger.info(
            "loaded checkpoint %s (architecture_version=%r, epoch %d, step %d)",
            path,
            checkpoint_version,
            payload["epoch"],
            self.global_step,
        )
        return resume_epoch

    def _is_improvement(self, value: float, best: float) -> bool:
        """Return whether ``value`` beats ``best`` under the configured mode."""
        if self.cfg.checkpoint.mode == "max":
            return value > best
        return value < best

    # -- orchestration ------------------------------------------------------

    def fit(self, start_epoch: int = 0) -> TrainingResult:
        """Run the training loop from ``start_epoch`` to :attr:`cfg`'s epoch count.

        Args:
            start_epoch: Zero-based epoch index to begin at. ``0`` for a fresh
                run; the return value of :meth:`load_checkpoint` when
                resuming, so that the epoch a checkpoint recorded is not
                repeated.

        Returns:
            The training result, including the efficiency profile. Covers
            only epochs actually run in this call -- on resume, the caller
            gets a result for the tail of training, not the whole history.
        """
        result = TrainingResult(parameters=self.model.parameter_breakdown().as_dict())
        monitor = self.cfg.checkpoint.monitor
        best = float("-inf") if self.cfg.checkpoint.mode == "max" else float("inf")
        epochs_without_improvement = 0

        if start_epoch >= self.cfg.epochs:
            logger.info(
                "start_epoch=%d >= training.epochs=%d; nothing left to train",
                start_epoch,
                self.cfg.epochs,
            )

        self.tracker.log_hyperparameters(
            {key: str(value) for key, value in self.resolved_config.items()}
        )

        for epoch in range(start_epoch, self.cfg.epochs):
            started = time.perf_counter()
            train_metrics = self.train_epoch()

            should_validate = (
                self.val_loader is not None
                and (epoch + 1) % max(self.cfg.evaluation.every_n_epochs, 1) == 0
            )
            val_metrics = self.validate() if should_validate else {}

            epoch_result = EpochResult(
                epoch=epoch,
                train=train_metrics,
                validation=val_metrics,
                duration_s=time.perf_counter() - started,
                learning_rate=current_lr(self.optimizer),
            )
            result.epochs.append(epoch_result)
            self.tracker.log_scalars(epoch_result.merged(), epoch)

            logger.info(
                "epoch %d/%d | train loss %.5f%s | %.1fs | lr %.2e",
                epoch + 1,
                self.cfg.epochs,
                train_metrics.get("train/loss", 0.0),
                f" | val loss {val_metrics['val/loss']:.5f}" if val_metrics else "",
                epoch_result.duration_s,
                epoch_result.learning_rate,
            )

            if self.cfg.checkpoint.save_last:
                self.save_checkpoint("last", epoch, epoch_result.merged())

            if monitor in val_metrics:
                value = val_metrics[monitor]
                if self._is_improvement(value, best):
                    best = value
                    result.best_metric = value
                    result.best_epoch = epoch
                    epochs_without_improvement = 0
                    if self.cfg.checkpoint.save_best:
                        self.save_checkpoint("best", epoch, epoch_result.merged())
                else:
                    epochs_without_improvement += 1

                patience = self.cfg.early_stopping_patience
                if patience is not None and epochs_without_improvement >= patience:
                    logger.info(
                        "early stopping: %s has not improved for %d epochs",
                        monitor,
                        epochs_without_improvement,
                    )
                    result.stopped_early = True
                    break

        if self.cfg.evaluation.efficiency:
            result.efficiency = self.profile()

        return result

    def profile(self) -> EfficiencyReport | None:
        """Measure the efficiency metrics on a representative batch.

        Returns:
            The profile, or ``None`` if no batch was available or profiling
            failed. Profiling must never take down a completed training run.
        """
        try:
            batch = next(iter(self.val_loader or self.train_loader))
        except StopIteration:  # pragma: no cover - empty loader
            logger.warning("no batch available for profiling")
            return None

        images = batch["images"].to(self.device)
        report = profile_model(
            self.model,
            images,
            warmup=self.cfg.evaluation.efficiency_warmup,
            iterations=self.cfg.evaluation.efficiency_iterations,
        )
        log_section(logger, "Efficiency")
        for line in report.format_table().splitlines():
            logger.info("  %s", line)
        return report
