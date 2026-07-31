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

import time
from dataclasses import dataclass, field
from pathlib import Path

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
        torch.save(
            {
                "epoch": epoch,
                "global_step": self.global_step,
                "model": self.model.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "scheduler": self.scheduler.state_dict() if self.scheduler else None,
                "metrics": metrics,
                "config": self.resolved_config,
                "rng": {"python": rng.python_state, "torch": rng.torch_state},
            },
            path,
        )
        return path

    def _is_improvement(self, value: float, best: float) -> bool:
        """Return whether ``value`` beats ``best`` under the configured mode."""
        if self.cfg.checkpoint.mode == "max":
            return value > best
        return value < best

    # -- orchestration ------------------------------------------------------

    def fit(self) -> TrainingResult:
        """Run the full training loop.

        Returns:
            The training result, including the efficiency profile.
        """
        result = TrainingResult(parameters=self.model.parameter_breakdown().as_dict())
        monitor = self.cfg.checkpoint.monitor
        best = float("-inf") if self.cfg.checkpoint.mode == "max" else float("inf")
        epochs_without_improvement = 0

        self.tracker.log_hyperparameters(
            {key: str(value) for key, value in self.resolved_config.items()}
        )

        for epoch in range(self.cfg.epochs):
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
