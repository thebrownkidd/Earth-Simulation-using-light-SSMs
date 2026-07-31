"""``tinyearth-train``: train a forecaster from a config.

The Phase 3 entry point. It composes a config, builds the data, model, loss and
optimiser, runs the training loop, and reports quality and efficiency metrics.

It **orchestrates only** -- every piece of logic lives in
:mod:`tinyearth.datasets`, :mod:`tinyearth.models` or
:mod:`tinyearth.training`, so the training loop is testable without a
subprocess.

Example:
    ```bash
    tinyearth-train                                       # defaults
    tinyearth-train +experiment=baseline_smoke            # the Phase 3 experiment
    tinyearth-train model=transformer                     # swap the backbone
    tinyearth-train model=convlstm training.epochs=50 data=earthnet2021
    ```
"""

from __future__ import annotations

import json

from omegaconf import DictConfig

from tinyearth.bootstrap import RunContext, initialise_run
from tinyearth.cli._hydra import run_with_hydra
from tinyearth.config.resolution import to_container, to_dataclass
from tinyearth.datasets.factory import build_datamodule
from tinyearth.datasets.splits import Split
from tinyearth.models.factory import build_forecaster, build_loss
from tinyearth.training.optim import build_optimizer, build_scheduler
from tinyearth.training.tracking import build_tracker
from tinyearth.training.trainer import Trainer, TrainingResult
from tinyearth.utils.logging import log_section

__all__ = ["main", "run_training"]


def run_training(context: RunContext) -> TrainingResult:
    """Build everything and train.

    Args:
        context: An initialised run context.

    Returns:
        The training result.

    Raises:
        ValueError: If the composed config lacks a ``data``, ``model`` or
            ``training`` group.
    """
    logger = context.logger
    cfg = to_dataclass(context.cfg)

    missing = [
        name
        for name, value in (
            ("data", cfg.data),
            ("model", cfg.model),
            ("training", cfg.training),
        )
        if value is None
    ]
    if missing:
        raise ValueError(
            f"Config is missing the {', '.join(missing)} group(s). Compose them with "
            "e.g. `data=synthetic model=convlstm training=default`."
        )

    # Rebound to non-optional locals so the rest of the function is statically
    # known to have them; the check above is what guarantees it.
    data_cfg, model_cfg, training_cfg = cfg.data, cfg.model, cfg.training
    if data_cfg is None or model_cfg is None or training_cfg is None:  # pragma: no cover
        raise ValueError("Unreachable: missing config groups were already rejected.")

    # -- data ---------------------------------------------------------------
    log_section(logger, "Data")
    train = build_datamodule(
        data_cfg, context.paths, split=Split.TRAIN, seed=context.seed, device=context.device
    )
    validation = build_datamodule(
        data_cfg, context.paths, split=Split.VAL, seed=context.seed, device=context.device
    )
    logger.info(
        "  train %d windows (%d batches) | val %d windows (%d batches)",
        len(train.dataset),
        len(train.loader),
        len(validation.dataset),
        len(validation.loader),
    )

    # -- model --------------------------------------------------------------
    log_section(logger, "Model")
    model = build_forecaster(
        model_cfg, in_channels=train.dataset.n_channels, horizon=data_cfg.horizon
    )
    loss = build_loss(model_cfg.loss)
    for line in model.parameter_breakdown().format_table().splitlines():
        logger.info("  %s", line)

    _warn_on_activation_mismatch(context, model_cfg.decoder.output_activation, data_cfg)

    # -- optimisation -------------------------------------------------------
    optimizer = build_optimizer(model, training_cfg.optimizer)
    steps_per_epoch = training_cfg.max_steps_per_epoch or len(train.loader)
    scheduler = build_scheduler(
        optimizer,
        training_cfg.scheduler,
        steps_per_epoch=max(steps_per_epoch, 1),
        epochs=training_cfg.epochs,
    )

    resolved = to_container(context.cfg)
    tracker = build_tracker(
        cfg.logging, context.paths.run_dir, f"{cfg.run.group}/{cfg.run.name}", resolved
    )

    # -- train --------------------------------------------------------------
    log_section(logger, "Training")
    trainer = Trainer(
        model=model,
        loss=loss,
        optimizer=optimizer,
        cfg=training_cfg,
        device=context.device,
        train_loader=train.loader,
        val_loader=validation.loader,
        scheduler=scheduler,
        tracker=tracker,
        run_dir=context.paths.run_dir,
        resolved_config=resolved,
    )

    try:
        result = trainer.fit()
    finally:
        tracker.close()

    _report(context, result)
    return result


def _warn_on_activation_mismatch(context: RunContext, output_activation: str, data: object) -> None:
    """Warn when a sigmoid decoder is paired with standardised targets.

    The combination is a silent trap: targets leave ``[0, 1]`` while the model
    cannot, so the loss plateaus for a reason that looks like an optimisation
    failure rather than a config error.

    Args:
        context: The run context, for logging.
        output_activation: The decoder's output activation.
        data: The data config, inspected for its normalisation kind.
    """
    kind = getattr(getattr(data, "normalization", None), "kind", "identity")
    if output_activation == "sigmoid" and kind != "identity":
        context.logger.warning(
            "decoder.output_activation=sigmoid bounds predictions to [0, 1], but "
            "data.normalization.kind=%r produces unbounded targets. Set "
            "model.decoder.output_activation=none.",
            kind,
        )


def _report(context: RunContext, result: TrainingResult) -> None:
    """Log the final summary and write it to the run directory.

    Args:
        context: The run context.
        result: The completed training result.
    """
    logger = context.logger
    log_section(logger, "Results")

    if result.epochs:
        final = result.epochs[-1]
        for key, value in sorted({**final.train, **final.validation}.items()):
            logger.info("  %-22s %.6f", key, value)
    if result.best_epoch >= 0:
        logger.info("  %-22s %.6f (epoch %d)", "best", result.best_metric, result.best_epoch + 1)

    summary = result.summary()
    destination = context.paths.run_dir / "summary.json"
    destination.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    logger.info("wrote %s", destination)


def main() -> None:
    """Console-script entry point for ``tinyearth-train``."""

    def entrypoint(cfg: DictConfig) -> None:
        """Initialise the run and train."""
        context = initialise_run(cfg)
        run_training(context)
        context.logger.info("training complete")

    run_with_hydra(entrypoint)


if __name__ == "__main__":
    main()
