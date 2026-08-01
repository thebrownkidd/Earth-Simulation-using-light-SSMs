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
    tinyearth-train +experiment=earthnet model=s4d --resume outputs/earthnet/s4d/last.ckpt
    ```
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

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

_RESUME_FLAG = "--resume"


def _extract_resume_path(argv: list[str]) -> tuple[list[str], Path | None]:
    """Strip a non-Hydra ``--resume <path>`` pair out of an argument list.

    Follows the same pattern as ``--dry-run`` in ``cli/inspect_config.py``:
    Hydra's override grammar has no notion of a bare flag taking a value, so
    anything not in ``key=value`` form must be consumed before Hydra sees it.

    Args:
        argv: Raw argument list, as from :data:`sys.argv`.

    Returns:
        ``(remaining_argv, path)``; ``path`` is ``None`` if ``--resume`` was
        not present.

    Raises:
        SystemExit: If ``--resume`` is the last argument, with nothing after it.
    """
    if _RESUME_FLAG not in argv:
        return argv, None

    index = argv.index(_RESUME_FLAG)
    if index + 1 >= len(argv):
        raise SystemExit(
            f"{_RESUME_FLAG} requires a checkpoint path, e.g. {_RESUME_FLAG} last.ckpt"
        )

    remaining = argv[:index] + argv[index + 2 :]
    return remaining, Path(argv[index + 1])


def run_training(context: RunContext, *, resume_path: Path | None = None) -> TrainingResult:
    """Build everything and train.

    Args:
        context: An initialised run context.
        resume_path: A checkpoint to resume from. When given, training
            continues from one epoch past the checkpoint's recorded epoch,
            with model, optimiser, scheduler and RNG state restored --
            rather than starting fresh at epoch 0.

    Returns:
        The training result.

    Raises:
        ValueError: If the composed config lacks a ``data``, ``model`` or
            ``training`` group, or if ``resume_path``'s checkpoint was
            trained under a different ``model.architecture_version``.
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
        architecture_version=model_cfg.architecture_version,
    )

    start_epoch = 0
    if resume_path is not None:
        logger.info("resuming from %s", resume_path)
        start_epoch = trainer.load_checkpoint(
            resume_path, architecture_version=model_cfg.architecture_version
        )
        logger.info("resuming at epoch %d/%d", start_epoch + 1, training_cfg.epochs)

    try:
        result = trainer.fit(start_epoch=start_epoch)
    finally:
        tracker.close()

    _report(context, result, architecture_version=model_cfg.architecture_version)
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


def _report(context: RunContext, result: TrainingResult, *, architecture_version: str) -> None:
    """Log the final summary and write it to the run directory.

    Args:
        context: The run context.
        result: The completed training result.
        architecture_version: Tag identifying this model's architecture,
            written alongside the numeric summary so a run's numbers can
            always be traced back to the code that produced them. Kept
            outside ``result.summary()`` itself, which every other caller of
            :meth:`~tinyearth.training.trainer.TrainingResult.summary`
            expects to be all-numeric.
    """
    logger = context.logger
    log_section(logger, "Results")

    if result.epochs:
        final = result.epochs[-1]
        for key, value in sorted({**final.train, **final.validation}.items()):
            logger.info("  %-22s %.6f", key, value)
    if result.best_epoch >= 0:
        logger.info("  %-22s %.6f (epoch %d)", "best", result.best_metric, result.best_epoch + 1)

    summary: dict[str, object] = {"architecture_version": architecture_version, **result.summary()}
    destination = context.paths.run_dir / "summary.json"
    destination.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    logger.info("wrote %s", destination)


def main() -> None:
    """Console-script entry point for ``tinyearth-train``.

    The non-Hydra ``--resume <path>`` flag is stripped from :data:`sys.argv`
    before handing over, since Hydra's parser only understands ``key=value``
    overrides -- the same pattern ``cli/inspect_config.py`` uses for
    ``--dry-run``.
    """
    sys.argv, resume_path = _extract_resume_path(sys.argv)

    def entrypoint(cfg: DictConfig) -> None:
        """Initialise the run and train."""
        context = initialise_run(cfg)
        run_training(context, resume_path=resume_path)
        context.logger.info("training complete")

    run_with_hydra(entrypoint)


if __name__ == "__main__":
    main()
