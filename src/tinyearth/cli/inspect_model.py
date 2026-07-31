"""``tinyearth-model``: report a model's size and cost without training it.

Answers "how big is this, and what does a forward pass cost?" in a second,
which is what makes tuning a backbone to a parameter budget practical. Phase 4's
size tiers (tiny ~2M, small ~5M, base ~10M, large ~20M) are reached by sweeping
``model.backbone.kwargs.hidden_dim`` against this command.

Example:
    ```bash
    tinyearth-model                                        # the default backbone
    tinyearth-model model=transformer                      # the other baseline
    tinyearth-model --compare                              # every backbone, side by side
    tinyearth-model model.backbone.kwargs.hidden_dim=256
    ```
"""

from __future__ import annotations

import sys

import torch
from omegaconf import DictConfig

from tinyearth.bootstrap import RunContext, initialise_run
from tinyearth.cli._hydra import run_with_hydra
from tinyearth.config.resolution import to_dataclass
from tinyearth.evaluation.efficiency import profile_model
from tinyearth.models.factory import build_forecaster
from tinyearth.models.temporal import TEMPORAL_BACKBONES
from tinyearth.utils.logging import log_section

__all__ = ["main"]

_COMPARE_FLAG = "--compare"


def _sample_batch(context: RunContext, cfg: object) -> torch.Tensor:
    """Build a representative input batch from the data config.

    Args:
        context: The run context, for the device.
        cfg: The composed data config.

    Returns:
        A random batch shaped like real input, on the run device.
    """
    data = cfg
    batch = int(getattr(getattr(data, "loader", None), "batch_size", 4))
    history = int(getattr(data, "history_length", 4))
    channels = len(getattr(data, "channels", []) or []) or 4
    size = int(getattr(getattr(data, "synthetic", None), "size", 64))
    return torch.rand(batch, history, channels, size, size, device=context.device)


def report_model(context: RunContext, compare: bool) -> None:
    """Build and profile the configured model, or every backbone.

    Args:
        context: An initialised run context.
        compare: Profile every registered backbone rather than just the
            configured one.

    Raises:
        ValueError: If the composed config lacks a ``model`` or ``data`` group.
    """
    logger = context.logger
    cfg = to_dataclass(context.cfg)
    if cfg.model is None or cfg.data is None:
        raise ValueError("Both a `model` and a `data` group are required.")

    sample = _sample_batch(context, cfg.data)
    channels = int(sample.shape[2])
    names = list(TEMPORAL_BACKBONES.keys()) if compare else [cfg.model.backbone.name]

    logger.info("input %s on %s", tuple(sample.shape), context.device)

    rows: list[tuple[str, int, float, float, float | None]] = []
    for name in names:
        model_cfg = cfg.model
        model_cfg.backbone.name = name
        if compare and name != cfg.model.backbone.name:
            # Other backbones take different kwargs; fall back to their defaults
            # rather than forwarding arguments they do not accept.
            model_cfg.backbone.kwargs = {}

        model = build_forecaster(model_cfg, in_channels=channels, horizon=cfg.data.horizon)
        model = model.to(context.device)
        breakdown = model.parameter_breakdown()

        log_section(logger, f"Backbone: {name}")
        for line in breakdown.format_table().splitlines():
            logger.info("  %s", line)

        report = profile_model(model, sample, warmup=2, iterations=5)
        for line in report.format_table().splitlines():
            logger.info("  %s", line)

        rows.append(
            (
                name,
                breakdown.total,
                breakdown.backbone_fraction,
                report.latency_ms,
                report.flops_per_sample,
            )
        )

    if len(rows) > 1:
        log_section(logger, "Comparison")
        logger.info(
            "  %-14s %>12s %>10s %>12s %>12s".replace(">", ""),
            "backbone",
            "params",
            "backbone%",
            "latency ms",
            "GFLOPs",
        )
        for name, total, fraction, latency, flops in rows:
            logger.info(
                "  %-14s %12,d %9.1f%% %12.2f %12s",
                name,
                total,
                100 * fraction,
                latency,
                f"{flops / 1e9:.3f}" if flops else "n/a",
            )


def main() -> None:
    """Console-script entry point for ``tinyearth-model``.

    Strips the non-Hydra ``--compare`` flag before handing over.
    """
    compare = _COMPARE_FLAG in sys.argv
    if compare:
        sys.argv = [arg for arg in sys.argv if arg != _COMPARE_FLAG]

    def entrypoint(cfg: DictConfig) -> None:
        """Initialise the run and report on the model."""
        context = initialise_run(cfg, create_dirs=False)
        report_model(context, compare)

    run_with_hydra(entrypoint)


if __name__ == "__main__":
    main()
