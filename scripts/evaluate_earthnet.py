#!/usr/bin/env python
"""Evaluate the trained EarthNet2021 models against each other and against doing nothing.

Produces ``experiments/results/earthnet.json``: the authoritative quality
numbers for the study, and the data behind the quality figures.

Two things this does that the training loop does not
-----------------------------------------------------
**It scores the parameter-free references.** Persistence and climatology are
evaluated on exactly the same windows, with exactly the same masked metrics.
Without them a learned MAE is uninterpretable, because Earth surface imagery is
mostly static over 100 days and a model that has learned only to echo its input
would still post a respectable number.

**It scores at full 128x128 resolution**, whereas training used 32x32 crops.
The models are fully convolutional in space, so this costs nothing and asks the
harder, more honest question: how do they do on whole scenes? These numbers,
not the crop-sized ones in each run's ``summary.json``, are the ones to report.

Metrics are also broken down by lead time, which is where the interesting
structure is: every forecaster degrades as it predicts further ahead, and *how
fast* separates a model that has learned dynamics from one that has learned an
average.

Usage:
    python scripts/evaluate_earthnet.py
    python scripts/evaluate_earthnet.py --group earthnet --limit 200
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from dataclasses import replace
from pathlib import Path
from typing import Any

import torch

from tinyearth.config.resolution import from_container, resolve_paths
from tinyearth.datasets.factory import build_dataset
from tinyearth.datasets.loaders import LoaderSettings, build_dataloader
from tinyearth.datasets.splits import Split
from tinyearth.evaluation.metrics import forecast_metrics
from tinyearth.evaluation.references import REFERENCE_FORECASTS
from tinyearth.evaluation.visualization import ndvi
from tinyearth.models.factory import build_forecaster
from tinyearth.utils.logging import get_logger, setup_logging
from tinyearth.utils.paths import project_root

logger = get_logger("tinyearth.scripts.evaluate")

BACKBONES = ("s4d", "mamba", "convlstm", "transformer")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--group", default="earthnet", help="Run group under outputs/.")
    parser.add_argument("--outputs", type=Path, default=None, help="Run output root.")
    parser.add_argument("--output", type=Path, default=None, help="Destination JSON.")
    parser.add_argument("--batch-size", type=int, default=4, help="Evaluation batch size.")
    parser.add_argument(
        "--limit", type=int, default=None, help="Evaluate at most this many validation cubes."
    )
    parser.add_argument(
        "--crop",
        type=int,
        default=None,
        help="Evaluate on crops of this size instead of whole 128x128 scenes.",
    )
    return parser.parse_args(argv)


def load_checkpoint(path: Path) -> tuple[Any, Any]:
    """Load a checkpoint and rebuild its model in eval mode."""
    payload = torch.load(path, map_location="cpu", weights_only=False)
    cfg = from_container(payload["config"])
    if cfg.model is None or cfg.data is None:
        raise ValueError(f"{path} carries no model/data config.")

    channels = len(cfg.data.channels) if cfg.data.channels else 4
    model = build_forecaster(cfg.model, in_channels=channels, horizon=cfg.data.horizon)
    model.load_state_dict(payload["model"])
    model.eval()
    return model, cfg


class Tally:
    """Sample-weighted accumulator for one forecaster.

    Batch means cannot simply be averaged: a final partial batch would count as
    heavily as a full one. Every quantity is therefore accumulated as a sum of
    per-sample contributions and divided at the end.
    """

    def __init__(self, horizon: int) -> None:
        """Create an empty tally.

        Args:
            horizon: Forecast length, fixing the per-lead-time array size.
        """
        self.samples = 0
        self.totals: dict[str, float] = {}
        self.per_lead_mae = [0.0] * horizon
        self.per_lead_ndvi = [0.0] * horizon
        self.seconds = 0.0

    def update(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor,
        mask: torch.Tensor,
        elapsed: float,
    ) -> None:
        """Fold one batch into the tally.

        Args:
            prediction: Forecast, ``[B, K, C, H, W]``.
            target: Ground truth, same shape.
            mask: Validity, ``[B, K, 1, H, W]``.
            elapsed: Seconds spent producing the forecast.
        """
        batch = prediction.shape[0]
        self.samples += batch
        self.seconds += elapsed

        for name, value in forecast_metrics(prediction, target, mask).items():
            self.totals[name] = self.totals.get(name, 0.0) + value * batch

        # Per lead time, masked so cloud does not read as model error.
        weights = mask.expand_as(prediction)
        error = (prediction - target).abs() * weights
        observed = weights.sum(dim=(0, 2, 3, 4)).clamp(min=1.0)
        for lead, value in enumerate((error.sum(dim=(0, 2, 3, 4)) / observed).tolist()):
            self.per_lead_mae[lead] += value * batch

        ndvi_error = (ndvi(prediction) - ndvi(target)).abs() * mask[:, :, 0]
        ndvi_observed = mask[:, :, 0].sum(dim=(0, 2, 3)).clamp(min=1.0)
        for lead, value in enumerate((ndvi_error.sum(dim=(0, 2, 3)) / ndvi_observed).tolist()):
            self.per_lead_ndvi[lead] += value * batch

    def result(self) -> dict[str, Any]:
        """Return the averaged metrics."""
        divisor = max(self.samples, 1)
        return {
            "samples": self.samples,
            **{name: total / divisor for name, total in self.totals.items()},
            "mae_by_lead": [value / divisor for value in self.per_lead_mae],
            "ndvi_mae_by_lead": [value / divisor for value in self.per_lead_ndvi],
            "seconds_per_sample": self.seconds / divisor,
        }


def evaluate(
    models: dict[str, Any],
    loader: Any,
    horizon: int,
) -> dict[str, dict[str, Any]]:
    """Score every model and reference over the validation split.

    Args:
        models: Backbone name to model.
        loader: Validation dataloader.
        horizon: Forecast length.

    Returns:
        Forecaster name to its metrics.
    """
    names = [*models, *REFERENCE_FORECASTS]
    tallies = {name: Tally(horizon) for name in names}

    with torch.no_grad():
        for index, batch in enumerate(loader):
            images = batch["images"]
            target = batch["target"]
            target_mask = batch.get("target_mask")
            if target_mask is None:
                target_mask = torch.ones_like(target[:, :, :1])
            images_mask = batch.get("images_mask")

            for name, model in models.items():
                start = time.perf_counter()
                prediction = model(images, horizon=horizon)
                tallies[name].update(prediction, target, target_mask, time.perf_counter() - start)

            for name, reference in REFERENCE_FORECASTS.items():
                start = time.perf_counter()
                prediction = reference(images, horizon, mask=images_mask)
                tallies[name].update(prediction, target, target_mask, time.perf_counter() - start)

            if index % 10 == 0:
                logger.info("  batch %d | %d samples scored", index, tallies[names[0]].samples)

    return {name: tally.result() for name, tally in tallies.items()}


def _training_cost(summary_path: Path) -> dict[str, float]:
    """Read the efficiency block a training run already measured.

    Cost is not recomputed here. The trainer profiles each model with warmup and
    a median over repeats at the end of its run; repeating that inside an
    evaluation pass -- on a machine that may be busy training something else --
    would produce a worse number for no reason.

    Args:
        summary_path: Path to a run's ``summary.json``.

    Returns:
        Latency, throughput and FLOPs, or an empty mapping if unavailable.
    """
    if not summary_path.is_file():
        return {}
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    keys = ("latency_ms", "gflops_per_sample", "throughput_samples_per_s")
    return {key: summary[f"efficiency/{key}"] for key in keys if f"efficiency/{key}" in summary}


def main(argv: list[str] | None = None) -> int:
    """Entry point.

    Args:
        argv: Argument list. ``None`` uses :data:`sys.argv`.

    Returns:
        A process exit code.
    """
    args = parse_args(argv)
    setup_logging(rich=True, force=True)

    root = args.outputs or project_root() / "outputs"
    destination = args.output or project_root() / "experiments" / "results" / "earthnet.json"

    models: dict[str, Any] = {}
    parameters: dict[str, int] = {}
    cost: dict[str, dict[str, float]] = {}
    reference_cfg = None
    for backbone in BACKBONES:
        checkpoint = root / args.group / backbone / "best.ckpt"
        if not checkpoint.is_file():
            logger.warning("no checkpoint for %s at %s", backbone, checkpoint)
            continue
        logger.info("loading %s", backbone)
        model, reference_cfg = load_checkpoint(checkpoint)
        models[backbone] = model
        parameters[backbone] = model.parameter_breakdown().total
        cost[backbone] = _training_cost(checkpoint.parent / "summary.json")

    if not models or reference_cfg is None:
        logger.error("No checkpoints found under %s. Train first.", root / args.group)
        return 1

    # `max_cubes` rather than truncating the window index: it is the supported
    # knob, and at this protocol a cube yields exactly one window anyway.
    data_cfg = replace(reference_cfg.data, crop_size=args.crop, max_cubes=args.limit)
    dataset = build_dataset(data_cfg, resolve_paths(reference_cfg), split=Split.VAL)

    loader = build_dataloader(
        dataset,
        LoaderSettings(batch_size=args.batch_size, num_workers=0, shuffle=False, drop_last=False),
    )
    horizon = data_cfg.horizon
    logger.info(
        "scoring %d windows at %s, horizon %d",
        len(dataset),
        f"{args.crop}x{args.crop} crops" if args.crop else "full 128x128 scenes",
        horizon,
    )

    results = evaluate(models, loader, horizon)

    record = {
        "setup": {
            "group": args.group,
            "history": data_cfg.history_length,
            "horizon": horizon,
            "crop_size": args.crop,
            "split": "val",
            "windows": len(dataset),
            "revisit_days": 5,
        },
        "hardware": {"device": "cpu", "processor": platform.processor()},
        "parameters": parameters,
        "cost": cost,
        "results": results,
    }

    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(record, indent=2), encoding="utf-8")
    logger.info("wrote %s", destination)

    logger.info("--- validation MAE (lower is better) ---")
    ranked = sorted(results.items(), key=lambda item: item[1]["mae"])
    for name, metrics in ranked:
        marker = "  " if name in REFERENCE_FORECASTS else "* "
        logger.info(
            "%s%-14s mae %.5f  rmse %.5f  ssim %.4f  (%s params)",
            marker,
            name,
            metrics["mae"],
            metrics["rmse"],
            metrics["ssim"],
            f"{parameters[name]:,}" if name in parameters else "none",
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
