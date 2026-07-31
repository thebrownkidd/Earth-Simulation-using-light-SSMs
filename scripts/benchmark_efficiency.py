#!/usr/bin/env python
"""Benchmark every backbone at every size tier.

Produces the **cost** half of the study: parameters, FLOPs, latency and
throughput for each architecture at matched parameter budgets. No training and
no dataset required, so this runs anywhere in a few minutes.

The quality half needs the EarthNet2021 download and a GPU; see the README.

Results are written as JSON to ``experiments/results/`` and plotted by
``scripts/plot_results.py``.

Usage:
    python scripts/benchmark_efficiency.py
    python scripts/benchmark_efficiency.py --image-size 128 --iterations 10
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import torch

from tinyearth.evaluation.efficiency import profile_model
from tinyearth.models.decoders import CNNDecoder
from tinyearth.models.encoders import CNNEncoder
from tinyearth.models.forecaster import Forecaster
from tinyearth.models.sizes import (
    CALIBRATION_LAYERS,
    TARGET_PARAMETERS,
    available_sizes,
    resolve_hidden_dim,
)
from tinyearth.models.temporal import TEMPORAL_BACKBONES
from tinyearth.utils.device import describe_device, resolve_device
from tinyearth.utils.logging import get_logger, setup_logging
from tinyearth.utils.paths import project_root
from tinyearth.utils.seed import seed_everything

logger = get_logger("tinyearth.scripts.benchmark")

# Backbone-specific arguments held fixed while width varies with the tier.
FIXED_KWARGS: dict[str, dict[str, object]] = {
    "s4d": {"state_dim": 64},
    "mamba": {"state_dim": 16},
    "convlstm": {},
    "transformer": {"n_heads": 4},
}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments.

    Args:
        argv: Argument list. ``None`` uses :data:`sys.argv`.

    Returns:
        The parsed arguments.
    """
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--image-size", type=int, default=64)
    parser.add_argument("--history", type=int, default=4)
    parser.add_argument("--horizon", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--latent-dim", type=int, default=128)
    parser.add_argument("--channels", type=int, default=4)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--iterations", type=int, default=5)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Destination JSON (default: experiments/results/efficiency.json).",
    )
    return parser.parse_args(argv)


def benchmark(args: argparse.Namespace) -> dict[str, Any]:
    """Measure every backbone at every tier.

    Args:
        args: Parsed arguments.

    Returns:
        A record with provenance and one row per (backbone, tier).
    """
    device = resolve_device(args.device)
    # Benchmarks run non-deterministic on purpose: deterministic kernels are not
    # the ones anyone deploys, so timing them understates real throughput.
    seed_everything(args.seed, deterministic=False, cudnn_benchmark=device.type == "cuda")

    sample = torch.rand(
        args.batch_size,
        args.history,
        args.channels,
        args.image_size,
        args.image_size,
        device=device,
    )

    rows: list[dict[str, Any]] = []
    for name in sorted(TEMPORAL_BACKBONES.keys()):
        for tier in available_sizes():
            width = resolve_hidden_dim(name, tier)
            backbone = TEMPORAL_BACKBONES.build(
                name,
                latent_dim=args.latent_dim,
                hidden_dim=width,
                n_layers=CALIBRATION_LAYERS,
                **FIXED_KWARGS.get(name, {}),
            )
            model = Forecaster(
                encoder=CNNEncoder(args.channels, args.latent_dim, base_channels=32, depth=2),
                backbone=backbone,
                decoder=CNNDecoder(args.channels, args.latent_dim, base_channels=32, depth=2),
                horizon=args.horizon,
            ).to(device)

            breakdown = model.parameter_breakdown()
            report = profile_model(model, sample, warmup=args.warmup, iterations=args.iterations)

            rows.append(
                {
                    "backbone": name,
                    "tier": tier,
                    "target_parameters": TARGET_PARAMETERS[tier],
                    "hidden_dim": width,
                    "parameters": breakdown.total,
                    "parameters_backbone": breakdown.backbone,
                    "backbone_fraction": breakdown.backbone_fraction,
                    "gflops_per_sample": (
                        report.flops_per_sample / 1e9 if report.flops_per_sample else None
                    ),
                    "latency_ms": report.latency_ms,
                    "throughput_samples_per_s": report.throughput_samples_per_s,
                    "peak_memory_mb": report.peak_memory_mb,
                }
            )
            logger.info(
                "%-12s %-6s H=%-4d %10s params  %6.1f GFLOPs  %8.1f ms",
                name,
                tier,
                width,
                f"{breakdown.total:,}",
                rows[-1]["gflops_per_sample"] or 0.0,
                report.latency_ms,
            )

    info = describe_device(device)
    return {
        "generated": datetime.now(UTC).isoformat(timespec="seconds"),
        "setup": {
            "image_size": args.image_size,
            "history": args.history,
            "horizon": args.horizon,
            "batch_size": args.batch_size,
            "latent_dim": args.latent_dim,
            "n_layers": CALIBRATION_LAYERS,
            "warmup": args.warmup,
            "iterations": args.iterations,
        },
        "hardware": {
            "device": info.device,
            "device_name": info.device_name,
            "torch": info.torch_version,
            "platform": platform.platform(),
        },
        "rows": rows,
    }


def main(argv: list[str] | None = None) -> int:
    """Entry point.

    Args:
        argv: Argument list. ``None`` uses :data:`sys.argv`.

    Returns:
        A process exit code.
    """
    args = parse_args(argv)
    setup_logging(rich=True, force=True)

    record = benchmark(args)
    destination = args.output or (project_root() / "experiments" / "results" / "efficiency.json")
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(record, indent=2), encoding="utf-8")

    logger.info("wrote %s (%d rows)", destination, len(record["rows"]))
    logger.info("plot with:  python scripts/plot_results.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
