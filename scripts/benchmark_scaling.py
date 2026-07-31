#!/usr/bin/env python
"""Measure how cost scales along the axes the architectures actually differ on.

Two sweeps, both dataset-free and reproducible on CPU:

``sequence``
    Cost against history length ``T``. This is the **direct empirical test of
    the asymptotic claim**: attention is ``O(T²)`` in the temporal mixing while
    an SSM is ``O(T log T)`` (FFT convolution) or ``O(T)`` (recurrence). Fitting
    a power law to measured cost gives an exponent per architecture, which is a
    quantitative answer rather than an appeal to the complexity class.

``mixing``
    The same, but measuring the **temporal backbone in isolation** over a much
    wider ``T`` range. This is the sweep that actually answers the question: in
    the full model the encoder and decoder do ``O(T)`` work that swamps the
    temporal term entirely, so a whole-model measurement cannot see the
    difference between ``O(T)`` and ``O(T²)`` mixing. Isolating the component
    under study is the whole point of the architecture, and here it is the only
    way to get a clean reading.

``state``
    Cost against SSM state size ``N``, which has no counterpart in the ConvLSTM
    or transformer. Tests whether ``state_dim`` really is the cheap capacity
    axis the design assumes -- parameters should grow ``~4HN``, i.e. linearly
    and with a small constant, while width grows quadratically.

Usage:
    python scripts/benchmark_scaling.py --sweep sequence
    python scripts/benchmark_scaling.py --sweep mixing
    python scripts/benchmark_scaling.py --sweep state
"""

from __future__ import annotations

import argparse
import json
import math
import platform
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import torch

from tinyearth.evaluation.efficiency import measure_flops, measure_latency
from tinyearth.models.decoders import CNNDecoder
from tinyearth.models.encoders import CNNEncoder
from tinyearth.models.forecaster import Forecaster, count_parameters
from tinyearth.models.sizes import CALIBRATION_LAYERS, resolve_hidden_dim
from tinyearth.models.temporal import TEMPORAL_BACKBONES
from tinyearth.utils.device import describe_device, resolve_device
from tinyearth.utils.logging import get_logger, setup_logging
from tinyearth.utils.paths import project_root
from tinyearth.utils.seed import seed_everything

logger = get_logger("tinyearth.scripts.scaling")

FIXED_KWARGS: dict[str, dict[str, object]] = {
    "s4d": {"state_dim": 64},
    "mamba": {"state_dim": 16},
    "convlstm": {},
    "transformer": {"n_heads": 4},
}
SEQUENCE_LENGTHS = (2, 4, 8, 16, 32)
MIXING_LENGTHS = (8, 16, 32, 64, 128, 256, 512)
STATE_DIMS = (8, 16, 32, 64, 128, 256)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments.

    Args:
        argv: Argument list. ``None`` uses :data:`sys.argv`.

    Returns:
        The parsed arguments.
    """
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--sweep", choices=["sequence", "mixing", "state"], required=True)
    parser.add_argument("--tier", default="tiny", help="Size tier held fixed (default: tiny).")
    parser.add_argument("--image-size", type=int, default=32)
    parser.add_argument("--horizon", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--latent-dim", type=int, default=128)
    parser.add_argument("--channels", type=int, default=4)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--iterations", type=int, default=5)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args(argv)


def _build(args: argparse.Namespace, backbone: str, **overrides: object) -> Forecaster:
    """Assemble a forecaster with the standard fixed encoder and decoder."""
    kwargs: dict[str, Any] = dict(FIXED_KWARGS.get(backbone, {}))
    kwargs.update(overrides)
    kwargs.setdefault("hidden_dim", resolve_hidden_dim(backbone, args.tier))

    module = TEMPORAL_BACKBONES.build(
        backbone, latent_dim=args.latent_dim, n_layers=CALIBRATION_LAYERS, **kwargs
    )
    return Forecaster(
        encoder=CNNEncoder(args.channels, args.latent_dim, base_channels=32, depth=2),
        backbone=module,
        decoder=CNNDecoder(args.channels, args.latent_dim, base_channels=32, depth=2),
        horizon=args.horizon,
    )


def fit_exponent(xs: list[float], ys: list[float]) -> float | None:
    """Fit ``y = c * x**k`` by least squares in log space and return ``k``.

    The exponent is the quantity of interest: ``k ≈ 1`` is linear scaling,
    ``k ≈ 2`` is quadratic. Fitting measured cost is a far more honest answer
    than quoting a complexity class, because constant factors and the
    architecture's non-temporal work both dilute the asymptotic term at the
    sizes anyone actually runs.

    Args:
        xs: Independent variable, all positive.
        ys: Measured cost, all positive.

    Returns:
        The fitted exponent, or ``None`` with fewer than two usable points.
    """
    points = [(math.log(x), math.log(y)) for x, y in zip(xs, ys, strict=True) if x > 0 and y > 0]
    if len(points) < 2:
        return None

    n = len(points)
    mean_x = sum(x for x, _ in points) / n
    mean_y = sum(y for _, y in points) / n
    covariance = sum((x - mean_x) * (y - mean_y) for x, y in points)
    variance = sum((x - mean_x) ** 2 for x, _ in points)
    return covariance / variance if variance else None


def sweep_sequence(args: argparse.Namespace, device: torch.device) -> list[dict[str, Any]]:
    """Measure cost against history length for every backbone."""
    rows: list[dict[str, Any]] = []
    for backbone in sorted(TEMPORAL_BACKBONES.keys()):
        model = _build(args, backbone).to(device)
        for length in SEQUENCE_LENGTHS:
            sample = torch.rand(
                args.batch_size,
                length,
                args.channels,
                args.image_size,
                args.image_size,
                device=device,
            )
            flops = measure_flops(model, sample)
            latency = measure_latency(model, sample, warmup=args.warmup, iterations=args.iterations)
            rows.append(
                {
                    "backbone": backbone,
                    "history": length,
                    "gflops_per_sample": flops / 1e9 if flops else None,
                    "latency_ms": latency,
                }
            )
            logger.info(
                "%-12s T=%-3d %8.2f GFLOPs %9.1f ms",
                backbone,
                length,
                rows[-1]["gflops_per_sample"] or 0.0,
                latency,
            )
    return rows


def sweep_mixing(args: argparse.Namespace, device: torch.device) -> list[dict[str, Any]]:
    """Measure the temporal backbone alone against sequence length.

    The encoder and decoder are excluded deliberately. Their cost is linear in
    ``T`` and dominates the whole-model measurement, so including them hides
    exactly the difference this sweep exists to expose.

    Args:
        args: Parsed arguments.
        device: Device to measure on.

    Returns:
        One row per (backbone, sequence length).
    """
    rows: list[dict[str, Any]] = []
    grid = 8  # small latent grid: this sweep is about T, not spatial extent

    for backbone in sorted(TEMPORAL_BACKBONES.keys()):
        kwargs: dict[str, Any] = dict(FIXED_KWARGS.get(backbone, {}))
        kwargs["hidden_dim"] = 128
        module = TEMPORAL_BACKBONES.build(
            backbone, latent_dim=args.latent_dim, n_layers=2, **kwargs
        ).to(device)

        for length in MIXING_LENGTHS:
            latents = torch.rand(1, length, args.latent_dim, grid, grid, device=device)

            class _Wrapped(torch.nn.Module):
                """Pins the horizon so the module presents a single-argument call."""

                def __init__(self, inner: torch.nn.Module, horizon: int) -> None:
                    super().__init__()
                    self.inner = inner
                    self.horizon = horizon

                def forward(self, x: torch.Tensor) -> torch.Tensor:
                    result: torch.Tensor = self.inner(x, self.horizon)
                    return result

            wrapped = _Wrapped(module, 1)
            flops = measure_flops(wrapped, latents)
            latency = measure_latency(wrapped, latents, warmup=1, iterations=3)

            rows.append(
                {
                    "backbone": backbone,
                    "history": length,
                    "gflops_per_sample": flops / 1e9 if flops else None,
                    "latency_ms": latency,
                }
            )
            logger.info(
                "%-12s T=%-4d %8.3f GFLOPs %9.1f ms",
                backbone,
                length,
                rows[-1]["gflops_per_sample"] or 0.0,
                latency,
            )
    return rows


def sweep_state(args: argparse.Namespace, device: torch.device) -> list[dict[str, Any]]:
    """Measure cost against SSM state size, plus the width axis for comparison."""
    rows: list[dict[str, Any]] = []
    sample = torch.rand(
        args.batch_size, 4, args.channels, args.image_size, args.image_size, device=device
    )

    for backbone in ("s4d", "mamba"):
        width = resolve_hidden_dim(backbone, args.tier)
        for state_dim in STATE_DIMS:
            model = _build(args, backbone, state_dim=state_dim).to(device)
            latency = measure_latency(model, sample, warmup=args.warmup, iterations=args.iterations)
            rows.append(
                {
                    "backbone": backbone,
                    "axis": "state_dim",
                    "value": state_dim,
                    "hidden_dim": width,
                    "parameters": count_parameters(model),
                    "latency_ms": latency,
                }
            )
            logger.info(
                "%-12s state_dim=%-4d %11s params %8.1f ms",
                backbone,
                state_dim,
                f"{rows[-1]['parameters']:,}",
                latency,
            )

    # The comparison axis: width, at the same backbone, for the same model.
    for backbone in ("s4d", "mamba"):
        for width in (128, 176, 256, 368, 512, 736):
            model = _build(args, backbone, hidden_dim=width).to(device)
            latency = measure_latency(model, sample, warmup=args.warmup, iterations=args.iterations)
            rows.append(
                {
                    "backbone": backbone,
                    "axis": "hidden_dim",
                    "value": width,
                    "hidden_dim": width,
                    "parameters": count_parameters(model),
                    "latency_ms": latency,
                }
            )
            logger.info(
                "%-12s hidden_dim=%-4d %11s params %8.1f ms",
                backbone,
                width,
                f"{rows[-1]['parameters']:,}",
                latency,
            )
    return rows


def main(argv: list[str] | None = None) -> int:
    """Entry point.

    Args:
        argv: Argument list. ``None`` uses :data:`sys.argv`.

    Returns:
        A process exit code.
    """
    args = parse_args(argv)
    setup_logging(rich=True, force=True)

    device = resolve_device(args.device)
    seed_everything(args.seed, deterministic=False, cudnn_benchmark=device.type == "cuda")

    sweeps = {"sequence": sweep_sequence, "mixing": sweep_mixing, "state": sweep_state}
    rows = sweeps[args.sweep](args, device)

    # Fitted exponents: the headline quantity for the length sweeps.
    exponents: dict[str, dict[str, float | None]] = {}
    if args.sweep in {"sequence", "mixing"}:
        for backbone in sorted({row["backbone"] for row in rows}):
            picked = [row for row in rows if row["backbone"] == backbone]
            lengths = [float(row["history"]) for row in picked]
            exponents[backbone] = {
                "latency": fit_exponent(lengths, [row["latency_ms"] for row in picked]),
                "gflops": fit_exponent(
                    lengths,
                    [row["gflops_per_sample"] or 0.0 for row in picked],
                ),
            }
            logger.info(
                "%-12s fitted exponent  latency k=%.2f  flops k=%.2f",
                backbone,
                exponents[backbone]["latency"] or float("nan"),
                exponents[backbone]["gflops"] or float("nan"),
            )

    info = describe_device(device)
    record = {
        "generated": datetime.now(UTC).isoformat(timespec="seconds"),
        "sweep": args.sweep,
        "setup": {
            "tier": args.tier,
            "image_size": args.image_size,
            "horizon": args.horizon,
            "batch_size": args.batch_size,
            "latent_dim": args.latent_dim,
            "n_layers": CALIBRATION_LAYERS,
            "iterations": args.iterations,
        },
        "hardware": {
            "device": info.device,
            "device_name": info.device_name,
            "torch": info.torch_version,
            "platform": platform.platform(),
        },
        "exponents": exponents,
        "rows": rows,
    }

    destination = args.output or (
        project_root() / "experiments" / "results" / f"scaling_{args.sweep}.json"
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(record, indent=2), encoding="utf-8")
    logger.info("wrote %s (%d rows)", destination, len(rows))
    return 0


if __name__ == "__main__":
    sys.exit(main())
