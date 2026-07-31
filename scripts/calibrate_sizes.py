#!/usr/bin/env python
"""Recalibrate the model size tiers.

The scaling study compares backbones at matched parameter budgets, so the width
for each ``(backbone, tier)`` pair is measured rather than guessed. This script
produces the table in :mod:`tinyearth.models.sizes`.

Rerun it whenever a backbone's block structure changes, or when adding a new
backbone -- ``tests/test_sizes.py`` fails if the stored table drifts from what
the models actually build.

Every tier fixes ``n_layers`` and varies only ``hidden_dim``. Letting depth and
width move together would confound the scaling result.

Usage:
    python scripts/calibrate_sizes.py
    python scripts/calibrate_sizes.py --layers 6 --emit-python
"""

from __future__ import annotations

import argparse
import sys

from tinyearth.models.decoders import CNNDecoder
from tinyearth.models.encoders import CNNEncoder
from tinyearth.models.forecaster import count_parameters
from tinyearth.models.sizes import TARGET_PARAMETERS
from tinyearth.models.temporal import TEMPORAL_BACKBONES
from tinyearth.utils.logging import get_logger, setup_logging

logger = get_logger("tinyearth.scripts.calibrate")

# Backbone-specific arguments held fixed while width is swept.
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
    parser.add_argument("--latent-dim", type=int, default=128)
    parser.add_argument("--layers", type=int, default=4)
    parser.add_argument("--channels", type=int, default=4)
    parser.add_argument("--base-channels", type=int, default=32)
    parser.add_argument("--depth", type=int, default=2)
    parser.add_argument("--step", type=int, default=16, help="hidden_dim search granularity.")
    parser.add_argument("--max-width", type=int, default=1600)
    parser.add_argument(
        "--emit-python",
        action="store_true",
        help="Print a SIZE_TIERS literal ready to paste into models/sizes.py.",
    )
    return parser.parse_args(argv)


def fixed_parameters(args: argparse.Namespace) -> int:
    """Count the encoder and decoder parameters, which are held fixed.

    Args:
        args: Parsed arguments.

    Returns:
        Combined encoder + decoder parameter count.
    """
    encoder = CNNEncoder(
        args.channels, args.latent_dim, base_channels=args.base_channels, depth=args.depth
    )
    decoder = CNNDecoder(
        args.channels, args.latent_dim, base_channels=args.base_channels, depth=args.depth
    )
    return count_parameters(encoder) + count_parameters(decoder)


def calibrate(args: argparse.Namespace) -> dict[str, dict[str, int]]:
    """Find the width that puts each backbone closest to each target.

    Args:
        args: Parsed arguments.

    Returns:
        ``{backbone: {tier: hidden_dim}}``.
    """
    overhead = fixed_parameters(args)
    logger.info("fixed encoder+decoder: %s parameters", f"{overhead:,}")

    table: dict[str, dict[str, int]] = {}
    for name in sorted(TEMPORAL_BACKBONES.keys()):
        cls = TEMPORAL_BACKBONES.get(name)
        kwargs = dict(FIXED_KWARGS.get(name, {}))
        widths: list[tuple[int, int]] = []

        for width in range(args.step, args.max_width + 1, args.step):
            try:
                backbone = cls(  # type: ignore[call-arg]
                    latent_dim=args.latent_dim,
                    hidden_dim=width,
                    n_layers=args.layers,
                    **kwargs,
                )
            except (ValueError, RuntimeError):
                continue  # e.g. width not divisible by n_heads
            widths.append((width, count_parameters(backbone) + overhead))

        tiers: dict[str, int] = {}
        for tier, target in TARGET_PARAMETERS.items():
            width, total = min(widths, key=lambda pair: abs(pair[1] - target))
            tiers[tier] = width
            logger.info(
                "%-12s %-6s hidden_dim=%-5d %10s params (%+.1f%%)",
                name,
                tier,
                width,
                f"{total:,}",
                100 * (total - target) / target,
            )
        table[name] = tiers
    return table


def main(argv: list[str] | None = None) -> int:
    """Entry point.

    Args:
        argv: Argument list. ``None`` uses :data:`sys.argv`.

    Returns:
        A process exit code.
    """
    args = parse_args(argv)
    setup_logging(rich=True, force=True)
    table = calibrate(args)

    if args.emit_python:
        print("\nSIZE_TIERS: Final[dict[str, dict[str, int]]] = {")
        for name, tiers in table.items():
            entries = ", ".join(f'"{tier}": {width}' for tier, width in tiers.items())
            print(f'    "{name}": {{{entries}}},')
        print("}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
