#!/usr/bin/env python
"""Compute per-channel normalisation statistics.

Statistics are computed over the **training split only**. Including validation
or test data leaks distributional information and inflates every reported
number.

Cloud-masked pixels are excluded. This matters more than it sounds: cloud is
near-white, so counting masked pixels biases every channel mean upward and
inflates the variance, which then propagates into every standardised input.

Usage:
    python scripts/compute_dataset_statistics.py
    python scripts/compute_dataset_statistics.py --dataset earthnet2021 --max-batches 500
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Iterable, Iterator
from pathlib import Path

import torch

from tinyearth.config.resolution import ResolvedPaths
from tinyearth.config.schema import DataConfig, LoaderConfig
from tinyearth.datasets.factory import build_datamodule
from tinyearth.datasets.normalization import compute_channel_statistics
from tinyearth.datasets.splits import Split
from tinyearth.datasets.types import Batch
from tinyearth.utils.logging import get_logger, setup_logging
from tinyearth.utils.paths import cache_dir, data_dir, outputs_dir, project_root
from tinyearth.utils.seed import seed_everything

logger = get_logger("tinyearth.scripts.statistics")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments.

    Args:
        argv: Argument list. ``None`` uses :data:`sys.argv`.

    Returns:
        The parsed arguments.
    """
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--dataset", default="synthetic", choices=["synthetic", "earthnet2021"])
    parser.add_argument("--root", default=None, help="Dataset root (default: config default).")
    parser.add_argument("--output", type=Path, default=None, help="Destination JSON file.")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument(
        "--max-batches",
        type=int,
        default=200,
        help="Cap on batches consumed. Statistics converge quickly; the full "
        "training set is rarely necessary (default: %(default)s).",
    )
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args(argv)


def _batches(
    loader: Iterable[Batch], max_batches: int
) -> Iterator[tuple[torch.Tensor, torch.Tensor | None]]:
    """Yield ``(images, valid)`` pairs from a loader, up to ``max_batches``.

    Both the history and target frames are used, since normalisation applies to
    the whole sequence.

    Args:
        loader: A dataloader over samples.
        max_batches: Maximum batches to consume.

    Yields:
        Imagery and its validity mask.
    """
    for count, batch in enumerate(loader):
        if count >= max_batches:
            return
        images = torch.cat([batch["images"], batch["target"]], dim=1)
        mask = None
        if "images_mask" in batch:
            mask = torch.cat([batch["images_mask"], batch["target_mask"]], dim=1)
        yield images, mask


def main(argv: list[str] | None = None) -> int:
    """Entry point.

    Args:
        argv: Argument list. ``None`` uses :data:`sys.argv`.

    Returns:
        A process exit code.
    """
    args = parse_args(argv)
    setup_logging(rich=True, force=True)
    seed_everything(args.seed, deterministic=False)

    paths = ResolvedPaths(
        root=project_root(),
        data=data_dir(),
        outputs=outputs_dir(),
        cache=cache_dir(),
        run_dir=outputs_dir() / "statistics",
    )
    cfg = DataConfig(
        name=args.dataset,
        root=args.root
        or ("synthetic_earthnet2021" if args.dataset == "synthetic" else "earthnet2021"),
        split=Split.TRAIN.value,
        loader=LoaderConfig(
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            shuffle=True,
            drop_last=False,
        ),
    )

    bundle = build_datamodule(cfg, paths, split=Split.TRAIN, seed=args.seed)
    logger.info(
        "computing statistics over %r: %d windows, up to %d batches",
        args.dataset,
        len(bundle.dataset),
        args.max_batches,
    )

    statistics = compute_channel_statistics(
        _batches(bundle.loader, args.max_batches),
        n_channels=bundle.dataset.n_channels,
    )

    destination = args.output or (paths.cache / f"{args.dataset}_channel_statistics.json")
    statistics.save(destination)
    logger.info("wrote %s", destination)
    logger.info("point data.normalization.statistics_path at it and set kind=standardize")
    return 0


if __name__ == "__main__":
    sys.exit(main())
