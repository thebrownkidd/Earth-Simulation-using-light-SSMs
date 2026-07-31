#!/usr/bin/env python
"""Download EarthNet2021.

TinyEarth does **not** redistribute the dataset. This script wraps the official
``earthnet`` toolkit downloader and, when that is unavailable, prints the manual
instructions rather than failing silently.

The dataset is large. Approximate sizes:

===================  ========  ======================================
Split                Size      Cubes
===================  ========  ======================================
``train``            ~100 GB   ~23000
``iid_test_split``   ~5 GB     ~4000
``ood_test_split``   ~5 GB     ~4000
``extreme_test``     ~7 GB     ~4000
``seasonal_test``    ~15 GB    ~4000
===================  ========  ======================================

If you only want to verify the pipeline, you do not need any of this -- the
synthetic dataset is the default and requires no download::

    tinyearth-data +experiment=data_smoke

Usage:
    python scripts/download_earthnet2021.py --root data/earthnet2021
    python scripts/download_earthnet2021.py --root data/earthnet2021 --splits train iid
    python scripts/download_earthnet2021.py --instructions
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from tinyearth.utils.logging import get_logger, setup_logging
from tinyearth.utils.paths import project_root

logger = get_logger("tinyearth.scripts.download")

SPLIT_CHOICES = ("train", "iid", "ood", "extreme", "seasonal")

INSTRUCTIONS = """
EarthNet2021 download
=====================

TinyEarth does not redistribute this dataset. Obtain it from the authors.

Option 1 -- the official toolkit (recommended)

    pip install -e ".[data]"
    python scripts/download_earthnet2021.py --root data/earthnet2021

Option 2 -- the toolkit directly

    pip install earthnet
    python -c "import earthnet as en; en.Downloader.get('data/earthnet2021', 'all')"

Option 3 -- by hand

    See https://www.earthnet.tech/ for the current download links.

Expected layout once complete:

    data/earthnet2021/
    ├── train/                  ~23000 cubes, ~100 GB
    ├── iid_test_split/
    ├── ood_test_split/
    ├── extreme_test_split/
    └── seasonal_test_split/

Each cube is a compressed .npz holding `highresdynamic` with shape
(128, 128, 7, T) and channels [blue, green, red, nir, cld, scl, cldmsk].

Then point the config at it:

    tinyearth-data data=earthnet2021 data.root=earthnet2021

Citation
--------
Requena-Mesa, Benson, Reichstein, Runge, Denzler.
"EarthNet2021: A large-scale dataset and challenge for Earth surface
forecasting as a guided video prediction task." CVPR Workshops, 2021.
"""


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments.

    Args:
        argv: Argument list. ``None`` uses :data:`sys.argv`.

    Returns:
        The parsed arguments.
    """
    parser = argparse.ArgumentParser(
        description="Download EarthNet2021 (not redistributed by TinyEarth).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=project_root() / "data" / "earthnet2021",
        help="Destination directory (default: %(default)s).",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        choices=[*SPLIT_CHOICES, "all"],
        default=["all"],
        help="Splits to fetch (default: all).",
    )
    parser.add_argument(
        "--instructions",
        action="store_true",
        help="Print manual download instructions and exit.",
    )
    return parser.parse_args(argv)


def download(root: Path, splits: list[str]) -> int:
    """Invoke the official downloader.

    Args:
        root: Destination directory.
        splits: Splits to fetch.

    Returns:
        A process exit code.
    """
    try:
        import earthnet
    except ImportError:
        logger.error("The `earthnet` toolkit is not installed.")
        logger.error('Install it with:  pip install -e ".[data]"')
        print(INSTRUCTIONS)
        return 1

    root.mkdir(parents=True, exist_ok=True)
    targets = list(SPLIT_CHOICES) if "all" in splits else splits

    logger.warning("EarthNet2021 is large; the full download exceeds 100 GB.")
    logger.info("destination: %s", root)
    logger.info("splits: %s", ", ".join(targets))

    for split in targets:
        logger.info("downloading %r ...", split)
        try:
            earthnet.Downloader.get(str(root), split)
        except Exception as error:
            logger.error("failed to download %r: %s", split, error)
            return 1

    logger.info("done. Verify with:  tinyearth-data data=earthnet2021")
    return 0


def main(argv: list[str] | None = None) -> int:
    """Entry point.

    Args:
        argv: Argument list. ``None`` uses :data:`sys.argv`.

    Returns:
        A process exit code.
    """
    args = parse_args(argv)
    setup_logging(rich=True, force=True)

    if args.instructions:
        print(INSTRUCTIONS)
        return 0
    return download(args.root, args.splits)


if __name__ == "__main__":
    sys.exit(main())
