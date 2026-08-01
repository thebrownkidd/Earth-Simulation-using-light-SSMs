#!/usr/bin/env python
"""Download EarthNet2021, in whole or in part.

TinyEarth does **not** redistribute the dataset. This script fetches it from the
servers of the original authors, using the URL manifest published by the
official ``earthnet`` toolkit.

Why not just call the toolkit's own downloader?

1. **It is all-or-nothing.** ``earthnet.Downloader.get`` walks every tarball in a
   split. The training split is 160 tarballs of roughly 1 GB each, so the
   smallest thing it can do is a ~155 GB download. A laptop that will train on a
   few thousand cubes has no use for that, and the dataset is chunked into
   independently-addressable tarballs, so there is no technical reason to take
   all of it. ``--max-tarballs`` takes a prefix instead.
2. **Its TLS verification fails on some hosts.** It downloads through the
   default SSL context, which on a stock Windows Python cannot build a trust
   chain to the download host and raises ``CERTIFICATE_VERIFY_FAILED``. This
   script verifies against the ``certifi`` bundle, which works on every platform.

Both downloaders are otherwise equivalent: same URLs, same SHA-256 manifest,
same on-disk layout.

Sizes, measured rather than quoted:

===================  =========  ==========  =========================
Split                Tarballs   Total       Approx. cubes
===================  =========  ==========  =========================
``train``            160        ~155 GB     ~23000
``iid``              29         ~5 GB       ~4000
``ood``              29         ~5 GB       ~4000
``extreme``          58         ~7 GB       ~4000
``seasonal``         200        ~15 GB      ~4000
===================  =========  ==========  =========================

Usage::

    # A subset sized for a laptop: ~8 GB, ~1200 cubes, spread across tiles.
    python scripts/download_earthnet2021.py --splits train --max-tarballs 8 --stride 20

    # Everything (needs ~190 GB free).
    python scripts/download_earthnet2021.py --splits all

    python scripts/download_earthnet2021.py --instructions

``--stride`` matters more than it looks: the tarballs are grouped by Sentinel-2
tile, so a plain prefix of the training split is one region rather than a sample
of the dataset. See :func:`select`.

Downloads resume: a tarball already recorded in ``.download_progress.json``
under the destination root is skipped, so an interrupted run continues where it
stopped.

Citation
--------
Requena-Mesa, Benson, Reichstein, Runge, Denzler.
"EarthNet2021: A large-scale dataset and challenge for Earth surface
forecasting as a guided video prediction task." CVPR Workshops, 2021.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import ssl
import sys
import tarfile
import time
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from urllib.error import URLError

from tqdm import tqdm  # type: ignore[import-untyped]

from tinyearth.utils.logging import get_logger, setup_logging
from tinyearth.utils.paths import project_root

logger = get_logger("tinyearth.scripts.download")

SPLIT_CHOICES = ("train", "iid", "ood", "extreme", "seasonal")

_PROGRESS_FILE = ".download_progress.json"
_CHUNK_BYTES = 1 << 20
_HASH_BUFFER_BYTES = 8 << 20

INSTRUCTIONS = """
EarthNet2021 download
=====================

TinyEarth does not redistribute this dataset. It is fetched from the servers of
the original authors.

    pip install -e ".[data]"          # provides the URL manifest
    python scripts/download_earthnet2021.py --splits train --max-tarballs 8 --stride 20

The dataset is chunked into ~1 GB tarballs, so `--max-tarballs N` takes a
reproducible subset of a split rather than all of it. Eight training tarballs is
roughly 8 GB and 1200 cubes, which is enough to train the models in this
repository.

Use `--stride` as well. The tarballs are grouped by Sentinel-2 tile, so the
first eight archives are 1200 cubes of one region in southwest Iberia. Spreading
the selection across the manifest gives several landscapes for the same bytes.
Check with `ls data/earthnet2021/train/`, which lists one directory per tile.

Expected layout once complete:

    data/earthnet2021/
    |-- train/                  cubes grouped by Sentinel-2 tile
    |-- iid_test_split/
    |-- ood_test_split/
    |-- extreme_test_split/
    `-- seasonal_test_split/

Each cube is a compressed .npz holding `highresdynamic` with shape
(128, 128, 7, T) and channels [blue, green, red, nir, cld, scl, cldmsk].

Then point a run at it:

    tinyearth-train +experiment=earthnet_s4d
"""


@dataclass(frozen=True)
class Tarball:
    """One downloadable chunk of a split.

    Attributes:
        name: Archive filename, e.g. ``train_000.tar.gz``.
        url: Direct download URL.
        sha256: Expected digest of the archive.
    """

    name: str
    url: str
    sha256: str


def load_manifest(split: str) -> tuple[Tarball, ...]:
    """Return the tarballs making up a split, in a stable order.

    The manifest ships with the ``earthnet`` toolkit. Its order is not sorted --
    the training split begins with ``train_159`` -- so it is sorted by name here.
    A subset is only reproducible if the order is deterministic.

    Args:
        split: One of :data:`SPLIT_CHOICES`.

    Returns:
        The split's tarballs, ordered by name.

    Raises:
        SystemExit: If the ``earthnet`` toolkit is not installed.
    """
    try:
        from earthnet.download_links import DOWNLOAD_LINKS
    except ImportError:
        logger.error("The `earthnet` toolkit is not installed; it supplies the URL manifest.")
        logger.error('Install it with:  pip install -e ".[data]"')
        print(INSTRUCTIONS)
        raise SystemExit(1) from None

    entries = [Tarball(name, url, sha) for name, url, sha in DOWNLOAD_LINKS[split]]
    return tuple(sorted(entries, key=lambda entry: entry.name))


def select(manifest: tuple[Tarball, ...], count: int | None, stride: int) -> tuple[Tarball, ...]:
    """Choose which tarballs of a split to fetch.

    **The tarballs are grouped by Sentinel-2 tile**, so consecutive archives
    hold the same patch of the planet. Taking a plain prefix of the training
    split is therefore not a sample of EarthNet2021 -- the first eight tarballs
    are 1200 cubes of one tile in southwest Iberia, and a model trained and
    validated on them has seen exactly one landscape.

    ``stride`` spreads the selection across the manifest instead, so a small
    subset spans several tiles and the held-out cubes are not all neighbours of
    the training ones. It is the difference between a subset and a biased
    sample, and it costs nothing.

    Args:
        manifest: The split's tarballs, ordered by name.
        count: How many to take. ``None`` takes all.
        stride: Step between selected indices. ``1`` gives a contiguous prefix.

    Returns:
        The chosen tarballs.
    """
    if count is None:
        return manifest
    return tuple(manifest[index] for index in range(0, len(manifest), stride))[:count]


def _ssl_context() -> ssl.SSLContext:
    """Build an SSL context that trusts the ``certifi`` roots.

    The default context uses the platform trust store, which on Windows fails to
    build a chain to the download host. ``certifi`` ships the Mozilla root
    bundle and succeeds everywhere.

    Returns:
        A verifying SSL context.
    """
    try:
        import certifi
    except ImportError:  # pragma: no cover - certifi arrives with the data extra
        logger.warning("certifi is unavailable; falling back to the platform trust store.")
        return ssl.create_default_context()
    return ssl.create_default_context(cafile=certifi.where())


def digest_of(path: Path) -> str:
    """Return the SHA-256 digest of a file, read in bounded chunks with progress."""
    sha = hashlib.sha256()
    size = path.stat().st_size
    with (
        path.open("rb") as handle,
        tqdm(
            total=size,
            unit="B",
            unit_scale=True,
            unit_divisor=1024,
            desc=f"    Hashing {path.name}",
            leave=False,
            ncols=100,
        ) as pbar,
    ):
        while chunk := handle.read(_HASH_BUFFER_BYTES):
            sha.update(chunk)
            pbar.update(len(chunk))
    return sha.hexdigest()


def read_progress(root: Path) -> set[str]:
    """Return the names of tarballs already downloaded and extracted."""
    record = root / _PROGRESS_FILE
    if not record.is_file():
        return set()
    try:
        return set(json.loads(record.read_text(encoding="utf-8")))
    except (OSError, ValueError):
        logger.warning("Could not read %s; treating the download as fresh.", record)
        return set()


def write_progress(root: Path, done: set[str]) -> None:
    """Persist the set of completed tarballs."""
    (root / _PROGRESS_FILE).write_text(json.dumps(sorted(done), indent=2), encoding="utf-8")


def fetch(tarball: Tarball, destination: Path, context: ssl.SSLContext) -> None:
    """Stream one tarball to disk, reporting throughput with progress bar.

    Args:
        tarball: Archive to fetch.
        destination: File to write.
        context: SSL context used for the request.

    Raises:
        URLError: If the transfer fails.
    """
    opener = urllib.request.build_opener(urllib.request.HTTPSHandler(context=context))
    partial = destination.with_suffix(destination.suffix + ".part")
    start = time.perf_counter()

    logger.info("  Downloading %s ...", tarball.name)
    with opener.open(tarball.url, timeout=120) as response:
        declared = response.headers.get("Content-Length")
        total = int(declared) if declared else None

        with (
            partial.open("wb") as handle,
            tqdm(
                total=total,
                unit="B",
                unit_scale=True,
                unit_divisor=1024,
                desc=f"    {tarball.name}",
                leave=False,
                ncols=100,
            ) as pbar,
        ):
            while chunk := response.read(_CHUNK_BYTES):
                handle.write(chunk)
                pbar.update(len(chunk))

    elapsed = time.perf_counter() - start
    partial.replace(destination)
    throughput = total / 1e6 / max(elapsed, 1e-9) if total else 0
    logger.info(
        "    ✓ %s  %.0f MB in %.0f s (%.1f MB/s)",
        tarball.name,
        total / 1e6 if total else 0,
        elapsed,
        throughput,
    )


def extract(archive: Path, root: Path) -> None:
    """Unpack a tarball into the dataset root with progress bar.

    Uses the ``data`` extraction filter, which refuses absolute paths and
    parent-directory escapes. Python 3.14 makes this the default; setting it
    explicitly keeps the behaviour identical on 3.12 and 3.13.

    Args:
        archive: Tarball to unpack.
        root: Destination directory.
    """
    logger.info("    Extracting %s ...", archive.name)
    with tarfile.open(archive, "r:gz") as tar:
        members = tar.getmembers()
        with tqdm(total=len(members), desc=f"    {archive.name}", leave=False, ncols=100) as pbar:
            for member in members:
                tar.extract(member, path=root, filter="data")
                pbar.update(1)
    logger.info("    ✓ Extracted %s", archive.name)


def download(
    root: Path, splits: list[str], max_tarballs: int | None, keep: bool, stride: int
) -> int:
    """Fetch, verify and unpack the requested tarballs with progress tracking.

    Args:
        root: Destination directory.
        splits: Splits to fetch, or ``["all"]``.
        max_tarballs: Take only this many tarballs per split; ``None`` for all.
        keep: Retain the downloaded archives after extraction.
        stride: Step between selected tarballs; see :func:`select`.

    Returns:
        A process exit code.
    """
    targets = list(SPLIT_CHOICES) if "all" in splits else splits
    root.mkdir(parents=True, exist_ok=True)
    context = _ssl_context()

    logger.info("")
    logger.info("=" * 80)
    logger.info("EarthNet2021 Download with Verification")
    logger.info("=" * 80)
    logger.info("Destination: %s", root)

    # Load progress
    logger.info("Reading prior progress...")
    done = read_progress(root)

    # Load and plan manifests
    logger.info("Loading manifests for %s...", ", ".join(targets))
    plan = {}
    for split in targets:
        logger.info("  Loading %s split...", split)
        manifest = load_manifest(split)
        selected = select(manifest, max_tarballs, stride)
        plan[split] = selected
        logger.info("    → %d tarballs from %d total", len(selected), len(manifest))

    planned = sum(len(chunks) for chunks in plan.values())
    remaining = [chunk for chunks in plan.values() for chunk in chunks if chunk.name not in done]

    # Show detailed plan with tile info
    logger.info("")
    logger.info("Download Plan:")
    for split in targets:
        tarballs = plan[split]
        if tarballs:
            tiles = {tb.name.split("_")[1] for tb in tarballs}  # Extract tile ID
            logger.info("  %s: %d tarballs (%s)", split, len(tarballs), ", ".join(sorted(tiles)))

    # Disk space check
    disk = shutil.disk_usage(root)
    free_gb = disk.free / 1e9
    used_gb = disk.used / 1e9
    logger.info("")
    logger.info("Disk Space:")
    logger.info("  Available: %.1f GB", free_gb)
    logger.info("  In use:    %.1f GB", used_gb)
    logger.info("  Required:  ~%.1f GB (each tarball ~1 GB, expands to ~2 GB)", len(remaining) * 2)

    # Summary
    logger.info("")
    logger.info("Summary:")
    logger.info("  Total planned: %d tarballs", planned)
    logger.info("  Already done:  %d tarballs", planned - len(remaining))
    logger.info("  To fetch:      %d tarballs", len(remaining))

    if not remaining:
        logger.info("")
        logger.info("✓ Nothing to do; every planned tarball is already present.")
        return 0

    logger.info("")
    logger.info("Starting downloads...")
    logger.info("=" * 80)

    # Download loop with progress bar
    with tqdm(
        total=len(remaining), desc="Overall progress", unit="tarball", ncols=100
    ) as pbar_overall:
        for index, tarball in enumerate(remaining, start=1):
            logger.info("")
            logger.info("[%d/%d] Processing %s", index, len(remaining), tarball.name)

            archive = root / tarball.name

            # Download
            try:
                fetch(tarball, archive, context)
            except (URLError, OSError, TimeoutError) as error:
                logger.error("")
                logger.error("✗ Download failed for %s: %s", tarball.name, error)
                logger.error("  Re-run to resume; completed tarballs are not refetched.")
                return 1

            # Verify
            logger.info("    Verifying SHA-256...")
            with tqdm(
                total=archive.stat().st_size,
                unit="B",
                unit_scale=True,
                desc="    Hashing",
                leave=False,
                ncols=100,
            ) as pbar_hash:
                sha = hashlib.sha256()
                with archive.open("rb") as f:
                    while chunk := f.read(_HASH_BUFFER_BYTES):
                        sha.update(chunk)
                        pbar_hash.update(len(chunk))
                actual = sha.hexdigest()

            if actual != tarball.sha256:
                archive.unlink(missing_ok=True)
                logger.error("")
                logger.error("✗ Checksum mismatch for %s", tarball.name)
                logger.error("    Expected: %s", tarball.sha256)
                logger.error("    Got:      %s", actual)
                logger.error("    File deleted.")
                return 1
            logger.info("    ✓ Checksum verified")

            # Extract
            extract(archive, root)

            # Cleanup
            if not keep:
                archive.unlink(missing_ok=True)
                logger.info("    Cleaned up archive")

            done.add(tarball.name)
            write_progress(root, done)

            pbar_overall.update(1)

    # Final report
    logger.info("")
    logger.info("=" * 80)
    cubes = sum(1 for _ in root.rglob("*.npz"))
    logger.info("✓ Download complete!")
    logger.info("  Total cubes: %d", cubes)
    logger.info("  Location: %s", root)
    logger.info("")
    logger.info("Next step:")
    logger.info("  tinyearth-train +experiment=earthnet_s4d")
    logger.info("=" * 80)
    logger.info("")
    return 0


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
        default=["train"],
        help="Splits to fetch (default: %(default)s).",
    )
    parser.add_argument(
        "--max-tarballs",
        type=int,
        default=None,
        help=(
            "Fetch only the first N ~1 GB tarballs of each split, ordered by name. "
            "Omit to fetch the whole split (155 GB for train)."
        ),
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=1,
        help=(
            "Step between selected tarballs. Archives are grouped by Sentinel-2 tile, "
            "so a stride of 1 (the default) gives one region per few tarballs. Raise it "
            "to spread a small subset across the globe, e.g. --max-tarballs 8 --stride 20."
        ),
    )
    parser.add_argument(
        "--keep-archives",
        action="store_true",
        help="Keep the .tar.gz files after extraction (roughly doubles disk use).",
    )
    parser.add_argument(
        "--instructions",
        action="store_true",
        help="Print manual download instructions and exit.",
    )
    return parser.parse_args(argv)


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
    if args.max_tarballs is not None and args.max_tarballs < 1:
        logger.error("--max-tarballs must be >= 1, got %d.", args.max_tarballs)
        return 1
    if args.stride < 1:
        logger.error("--stride must be >= 1, got %d.", args.stride)
        return 1
    return download(args.root, args.splits, args.max_tarballs, args.keep_archives, args.stride)


if __name__ == "__main__":
    sys.exit(main())
