"""Dataset splits and on-disk layout.

EarthNet2021 ships a training set and four *test* tracks, each probing a
different kind of generalisation:

============  ==================================================================
Track         Tests generalisation to...
============  ==================================================================
``iid``       held-out cubes from the same distribution as training
``ood``       geographically disjoint regions
``extreme``   extreme climate events (drought, heatwave)
``seasonal``  a full seasonal cycle, with a much longer forecast horizon
============  ==================================================================

**There is no official validation split.** One is required nonetheless -- model
selection on a test track would invalidate every number reported against it. So
TinyEarth carves a validation set out of training cubes.

The partition is by *hash of the cube identifier*, not by shuffling with a
seed. Both are deterministic, but only hashing is stable when cubes are added
or removed: a shuffle reassigns every cube when the file count changes, which
silently moves previously-validation cubes into training and contaminates the
comparison against earlier runs. Hashing assigns each cube independently, so
the split of existing cubes never changes.

The hash is BLAKE2b rather than :func:`hash`, whose string seed varies per
process unless ``PYTHONHASHSEED`` is fixed.
"""

from __future__ import annotations

import hashlib
from collections.abc import Iterable, Sequence
from enum import StrEnum
from pathlib import Path

__all__ = [
    "TEST_SPLITS",
    "Split",
    "SplitNotFoundError",
    "assign_partition",
    "discover_cubes",
    "partition_train_val",
    "split_directory",
    "split_directory_name",
    "summarise_partition",
]

_HASH_BUCKETS = 10_000


class Split(StrEnum):
    """A named dataset split.

    Attributes:
        TRAIN: Training cubes, minus the held-out validation fraction.
        VAL: Validation cubes, carved deterministically from training.
        IID_TEST: Held-out cubes from the training distribution.
        OOD_TEST: Geographically out-of-distribution cubes.
        EXTREME_TEST: Extreme-climate-event cubes.
        SEASONAL_TEST: Full-seasonal-cycle cubes.
    """

    TRAIN = "train"
    VAL = "val"
    IID_TEST = "iid_test"
    OOD_TEST = "ood_test"
    EXTREME_TEST = "extreme_test"
    SEASONAL_TEST = "seasonal_test"

    @property
    def is_test(self) -> bool:
        """Whether this split is one of the four official test tracks."""
        return self in TEST_SPLITS

    @property
    def derives_from_train(self) -> bool:
        """Whether this split is carved out of the training cubes."""
        return self in {Split.TRAIN, Split.VAL}


TEST_SPLITS: frozenset[Split] = frozenset(
    {Split.IID_TEST, Split.OOD_TEST, Split.EXTREME_TEST, Split.SEASONAL_TEST}
)
"""The four official evaluation tracks."""

_DIRECTORY_NAMES: dict[Split, str] = {
    Split.TRAIN: "train",
    Split.VAL: "train",  # carved from train; no directory of its own
    Split.IID_TEST: "iid_test_split",
    Split.OOD_TEST: "ood_test_split",
    Split.EXTREME_TEST: "extreme_test_split",
    Split.SEASONAL_TEST: "seasonal_test_split",
}
"""Directory names used by the official EarthNet2021 download."""


class SplitNotFoundError(FileNotFoundError):
    """Raised when a split's directory is missing from the dataset root."""


def split_directory_name(split: Split) -> str:
    """Return the on-disk directory name for a split.

    Args:
        split: Split to name.

    Returns:
        The directory name used by the official download. :attr:`Split.VAL`
        maps to ``"train"``, since validation is carved from the training cubes.
    """
    return _DIRECTORY_NAMES[split]


def split_directory(root: Path | str, split: Split) -> Path:
    """Return the directory holding a split's cubes.

    Args:
        root: Dataset root, i.e. the directory containing ``train/``.
        split: Split to locate.

    Returns:
        The split's directory.

    Raises:
        SplitNotFoundError: If the directory does not exist, with download
            instructions -- a missing dataset is the single most common setup
            failure and deserves an actionable message.
    """
    directory = Path(root) / _DIRECTORY_NAMES[split]
    if not directory.is_dir():
        raise SplitNotFoundError(
            f"Split {split.value!r} expects {directory} to exist, but it does not.\n"
            f"Download EarthNet2021 with:\n"
            f"    python scripts/download_earthnet2021.py --root {Path(root)}\n"
            f"or see docs/datasets.md. TinyEarth does not redistribute the dataset."
        )
    return directory


def discover_cubes(directory: Path | str, pattern: str = "*.npz") -> tuple[Path, ...]:
    """List minicube files under ``directory``, recursively.

    Args:
        directory: Directory to scan.
        pattern: Glob pattern for minicube files.

    Returns:
        Absolute paths, sorted. Sorting matters: it makes dataset ordering
        independent of filesystem enumeration order, which differs between
        operating systems and would otherwise break reproducibility across
        machines.
    """
    base = Path(directory)
    return tuple(sorted(path.resolve() for path in base.rglob(pattern) if path.is_file()))


def assign_partition(cube_id: str, val_fraction: float, salt: str = "tinyearth") -> Split:
    """Assign one cube to train or validation.

    Args:
        cube_id: Stable cube identifier, normally the filename stem.
        val_fraction: Fraction of cubes to hold out, in ``[0, 1)``.
        salt: Changes the partition without changing cube identifiers. Use a
            different salt to obtain an independent split, e.g. for a
            cross-validation fold.

    Returns:
        :attr:`Split.VAL` or :attr:`Split.TRAIN`.

    Raises:
        ValueError: If ``val_fraction`` is outside ``[0, 1)``. A fraction of 1
            is rejected because it would leave no training data.
    """
    if not 0.0 <= val_fraction < 1.0:
        raise ValueError(f"val_fraction must be in [0, 1), got {val_fraction}.")
    if val_fraction == 0.0:
        return Split.TRAIN

    digest = hashlib.blake2b(f"{salt}:{cube_id}".encode(), digest_size=8).digest()
    bucket = int.from_bytes(digest, "big") % _HASH_BUCKETS
    return Split.VAL if bucket < val_fraction * _HASH_BUCKETS else Split.TRAIN


def partition_train_val(
    cubes: Iterable[Path],
    split: Split,
    val_fraction: float,
    salt: str = "tinyearth",
) -> tuple[Path, ...]:
    """Select the cubes belonging to ``split`` from the training directory.

    Args:
        cubes: Candidate cube paths.
        split: Either :attr:`Split.TRAIN` or :attr:`Split.VAL`.
        val_fraction: Fraction held out for validation.
        salt: Partition salt, see :func:`assign_partition`.

    Returns:
        The selected paths, order preserved.

    Raises:
        ValueError: If ``split`` is not derived from the training cubes.
    """
    if not split.derives_from_train:
        raise ValueError(
            f"{split.value!r} is an official test track and is not partitioned; "
            "read it directly from its own directory."
        )
    return tuple(path for path in cubes if assign_partition(path.stem, val_fraction, salt) is split)


def summarise_partition(
    cubes: Sequence[Path], val_fraction: float, salt: str = "tinyearth"
) -> dict[str, int]:
    """Count how a set of cubes divides into train and validation.

    Args:
        cubes: Candidate cube paths.
        val_fraction: Fraction held out for validation.
        salt: Partition salt.

    Returns:
        Counts keyed by split value.
    """
    counts = {Split.TRAIN.value: 0, Split.VAL.value: 0}
    for path in cubes:
        counts[assign_partition(path.stem, val_fraction, salt).value] += 1
    return counts
