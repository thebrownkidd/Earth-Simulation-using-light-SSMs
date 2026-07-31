"""The EarthNet2021 dataset.

Composes the pieces around it -- :mod:`~tinyearth.datasets.minicube` for
reading, :mod:`~tinyearth.datasets.windows` for temporal slicing,
:mod:`~tinyearth.datasets.masking` for cloud handling,
:mod:`~tinyearth.datasets.normalization` for scaling -- into a
:class:`torch.utils.data.Dataset` yielding the contract in
:mod:`tinyearth.datasets.types`.

Indexing strategy
-----------------
The dataset builds a flat index of ``(cube_path, window)`` pairs up front, so
``__len__`` is exact and ``__getitem__`` is O(1). The alternative -- computing
windows lazily -- would make the epoch length unknown, which breaks learning-rate
schedules and progress reporting.

Building that index requires knowing each cube's frame count. Reading every
cube header at startup is slow over 32k files, so the count is taken from
``expected_frames`` and verified per-cube at read time. A cube that disagrees is
skipped with a warning rather than crashing an overnight run.

Caching
-------
Consecutive windows overlap heavily: with ``history=4, horizon=1, stride=1``, a
30-frame cube yields 26 samples that all read the same file. An LRU cache of
decoded cubes therefore turns ~26 decompressions into one. This is the single
highest-value optimisation in the pipeline, and is on by default.

The cache lives per worker process. With ``num_workers=N`` the effective memory
is ``N * cache_size * cube_bytes``, so the default is deliberately small.
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass, replace
from pathlib import Path

from torch.utils.data import Dataset

from tinyearth.datasets.masking import (
    MaskPolicy,
    apply_mask_policy,
    passes_validity_threshold,
)
from tinyearth.datasets.minicube import Minicube, MinicubeFormatError, read_minicube
from tinyearth.datasets.normalization import IdentityNormalizer, Normalizer
from tinyearth.datasets.splits import (
    Split,
    discover_cubes,
    partition_train_val,
    split_directory,
)
from tinyearth.datasets.types import Sample, SampleMetadata
from tinyearth.datasets.windows import Window, WindowMode, WindowSpec, generate_windows
from tinyearth.utils.logging import get_logger

__all__ = ["EarthNet2021Dataset", "IndexEntry"]

logger = get_logger(__name__)

DEFAULT_TRAIN_FRAMES = 30
"""Frames in a training minicube: 30 timesteps at 5-day intervals."""

DEFAULT_CONTEXT_FRAMES = 10
"""Frames of context in the official evaluation protocol."""


@dataclass(frozen=True)
class IndexEntry:
    """One addressable sample: a cube plus the window cut from it.

    Attributes:
        path: Source minicube.
        window: Temporal window within that cube.
    """

    path: Path
    window: Window


class _CubeCache:
    """A small LRU cache of decoded minicubes.

    Not thread-safe. DataLoader workers are separate *processes*, so each holds
    its own cache and no locking is needed.
    """

    def __init__(self, capacity: int) -> None:
        """Create the cache.

        Args:
            capacity: Maximum cubes retained. ``0`` disables caching.
        """
        self.capacity = max(0, capacity)
        self._entries: OrderedDict[Path, Minicube] = OrderedDict()
        self.hits = 0
        self.misses = 0

    def get(self, path: Path) -> Minicube | None:
        """Return a cached cube, or ``None``."""
        if self.capacity == 0:
            return None
        cube = self._entries.get(path)
        if cube is None:
            self.misses += 1
            return None
        self._entries.move_to_end(path)
        self.hits += 1
        return cube

    def put(self, path: Path, cube: Minicube) -> None:
        """Insert a cube, evicting the least recently used if full."""
        if self.capacity == 0:
            return
        self._entries[path] = cube
        self._entries.move_to_end(path)
        while len(self._entries) > self.capacity:
            self._entries.popitem(last=False)

    def clear(self) -> None:
        """Drop all cached cubes."""
        self._entries.clear()

    def __len__(self) -> int:
        """Number of cubes currently held."""
        return len(self._entries)


class EarthNet2021Dataset(Dataset[Sample]):
    """EarthNet2021 minicubes as ``(history, forecast)`` windows.

    Args:
        root: Dataset root -- the directory containing ``train/`` and the
            ``*_test_split/`` directories.
        split: Which split to read.
        history_length: Context frames fed to the model.
        horizon: Frames to forecast.
        stride: Step between sliding windows. Ignored for test splits, which
            use a single anchored window per cube.
        channels: Imagery channel indices to keep, from
            ``(blue, green, red, nir)``. ``None`` keeps all four.
        cloud_masking: Read quality masks and emit ``images_mask`` /
            ``target_mask``. Strongly recommended -- see
            :mod:`tinyearth.datasets.masking`.
        mask_policy: How invalid pixels are filled.
        min_valid_fraction: Skip windows whose target is cloudier than this
            allows. Requires ``cloud_masking``.
        normalizer: Applied to imagery after masking. Defaults to identity.
        val_fraction: Fraction of training cubes held out for validation.
        split_salt: Salt for the deterministic train/val partition.
        expected_frames: Frame count assumed when building the index.
        context_frames: Context length for anchored (test) windowing.
        cache_size: Decoded cubes retained per worker. ``0`` disables.
        nan_is_invalid: Treat NaN imagery pixels as invalid.
        max_cubes: Read at most this many cubes. For smoke tests and debugging.

    Raises:
        SplitNotFoundError: If the split's directory is absent.
        ValueError: If the configuration is internally inconsistent.
    """

    def __init__(
        self,
        root: Path | str,
        split: Split | str = Split.TRAIN,
        *,
        history_length: int = 4,
        horizon: int = 1,
        stride: int = 1,
        channels: tuple[int, ...] | None = None,
        cloud_masking: bool = True,
        mask_policy: MaskPolicy | str = MaskPolicy.ZERO,
        min_valid_fraction: float = 0.0,
        normalizer: Normalizer | None = None,
        val_fraction: float = 0.1,
        split_salt: str = "tinyearth",
        expected_frames: int = DEFAULT_TRAIN_FRAMES,
        context_frames: int = DEFAULT_CONTEXT_FRAMES,
        cache_size: int = 4,
        nan_is_invalid: bool = True,
        max_cubes: int | None = None,
    ) -> None:
        self.root = Path(root)
        self.split = Split(split) if isinstance(split, str) else split
        self.spec = WindowSpec(history_length=history_length, horizon=horizon, stride=stride)
        self.channels = tuple(channels) if channels is not None else None
        self.cloud_masking = cloud_masking
        self.mask_policy = MaskPolicy(mask_policy) if isinstance(mask_policy, str) else mask_policy
        self.min_valid_fraction = min_valid_fraction
        self.normalizer: Normalizer = normalizer if normalizer is not None else IdentityNormalizer()
        self.expected_frames = expected_frames
        self.context_frames = context_frames
        self.nan_is_invalid = nan_is_invalid

        if min_valid_fraction > 0.0 and not cloud_masking:
            raise ValueError(
                "min_valid_fraction > 0 requires cloud_masking=True; without masks "
                "there is no way to tell which pixels are valid."
            )
        if self.channels is not None and not self.channels:
            raise ValueError("channels must be non-empty when specified.")

        self._cache = _CubeCache(cache_size)
        self._cubes = self._collect_cubes(val_fraction, split_salt, max_cubes)
        self._index = self._build_index()

        logger.info(
            "%s split %r: %d cubes -> %d windows (history=%d, horizon=%d, stride=%d)",
            type(self).__name__,
            self.split.value,
            len(self._cubes),
            len(self._index),
            history_length,
            horizon,
            stride,
        )

    # -- construction -------------------------------------------------------

    def _collect_cubes(
        self, val_fraction: float, salt: str, max_cubes: int | None
    ) -> tuple[Path, ...]:
        """Locate the cube files belonging to this split."""
        directory = split_directory(self.root, self.split)
        cubes = discover_cubes(directory)
        discovered = len(cubes)

        if self.split.derives_from_train:
            cubes = partition_train_val(cubes, self.split, val_fraction, salt)

        if not cubes:
            if discovered == 0:
                raise FileNotFoundError(
                    f"No .npz minicubes found for split {self.split.value!r} under "
                    f"{directory}. The directory exists but is empty -- the download "
                    "may be incomplete."
                )
            # Cubes exist, but the hash partition gave this split none of them.
            # Distinguishing these cases matters: the fix is completely different.
            raise ValueError(
                f"Split {self.split.value!r} is empty: the train/val partition assigned "
                f"none of the {discovered} cubes under {directory} to it "
                f"(val_fraction={val_fraction}, split_salt={salt!r}).\n"
                "Hash partitioning assigns each cube independently, so at small cube "
                "counts a low val_fraction can round down to zero. Raise val_fraction, "
                "add cubes, or change split_salt."
            )
        if max_cubes is not None:
            cubes = cubes[:max_cubes]
        return cubes

    @property
    def _window_mode(self) -> WindowMode:
        """Sliding for train/val, anchored for the official test tracks."""
        return WindowMode.ANCHORED if self.split.is_test else WindowMode.SLIDING

    def _build_index(self) -> tuple[IndexEntry, ...]:
        """Enumerate every ``(cube, window)`` pair for this split."""
        mode = self._window_mode
        windows = generate_windows(
            self.expected_frames,
            self.spec,
            mode=mode,
            context_length=self.context_frames if mode is WindowMode.ANCHORED else None,
        )
        if not windows:
            raise ValueError(
                f"A window of {self.spec.total_length} frames "
                f"(history={self.spec.history_length} + horizon={self.spec.horizon}) "
                f"does not fit in a cube of {self.expected_frames} frames."
            )
        return tuple(
            IndexEntry(path=path, window=window) for path in self._cubes for window in windows
        )

    # -- reading ------------------------------------------------------------

    def _load_cube(self, path: Path) -> Minicube:
        """Read a cube, using the LRU cache when possible."""
        cached = self._cache.get(path)
        if cached is not None:
            return cached
        cube = read_minicube(
            path, nan_is_invalid=self.nan_is_invalid, apply_mask=self.cloud_masking
        )
        if self.channels is not None:
            cube = replace(cube, images=cube.images[:, list(self.channels)])
        self._cache.put(path, cube)
        return cube

    def _build_sample(self, entry: IndexEntry) -> Sample | None:
        """Materialise one sample, or ``None`` if it should be skipped."""
        cube = self._load_cube(entry.path)
        window = entry.window

        if cube.n_frames < window.stop:
            logger.warning(
                "%s has %d frames but the index expects at least %d; skipping. "
                "Set expected_frames to match the data.",
                entry.path.name,
                cube.n_frames,
                window.stop,
            )
            return None

        history_valid = cube.valid[window.history_slice]
        target_valid = cube.valid[window.target_slice]

        if self.cloud_masking and not passes_validity_threshold(
            target_valid, self.min_valid_fraction
        ):
            return None

        history = cube.images[window.history_slice]
        target = cube.images[window.target_slice]

        if self.cloud_masking:
            history = apply_mask_policy(history, history_valid, self.mask_policy)
            target = apply_mask_policy(target, target_valid, self.mask_policy)

        sample: Sample = {
            "images": self.normalizer(history),
            "target": self.normalizer(target),
            "metadata": SampleMetadata(
                cube_id=cube.cube_id,
                split=self.split.value,
                source=str(entry.path),
                start_index=window.start,
                history_length=window.history_length,
                horizon=window.horizon,
                valid_fraction=float(target_valid.mean().item()) if self.cloud_masking else 1.0,
            ),
        }
        if self.cloud_masking:
            sample["images_mask"] = history_valid
            sample["target_mask"] = target_valid
        return sample

    # -- Dataset protocol ---------------------------------------------------

    def __len__(self) -> int:
        """Number of windows across all cubes in this split."""
        return len(self._index)

    def __getitem__(self, index: int) -> Sample:
        """Return the sample at ``index``.

        Windows rejected by ``min_valid_fraction``, and cubes that fail to
        decode, fall through to the next index rather than raising. A single
        corrupt file in a 32k-cube download should not kill a training run;
        the failure is logged instead.

        Args:
            index: Position in the flat window index.

        Returns:
            The sample.

        Raises:
            IndexError: If ``index`` is out of range, or if no usable sample
                exists at or after it -- which means the validity threshold has
                rejected the entire tail of the dataset.
        """
        if not -len(self._index) <= index < len(self._index):
            raise IndexError(f"Index {index} out of range for {len(self._index)} windows.")
        start = index % len(self._index)

        for offset in range(len(self._index)):
            position = (start + offset) % len(self._index)
            entry = self._index[position]
            try:
                sample = self._build_sample(entry)
            except (MinicubeFormatError, OSError, ValueError) as error:
                logger.warning("Skipping %s: %s", entry.path.name, error)
                continue
            if sample is not None:
                return sample

        raise IndexError(
            f"No usable sample found starting from index {index}. Every window was "
            f"rejected -- min_valid_fraction={self.min_valid_fraction} is probably too "
            "strict for this data."
        )

    # -- introspection ------------------------------------------------------

    @property
    def cubes(self) -> tuple[Path, ...]:
        """Cube files backing this split."""
        return self._cubes

    @property
    def index(self) -> tuple[IndexEntry, ...]:
        """The flat window index."""
        return self._index

    @property
    def n_channels(self) -> int:
        """Number of imagery channels a sample carries."""
        return len(self.channels) if self.channels is not None else 4

    def cache_statistics(self) -> dict[str, int]:
        """Return cache hit/miss counters for this worker.

        Returns:
            Counts keyed ``hits``, ``misses`` and ``size``.
        """
        return {
            "hits": self._cache.hits,
            "misses": self._cache.misses,
            "size": len(self._cache),
        }

    def __repr__(self) -> str:
        """Return a debugging representation."""
        return (
            f"{type(self).__name__}(root={str(self.root)!r}, split={self.split.value!r}, "
            f"cubes={len(self._cubes)}, windows={len(self._index)}, "
            f"history={self.spec.history_length}, horizon={self.spec.horizon})"
        )
