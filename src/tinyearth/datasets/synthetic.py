"""Synthetic minicubes in the EarthNet2021 format.

EarthNet2021 is over 100 GB. Requiring it to run a test, execute the example
notebook, or check that a model wires together would make the repository
effectively unusable for anyone evaluating it -- and would leave CI unable to
test the data path at all.

So this module writes **real** ``.npz`` files in the real on-disk layout. The
important consequence is that :class:`SyntheticEarthNet2021` reuses
:class:`~tinyearth.datasets.earthnet2021.EarthNet2021Dataset` unchanged, which
means the tests exercise the production reader rather than a parallel
implementation that could drift away from it.

The content is *plausible*, not realistic: a smooth spatial gradient with a
seasonal cycle, plus drifting synthetic cloud blobs. It is sufficient to verify
shapes, masking, windowing, normalisation and determinism. It is **not**
sufficient to draw any conclusion about forecasting quality, and no result from
it should ever be reported.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from tinyearth.datasets.earthnet2021 import (
    DEFAULT_CONTEXT_FRAMES,
    DEFAULT_TRAIN_FRAMES,
    EarthNet2021Dataset,
)
from tinyearth.datasets.minicube import N_IMAGE_CHANNELS
from tinyearth.datasets.splits import Split, split_directory_name
from tinyearth.utils.logging import get_logger

__all__ = ["SyntheticEarthNet2021", "SyntheticSpec", "write_synthetic_dataset"]

logger = get_logger(__name__)

_N_CHANNELS = 7
"""Matches the real format: 4 imagery + cld + scl + cldmsk."""

_CLOUD_MASK_CHANNEL = -1
_MARKER_NAME = ".tinyearth_synthetic.json"


@dataclass(frozen=True)
class SyntheticSpec:
    """Parameters of a synthetic dataset.

    Attributes:
        n_cubes: Cubes written per split directory.
        n_frames: Timesteps per cube.
        size: Spatial extent; cubes are ``size x size``. Much smaller than the
            real 128 to keep tests fast.
        cloud_fraction: Approximate share of pixels marked cloudy.
        nan_fraction: Share of pixels set to NaN, exercising the cleaning path.
        seed: Base RNG seed. Cube ``i`` uses ``seed + i``, so content is stable
            regardless of generation order.

    Raises:
        ValueError: If any field is out of range.
    """

    n_cubes: int = 6
    n_frames: int = DEFAULT_TRAIN_FRAMES
    size: int = 16
    cloud_fraction: float = 0.2
    nan_fraction: float = 0.01
    seed: int = 0

    def __post_init__(self) -> None:
        """Validate the specification."""
        if self.n_cubes < 1:
            raise ValueError(f"n_cubes must be >= 1, got {self.n_cubes}.")
        if self.n_frames < 2:
            raise ValueError(f"n_frames must be >= 2, got {self.n_frames}.")
        if self.size < 1:
            raise ValueError(f"size must be >= 1, got {self.size}.")
        for name, value in (
            ("cloud_fraction", self.cloud_fraction),
            ("nan_fraction", self.nan_fraction),
        ):
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be in [0, 1], got {value}.")


def _make_cube(spec: SyntheticSpec, index: int) -> np.ndarray:
    """Generate one cube's ``highresdynamic`` array.

    Args:
        spec: Generation parameters.
        index: Cube index, mixed into the seed.

    Returns:
        A ``(size, size, 7, n_frames)`` float32 array matching the real layout.
    """
    rng = np.random.default_rng(spec.seed + index)
    size, frames = spec.size, spec.n_frames

    # A smooth spatial field with a seasonal cycle, so temporal models have
    # genuine structure to latch onto rather than pure noise.
    yy, xx = np.meshgrid(
        np.linspace(0.0, 1.0, size, dtype=np.float32),
        np.linspace(0.0, 1.0, size, dtype=np.float32),
        indexing="ij",
    )
    phase = rng.uniform(0.0, 2.0 * np.pi)
    time = np.arange(frames, dtype=np.float32)
    season = 0.5 + 0.3 * np.sin(2.0 * np.pi * time / max(frames - 1, 1) + phase)

    cube = np.zeros((size, size, _N_CHANNELS, frames), dtype=np.float32)
    for channel in range(N_IMAGE_CHANNELS):
        base = (0.2 + 0.15 * channel) * xx + 0.25 * yy
        field = base[:, :, None] * season[None, None, :]
        field = field + rng.normal(0.0, 0.02, size=(size, size, frames)).astype(np.float32)
        cube[:, :, channel, :] = np.clip(field, 0.0, 1.0)

    # Cloud mask: drifting blobs rather than i.i.d. noise, so masked regions are
    # spatially coherent the way real cloud is.
    cloudy = np.zeros((size, size, frames), dtype=np.float32)
    if spec.cloud_fraction > 0:
        centre = rng.uniform(0.0, 1.0, size=2)
        drift = rng.normal(0.0, 0.05, size=(frames, 2))
        radius = float(np.sqrt(spec.cloud_fraction / np.pi))
        for t in range(frames):
            centre = np.clip(centre + drift[t], 0.0, 1.0)
            distance = np.sqrt((xx - centre[0]) ** 2 + (yy - centre[1]) ** 2)
            cloudy[:, :, t] = (distance < radius).astype(np.float32)
    cube[:, :, _CLOUD_MASK_CHANNEL, :] = cloudy
    cube[:, :, 4, :] = cloudy  # cld
    cube[:, :, 5, :] = cloudy * 9.0  # scl, nominally a class index

    if spec.nan_fraction > 0:
        holes = rng.random((size, size, frames)) < spec.nan_fraction
        for channel in range(N_IMAGE_CHANNELS):
            cube[:, :, channel, :][holes] = np.nan

    return cube


def write_synthetic_dataset(
    root: Path | str,
    spec: SyntheticSpec | None = None,
    splits: tuple[Split, ...] = (Split.TRAIN,),
    *,
    force: bool = False,
) -> Path:
    """Write a synthetic dataset to ``root``, idempotently.

    A marker file records the spec used. If it matches, generation is skipped,
    so repeated calls in tests and notebooks are cheap.

    Args:
        root: Destination dataset root.
        spec: Generation parameters. Defaults to :class:`SyntheticSpec`.
        splits: Splits to populate. Validation is carved from train and needs
            no directory of its own.
        force: Regenerate even when a matching marker exists.

    Returns:
        The dataset root.
    """
    spec = spec or SyntheticSpec()
    base = Path(root)
    marker = base / _MARKER_NAME
    fingerprint = {"spec": asdict(spec), "splits": sorted(split.value for split in splits)}

    if not force and marker.is_file():
        try:
            if json.loads(marker.read_text(encoding="utf-8")) == fingerprint:
                return base
        except (OSError, json.JSONDecodeError):
            pass  # regenerate on an unreadable marker

    for split in splits:
        directory = base / split_directory_name(split)
        directory.mkdir(parents=True, exist_ok=True)

        # Remove cubes from a previous generation first. Without this, lowering
        # n_cubes leaves stale files behind and the dataset silently keeps
        # reading them. Scoped to our own naming pattern so that a root
        # accidentally pointed at real data cannot lose anything.
        for stale in directory.glob(f"{split.value}_cube_*.npz"):
            stale.unlink()

        for index in range(spec.n_cubes):
            path = directory / f"{split.value}_cube_{index:04d}.npz"
            np.savez_compressed(path, highresdynamic=_make_cube(spec, index))

    marker.parent.mkdir(parents=True, exist_ok=True)
    marker.write_text(json.dumps(fingerprint, indent=2), encoding="utf-8")
    logger.info(
        "wrote %d synthetic cubes per split (%s) to %s",
        spec.n_cubes,
        ", ".join(split.value for split in splits),
        base,
    )
    return base


class SyntheticEarthNet2021(EarthNet2021Dataset):
    """A synthetic dataset that behaves exactly like the real one.

    Materialises cubes on first use, then defers entirely to
    :class:`~tinyearth.datasets.earthnet2021.EarthNet2021Dataset`. Use for
    smoke tests, CI, and running the example notebook without the download.

    Results obtained from this data are meaningless and must never be reported.

    Args:
        root: Where synthetic cubes are written. Should be under the cache
            directory, not the data directory -- these are generated artefacts.
        split: Split to read.
        spec: Generation parameters.
        **kwargs: Forwarded to the base dataset.
    """

    def __init__(
        self,
        root: Path | str,
        split: Split | str = Split.TRAIN,
        *,
        spec: SyntheticSpec | None = None,
        **kwargs: object,
    ) -> None:
        self.spec_synthetic = spec or SyntheticSpec()
        requested = Split(split) if isinstance(split, str) else split

        # Test tracks need their own directories; train and val share one.
        to_write: tuple[Split, ...] = (
            (Split.TRAIN,) if requested.derives_from_train else (Split.TRAIN, requested)
        )
        write_synthetic_dataset(root, self.spec_synthetic, to_write)

        kwargs.setdefault("expected_frames", self.spec_synthetic.n_frames)
        kwargs.setdefault(
            "context_frames", min(DEFAULT_CONTEXT_FRAMES, self.spec_synthetic.n_frames - 1)
        )
        super().__init__(root, requested, **kwargs)  # type: ignore[arg-type]

    def describe(self) -> str:
        """Return a warning suitable for logging at the top of a run.

        Returns:
            A one-line reminder that these results are not meaningful.
        """
        return (
            f"SYNTHETIC DATA ({self.spec_synthetic.n_cubes} cubes, "
            f"{self.spec_synthetic.size}x{self.spec_synthetic.size}, "
            f"{self.spec_synthetic.n_frames} frames) -- results are not meaningful."
        )
