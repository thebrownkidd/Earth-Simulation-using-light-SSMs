"""Reading EarthNet2021 minicubes.

This module is the **only** place that knows the on-disk format. Everything
downstream works in terms of ``[T, C, H, W]`` tensors, so adapting to a new
data source means writing a new reader, not touching the pipeline.

Format
------
A minicube is a compressed ``.npz`` holding:

======================  =====================  ===============================
Array                   Shape                  Contents
======================  =====================  ===============================
``highresdynamic``      ``(128, 128, 7, T)``   Sentinel-2, 20 m, 5-daily
``highresstatic``       ``(128, 128)``         EU-DEM elevation, 20 m
``mesodynamic``         ``(80, 80, 5, T*5)``   E-OBS weather, 1.28 km, daily
``mesostatic``          ``(80, 80)``           Elevation, 1.28 km
======================  =====================  ===============================

``highresdynamic`` channels are ``[blue, green, red, nir, cld, scl, cldmsk]``.
Reflectance is nominally in ``[0, 1]`` and contains NaNs.

Two conventions are inherited from the official ``earthnet`` toolkit, and are
load-bearing for comparability with published numbers:

1. **Imagery is the first four channels; the quality mask is the last.**
   Indexing the mask as ``[-1]`` rather than ``[6]`` is what the toolkit does,
   and it keeps the reader working on test cubes that carry fewer channels.
2. **``cldmsk == 1`` means cloudy.** Validity is therefore ``1 - cldmsk``.
   Getting this backwards trains the model exclusively on clouds, and does so
   silently -- losses still fall.

TinyEarth adds one deliberate deviation: pixels that are NaN in the imagery are
also marked invalid. The toolkit only zero-fills them. Zero-filled NaN is
indistinguishable from genuine zero reflectance, so a model trained against it
learns to predict black where data is missing. Disable with
``nan_is_invalid=False`` when reproducing official EarthNetScore numbers.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Final

import numpy as np
import torch

__all__ = [
    "BAND_NAMES",
    "HIGHRESDYNAMIC",
    "N_IMAGE_CHANNELS",
    "Minicube",
    "MinicubeFormatError",
    "read_minicube",
]

HIGHRESDYNAMIC: Final = "highresdynamic"
"""Name of the Sentinel-2 array inside the ``.npz``."""

HIGHRESSTATIC: Final = "highresstatic"
MESODYNAMIC: Final = "mesodynamic"
MESOSTATIC: Final = "mesostatic"

BAND_NAMES: Final = ("blue", "green", "red", "nir")
"""Human-readable names of the four imagery channels, in index order."""

N_IMAGE_CHANNELS: Final = 4
"""Imagery occupies channels ``0..3``; later channels are quality layers."""

RGB_INDICES: Final = (2, 1, 0)
"""Channel indices producing a true-colour image, for visualisation."""

_MIN_CHANNELS: Final = N_IMAGE_CHANNELS + 1
"""Four imagery channels plus at least the quality mask."""


class MinicubeFormatError(ValueError):
    """Raised when a file does not match the expected minicube layout."""


@dataclass(frozen=True)
class Minicube:
    """A decoded minicube.

    Attributes:
        images: Reflectance, ``[T, C, H, W]``, float32, cleaned to ``[0, 1]``.
        valid: Validity mask, ``[T, 1, H, W]``, float32, where 1 means the pixel
            is usable. 1 everywhere when the source carried no mask channel.
        cube_id: Source filename without its extension.
        source: Absolute path the cube was read from.
    """

    images: torch.Tensor
    valid: torch.Tensor
    cube_id: str
    source: Path

    @property
    def n_frames(self) -> int:
        """Number of timesteps."""
        return int(self.images.shape[0])

    @property
    def n_channels(self) -> int:
        """Number of imagery channels."""
        return int(self.images.shape[1])

    @property
    def spatial_size(self) -> tuple[int, int]:
        """Spatial size as ``(height, width)``."""
        return int(self.images.shape[2]), int(self.images.shape[3])

    def valid_fraction(self, time_slice: slice | None = None) -> float:
        """Fraction of usable pixels, optionally over a subset of frames.

        Args:
            time_slice: Frames to consider. ``None`` uses all frames.

        Returns:
            A value in ``[0, 1]``.
        """
        mask = self.valid if time_slice is None else self.valid[time_slice]
        if mask.numel() == 0:
            return 0.0
        return float(mask.mean().item())


def _to_time_major(array: np.ndarray) -> np.ndarray:
    """Reorder a ``(H, W, C, T)`` array to ``(T, C, H, W)``.

    Args:
        array: Source array in EarthNet's spatial-major layout.

    Returns:
        A time-major view.
    """
    return np.transpose(array, (3, 2, 0, 1))


def read_minicube(
    path: Path | str,
    *,
    nan_is_invalid: bool = True,
    apply_mask: bool = True,
) -> Minicube:
    """Read and clean a minicube from disk.

    Args:
        path: Path to a ``.npz`` minicube.
        nan_is_invalid: Mark NaN imagery pixels invalid in addition to
            zero-filling them. See the module docstring for why this defaults
            to ``True`` and when to turn it off.
        apply_mask: Read the quality mask. When ``False``, :attr:`Minicube.valid`
            is all ones and no masking information is retained.

    Returns:
        The decoded cube.

    Raises:
        FileNotFoundError: If ``path`` does not exist.
        MinicubeFormatError: If the file lacks ``highresdynamic``, or that array
            has the wrong rank or too few channels.
    """
    source = Path(path).resolve()
    if not source.is_file():
        raise FileNotFoundError(f"No minicube at {source}.")

    with np.load(source) as npz:
        if HIGHRESDYNAMIC not in npz:
            available = ", ".join(sorted(npz.files)) or "<empty>"
            raise MinicubeFormatError(
                f"{source.name} has no {HIGHRESDYNAMIC!r} array. Found: {available}."
            )
        raw = np.asarray(npz[HIGHRESDYNAMIC])

    if raw.ndim != 4:
        raise MinicubeFormatError(
            f"{source.name}: expected {HIGHRESDYNAMIC} with 4 dimensions "
            f"(H, W, C, T), got shape {raw.shape}."
        )
    if raw.shape[2] < _MIN_CHANNELS and apply_mask:
        raise MinicubeFormatError(
            f"{source.name}: expected at least {_MIN_CHANNELS} channels "
            f"({N_IMAGE_CHANNELS} imagery + quality mask), got {raw.shape[2]}. "
            "Pass apply_mask=False to read imagery without a mask."
        )
    if raw.shape[2] < N_IMAGE_CHANNELS:
        raise MinicubeFormatError(
            f"{source.name}: expected at least {N_IMAGE_CHANNELS} imagery "
            f"channels, got {raw.shape[2]}."
        )

    time_major = _to_time_major(raw)
    images = np.asarray(time_major[:, :N_IMAGE_CHANNELS], dtype=np.float32)

    nan_pixels = np.isnan(images)
    images = np.nan_to_num(images, nan=0.0, posinf=1.0, neginf=0.0)
    np.clip(images, 0.0, 1.0, out=images)

    if apply_mask:
        # Last channel, following the official toolkit: 1 means cloudy.
        cloudy = np.asarray(time_major[:, -1:], dtype=np.float32)
        cloudy = np.nan_to_num(cloudy, nan=1.0)  # unknown quality -> unusable
        valid = 1.0 - np.clip(cloudy, 0.0, 1.0)
    else:
        valid = np.ones((images.shape[0], 1, *images.shape[2:]), dtype=np.float32)

    if nan_is_invalid:
        # Any channel NaN invalidates the whole pixel.
        usable = (~nan_pixels).all(axis=1, keepdims=True).astype(np.float32)
        valid = valid * usable

    return Minicube(
        images=torch.from_numpy(np.ascontiguousarray(images)),
        valid=torch.from_numpy(np.ascontiguousarray(valid)),
        cube_id=source.stem,
        source=source,
    )
