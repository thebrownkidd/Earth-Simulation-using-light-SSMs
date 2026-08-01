"""Turning forecasts into images.

The metrics in :mod:`tinyearth.evaluation.metrics` compress a forecast to five
numbers. That is what a comparison needs and what a table should carry, but it
throws away the thing a reader actually wants to judge: *does the prediction
look like the place?* This module holds the pieces that answer that -- band
composition, contrast stretching, and the vegetation index the EarthNet2021
task is really about.

Rendering choices, and why
--------------------------
**One stretch for the whole figure.** Sentinel-2 surface reflectance occupies
the bottom fifth of ``[0, 1]``; displayed raw it is nearly black. Every image
therefore gets a percentile stretch -- but the limits are computed *once*, from
the observed sequence, and reused for the prediction and for every lead time.
Restretching each panel to its own range is the standard way to make a forecast
look better than it is: a prediction that has collapsed to the scene mean would
be rescaled back into a convincing-looking image.

**Error panels share one scale too**, for the same reason, and the scale is
reported in the figure rather than left implicit.

**NDVI is the honest summary.** True-colour composites flatter a forecaster,
because getting the average brightness of a landscape right is easy and looks
convincing. NDVI -- ``(NIR - Red) / (NIR + Red)`` -- is what EarthNet2021 exists
to predict: it tracks vegetation greenness and it is where the seasonal signal
lives. A model can produce a plausible RGB frame and still be wrong about
whether the vegetation greened up, and the NDVI panels will show it.
"""

from __future__ import annotations

import torch

__all__ = [
    "NDVI_RANGE",
    "composite_rgb",
    "ndvi",
    "stretch_limits",
]

_RED, _NIR = 2, 3
_RGB = (2, 1, 0)
_EPSILON = 1e-6

NDVI_RANGE: tuple[float, float] = (-0.2, 0.9)
"""Display range for NDVI.

Fixed rather than data-derived, so that colour means the same thing in every
figure this project produces. Bare soil and water sit near or below zero; dense
canopy approaches 0.9.
"""


def stretch_limits(
    images: torch.Tensor,
    lower_percentile: float = 2.0,
    upper_percentile: float = 98.0,
    mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute per-channel display limits from a reference sequence.

    Percentiles rather than min/max: a single specular pixel or an unmasked
    cloud edge would otherwise set the white point for the whole scene and wash
    everything else grey.

    Args:
        images: Reference imagery, ``[..., C, H, W]``. Normally the observed
            frames, so that predictions are judged against the truth's scale.
        lower_percentile: Percentile mapped to black.
        upper_percentile: Percentile mapped to white.
        mask: Optional validity mask, ``[..., 1, H, W]``, 1 where usable.
            **Pass it whenever the imagery has been cloud-masked.** The mask
            policy blanks cloudy pixels to zero, and on a scene that is a
            quarter cloud those zeros dominate the lower percentile, dragging
            the black point to 0.0 and washing the whole render out to grey.
            Excluding them recovers the true dynamic range of the land.

    Returns:
        ``(low, high)``, each ``[C]``.

    Raises:
        ValueError: If the percentiles are not ordered within ``[0, 100]``.
    """
    if not 0.0 <= lower_percentile < upper_percentile <= 100.0:
        raise ValueError(
            "Percentiles must satisfy 0 <= lower < upper <= 100, got "
            f"{lower_percentile} and {upper_percentile}."
        )

    channels = images.shape[-3]
    flat = images.reshape(-1, channels, images.shape[-2] * images.shape[-1])
    flat = flat.permute(1, 0, 2).reshape(channels, -1).float()

    quantiles = torch.tensor([lower_percentile / 100.0, upper_percentile / 100.0])
    if mask is None:
        limits = torch.quantile(flat, quantiles, dim=1)
        low, high = limits[0], limits[1]
    else:
        keep = mask.expand_as(images)
        keep = keep.reshape(-1, channels, images.shape[-2] * images.shape[-1])
        keep = keep.permute(1, 0, 2).reshape(channels, -1) > 0
        # Per channel, because a fully masked channel has no valid pixels at all
        # and must fall through to the degenerate handling below.
        bounds = [
            (
                torch.quantile(flat[channel][keep[channel]], quantiles)
                if bool(keep[channel].any())
                else torch.zeros(2)
            )
            for channel in range(channels)
        ]
        stacked = torch.stack(bounds, dim=1)
        low, high = stacked[0], stacked[1]

    # A degenerate channel -- constant, or fully masked to zero -- would divide
    # by zero on stretch. Widen it symmetrically about its value so it renders
    # mid-grey, the honest colour for "no contrast here". Widening upward
    # instead would render it black, which reads as real dark ground.
    degenerate = (high - low) < _EPSILON
    centre = (low + high) / 2.0
    low = torch.where(degenerate, centre - 0.5, low)
    high = torch.where(degenerate, centre + 0.5, high)
    return low, high


def composite_rgb(
    images: torch.Tensor,
    low: torch.Tensor,
    high: torch.Tensor,
    gamma: float = 0.85,
) -> torch.Tensor:
    """Build a display-ready true-colour composite.

    Args:
        images: Imagery, ``[..., C, H, W]``, channels ordered
            ``(blue, green, red, nir)``.
        low: Per-channel black point, ``[C]``, from :func:`stretch_limits`.
        high: Per-channel white point, ``[C]``.
        gamma: Tone curve applied after stretching. Below 1 lifts midtones,
            which is conventional for satellite imagery and makes vegetation
            texture legible without changing relative ordering.

    Returns:
        ``[..., H, W, 3]`` in ``[0, 1]``, ready for ``imshow``.

    Raises:
        ValueError: If ``images`` has fewer than three channels.
    """
    if images.shape[-3] < 3:
        raise ValueError(f"Need at least 3 channels for a composite, got {images.shape[-3]}.")

    scaled = (images - low[:, None, None]) / (high - low)[:, None, None]
    scaled = scaled.clamp(0.0, 1.0)
    rgb = scaled[..., _RGB, :, :]
    return rgb.pow(gamma).movedim(-3, -1)


def ndvi(images: torch.Tensor) -> torch.Tensor:
    """Compute the Normalised Difference Vegetation Index.

    ``NDVI = (NIR - Red) / (NIR + Red)``, the standard measure of live green
    vegetation and the quantity EarthNet2021 is fundamentally about.

    Args:
        images: Imagery, ``[..., C, H, W]``, with at least four channels
            ordered ``(blue, green, red, nir)``.

    Returns:
        ``[..., H, W]`` in ``[-1, 1]``.

    Raises:
        ValueError: If ``images`` has fewer than four channels.
    """
    if images.shape[-3] < 4:
        raise ValueError(f"NDVI needs the NIR channel, so 4 channels; got {images.shape[-3]}.")

    red = images[..., _RED, :, :]
    nir = images[..., _NIR, :, :]
    # Both bands are zero wherever the mask policy blanked a cloudy pixel; the
    # epsilon keeps those at 0 rather than NaN, which would poison the mean.
    return (nir - red) / (nir + red + _EPSILON)
