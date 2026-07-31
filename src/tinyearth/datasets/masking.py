"""Cloud-mask policy.

Optical satellite imagery is heavily contaminated by cloud -- across
EarthNet2021, a large share of pixels are unusable. How that is handled is a
modelling decision with real consequences, so it is made explicit here rather
than buried in the dataset class.

Two independent knobs:

**Fill policy** (:class:`MaskPolicy`) -- what the model *sees* where data is
missing. The pixel value has to be something, and every choice is a lie of some
kind; the question is which lie is least harmful.

**Window filtering** (:func:`passes_validity_threshold`) -- whether a training
sample is used at all. A window whose target is 95% cloud teaches the model
almost nothing about vegetation and a great deal about cloud.

The masks themselves are always propagated to the sample, so that Phase 3
losses can exclude invalid pixels from the objective regardless of fill policy.
That is the part that actually matters: a masked loss makes the fill value
mostly irrelevant for the target, though it still affects the encoder input.
"""

from __future__ import annotations

from enum import StrEnum

import torch

__all__ = [
    "MaskPolicy",
    "apply_mask_policy",
    "masked_fraction",
    "passes_validity_threshold",
]


class MaskPolicy(StrEnum):
    """How invalid pixels are filled before the model sees them.

    Attributes:
        KEEP: Leave the original values. Cloudy pixels stay bright, which is
            honest about the observation but injects high-frequency structure
            the model may waste capacity on.
        ZERO: Replace invalid pixels with 0. Simple, but indistinguishable from
            genuine dark reflectance such as deep water.
        MEAN: Replace invalid pixels with the per-channel mean of the valid
            pixels in the same frame. Keeps frame statistics stable and avoids
            inventing an out-of-distribution value. Falls back to zero when a
            frame has no valid pixels at all.
    """

    KEEP = "keep"
    ZERO = "zero"
    MEAN = "mean"


def apply_mask_policy(
    images: torch.Tensor,
    valid: torch.Tensor,
    policy: MaskPolicy = MaskPolicy.ZERO,
) -> torch.Tensor:
    """Fill invalid pixels according to ``policy``.

    Args:
        images: Imagery, ``[T, C, H, W]``.
        valid: Validity mask, ``[T, 1, H, W]``, where 1 means usable.
        policy: Fill policy.

    Returns:
        A new tensor with invalid pixels filled. The input is not modified.

    Raises:
        ValueError: If shapes are incompatible.
    """
    if images.ndim != 4:
        raise ValueError(f"images must be [T, C, H, W], got shape {tuple(images.shape)}.")
    if valid.shape[0] != images.shape[0] or valid.shape[-2:] != images.shape[-2:]:
        raise ValueError(
            f"valid {tuple(valid.shape)} is not broadcastable against images "
            f"{tuple(images.shape)}; expected [T, 1, H, W]."
        )

    if policy is MaskPolicy.KEEP:
        return images.clone()

    mask = valid.to(images.dtype)

    if policy is MaskPolicy.ZERO:
        return images * mask

    # MEAN: per-frame, per-channel average over valid pixels only.
    valid_count = mask.sum(dim=(-2, -1), keepdim=True)  # [T, 1, 1, 1]
    valid_sum = (images * mask).sum(dim=(-2, -1), keepdim=True)  # [T, C, 1, 1]
    # Frames with no valid pixels fall back to zero rather than dividing by zero.
    frame_mean = torch.where(
        valid_count > 0,
        valid_sum / valid_count.clamp_min(1.0),
        torch.zeros_like(valid_sum),
    )
    return images * mask + frame_mean * (1.0 - mask)


def masked_fraction(valid: torch.Tensor) -> float:
    """Return the fraction of pixels marked invalid.

    Args:
        valid: Validity mask where 1 means usable.

    Returns:
        A value in ``[0, 1]``. ``0.0`` for an empty mask.
    """
    if valid.numel() == 0:
        return 0.0
    return float(1.0 - valid.mean().item())


def passes_validity_threshold(valid: torch.Tensor, min_valid_fraction: float) -> bool:
    """Decide whether a window is clean enough to train on.

    Applied to the **target** frames. Filtering on the history instead would
    discard exactly the samples where forecasting from partial context is most
    valuable; filtering on the target avoids optimising against a supervision
    signal that is mostly cloud.

    Args:
        valid: Validity mask for the target frames, where 1 means usable.
        min_valid_fraction: Minimum fraction of usable pixels, in ``[0, 1]``.
            ``0.0`` accepts everything.

    Returns:
        ``True`` if the window should be kept.

    Raises:
        ValueError: If ``min_valid_fraction`` lies outside ``[0, 1]``.
    """
    if not 0.0 <= min_valid_fraction <= 1.0:
        raise ValueError(f"min_valid_fraction must be in [0, 1], got {min_valid_fraction}.")
    if min_valid_fraction == 0.0:
        return True
    if valid.numel() == 0:
        return False
    return float(valid.mean().item()) >= min_valid_fraction
