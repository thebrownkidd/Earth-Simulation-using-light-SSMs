"""Reconstruction losses.

Adding a loss is a new module plus a config entry -- never an edit to the
training loop. Every loss implements
:class:`~tinyearth.models.losses.base.ForecastLoss` and registers itself in
:data:`LOSSES`; :class:`~tinyearth.models.losses.base.CompositeLoss` combines
any number of them with weights.

All losses are mask-aware. A large share of EarthNet2021 pixels are cloudy, and
optimising against them teaches the model to predict cloud.

Phase 3 ships L1 (the default), L2 and Charbonnier. SSIM, SAM and multi-scale
reconstruction follow.
"""

from __future__ import annotations

from tinyearth.models.losses.base import (
    LOSSES,
    CompositeLoss,
    ForecastLoss,
    expand_mask,
    masked_mean,
)
from tinyearth.models.losses.reconstruction import CharbonnierLoss, L1Loss, L2Loss

__all__ = [
    "LOSSES",
    "CharbonnierLoss",
    "CompositeLoss",
    "ForecastLoss",
    "L1Loss",
    "L2Loss",
    "expand_mask",
    "masked_mean",
]
