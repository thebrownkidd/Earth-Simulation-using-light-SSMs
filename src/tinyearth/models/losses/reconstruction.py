"""Reconstruction losses.

Phase 3 ships L1 as the primary objective, plus L2 and Charbonnier for
comparison. Phase 6 adds :class:`GDLLoss`, a blur penalty meant to be *added
to* L1 rather than to replace it. SSIM, SAM and multi-scale reconstruction
follow; the shared interface in :mod:`tinyearth.models.losses.base` is what
makes adding a term a new module rather than a change to the trainer.

L1 is the default base term. On satellite imagery L2 over-penalises the rare
large errors that cloud edges and shadows produce, and the usual result is a
model that hedges toward the local mean -- visually, a blurred forecast. L1's
gradient is constant in magnitude, so it is far less sensitive to those
outliers. **Do not replace L1 with L2/MSE to fight blur** -- L2 is documented
to make it worse, not better; use :class:`GDLLoss` alongside L1 instead.
"""

from __future__ import annotations

import torch

from tinyearth.models.losses.base import LOSSES, ForecastLoss, masked_mean

__all__ = ["CharbonnierLoss", "GDLLoss", "L1Loss", "L2Loss"]


@LOSSES.register("l1")
class L1Loss(ForecastLoss):
    """Mean absolute error over valid pixels.

    The Phase 3 default. Robust to the outliers produced by cloud edges, and
    directly interpretable: with ``normalization: identity`` the value is mean
    reflectance error.
    """

    name = "l1"

    def forward(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute masked L1.

        Args:
            prediction: Forecast, ``[B, K, C, H, W]``.
            target: Ground truth, ``[B, K, C, H, W]``.
            mask: Validity mask, or ``None``.

        Returns:
            A scalar loss.
        """
        self.check_shapes(prediction, target)
        return masked_mean((prediction - target).abs(), mask)


@LOSSES.register("l2")
class L2Loss(ForecastLoss):
    """Mean squared error over valid pixels.

    Provided for comparison. Prefer :class:`L1Loss` unless you specifically
    want the mean-seeking behaviour.
    """

    name = "l2"

    def forward(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute masked MSE.

        Args:
            prediction: Forecast, ``[B, K, C, H, W]``.
            target: Ground truth, ``[B, K, C, H, W]``.
            mask: Validity mask, or ``None``.

        Returns:
            A scalar loss.
        """
        self.check_shapes(prediction, target)
        return masked_mean((prediction - target).pow(2), mask)


@LOSSES.register("charbonnier")
class CharbonnierLoss(ForecastLoss):
    """Charbonnier (smooth L1) loss over valid pixels.

    ``sqrt((x - y)^2 + eps^2)``. Behaves like L1 for large errors but is smooth
    at zero, which removes the gradient discontinuity that can make L1 chatter
    once the forecast is already close.

    Args:
        epsilon: Smoothing constant. Smaller values approach exact L1.
    """

    name = "charbonnier"

    def __init__(self, epsilon: float = 1e-3) -> None:
        super().__init__()
        if epsilon <= 0:
            raise ValueError(f"epsilon must be positive, got {epsilon}.")
        self.epsilon = epsilon

    def forward(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute masked Charbonnier loss.

        Args:
            prediction: Forecast, ``[B, K, C, H, W]``.
            target: Ground truth, ``[B, K, C, H, W]``.
            mask: Validity mask, or ``None``.

        Returns:
            A scalar loss.
        """
        self.check_shapes(prediction, target)
        difference = prediction - target
        return masked_mean(torch.sqrt(difference.pow(2) + self.epsilon**2), mask)

    def extra_repr(self) -> str:
        """Return the smoothing constant."""
        return f"epsilon={self.epsilon}"


@LOSSES.register("gdl")
class GDLLoss(ForecastLoss):
    """Gradient-difference loss: a blur penalty, meant to be added to L1.

    Compares the *magnitude* of horizontal and vertical pixel-to-pixel
    differences between prediction and target, rather than comparing raw
    pixel values -- so a prediction that gets the mean right everywhere but
    smooths over edges the target actually has (L1's blind spot) is
    penalised here even though L1 alone would score it well.

    Reference: Mathieu, Couprie & LeCun, "Deep Multi-Scale Video Prediction
    Beyond Mean Square Error," ICLR 2016.

    Masked consistently with every other loss in this module: a gradient
    spans two neighbouring pixels, so it counts as valid only where *both*
    are -- the product of the two shifted masks. A gradient computed across a
    cloud boundary is not a real edge in the data, and would otherwise teach
    the model to reproduce mask artefacts.
    """

    name = "gdl"

    def forward(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute the masked gradient-difference loss.

        Args:
            prediction: Forecast, ``[B, K, C, H, W]``.
            target: Ground truth, ``[B, K, C, H, W]``.
            mask: Validity mask, or ``None``.

        Returns:
            A scalar loss: the horizontal term plus the vertical term. Each
            term is naturally zero (contributes nothing, not NaN) when the
            corresponding spatial dimension has fewer than two pixels to
            take a difference across.
        """
        self.check_shapes(prediction, target)
        total = torch.zeros((), device=prediction.device, dtype=prediction.dtype)

        if prediction.shape[-1] >= 2:
            pred_h = _horizontal_gradient(prediction)
            target_h = _horizontal_gradient(target)
            mask_h = _horizontal_gradient_mask(mask)
            total = total + masked_mean((pred_h.abs() - target_h.abs()).abs(), mask_h)

        if prediction.shape[-2] >= 2:
            pred_v = _vertical_gradient(prediction)
            target_v = _vertical_gradient(target)
            mask_v = _vertical_gradient_mask(mask)
            total = total + masked_mean((pred_v.abs() - target_v.abs()).abs(), mask_v)

        return total


def _horizontal_gradient(x: torch.Tensor) -> torch.Tensor:
    """Pixel-to-pixel difference along the width axis."""
    return x[..., :, 1:] - x[..., :, :-1]


def _vertical_gradient(x: torch.Tensor) -> torch.Tensor:
    """Pixel-to-pixel difference along the height axis."""
    return x[..., 1:, :] - x[..., :-1, :]


def _horizontal_gradient_mask(mask: torch.Tensor | None) -> torch.Tensor | None:
    """Validity mask for a horizontal gradient: both neighbours must be valid."""
    if mask is None:
        return None
    return mask[..., :, 1:] * mask[..., :, :-1]


def _vertical_gradient_mask(mask: torch.Tensor | None) -> torch.Tensor | None:
    """Validity mask for a vertical gradient: both neighbours must be valid."""
    if mask is None:
        return None
    return mask[..., 1:, :] * mask[..., :-1, :]
