"""Reconstruction losses.

Phase 3 ships L1 as the primary objective, plus L2 and Charbonnier for
comparison. SSIM, SAM and multi-scale reconstruction follow; the shared
interface in :mod:`tinyearth.models.losses.base` is what makes adding them a
new module rather than a change to the trainer.

L1 is the default. On satellite imagery L2 over-penalises the rare large errors
that cloud edges and shadows produce, and the usual result is a model that
hedges toward the local mean -- visually, a blurred forecast. L1's gradient is
constant in magnitude, so it is far less sensitive to those outliers.
"""

from __future__ import annotations

import torch

from tinyearth.models.losses.base import LOSSES, ForecastLoss, masked_mean

__all__ = ["CharbonnierLoss", "L1Loss", "L2Loss"]


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
