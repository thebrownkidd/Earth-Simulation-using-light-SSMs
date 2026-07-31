"""Forecast-quality metrics.

All five metrics are **mask-aware**. On EarthNet2021 a large share of pixels are
cloud-contaminated, and scoring them measures cloud prediction rather than
vegetation forecasting. An unmasked PSNR on this dataset can be several dB off,
in whichever direction the cloud happens to fall.

Metrics are implemented here rather than taken from ``torchmetrics`` for one
reason: masked variants. Applying a library SSIM to a zero-filled image scores
the fill value as if it were data, and the resulting number is not comparable
across samples with different cloud cover.

Accumulation
------------
:class:`MetricAccumulator` sums over batches and divides once at the end, so the
reported value is the mean over *pixels*, not the mean of per-batch means. Those
differ whenever batches have unequal valid-pixel counts -- which, with variable
cloud cover, is always.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import torch
import torch.nn.functional as F

from tinyearth.models.losses.base import expand_mask

__all__ = [
    "MetricAccumulator",
    "forecast_metrics",
    "masked_mae",
    "masked_psnr",
    "masked_rmse",
    "masked_sam",
    "masked_ssim",
]

_EPS = 1e-8


def _masked_reduce(values: torch.Tensor, mask: torch.Tensor | None) -> tuple[float, float]:
    """Return the masked sum and the count of valid entries.

    Args:
        values: Per-element values.
        mask: Validity mask, or ``None``.

    Returns:
        ``(sum, count)`` as floats.
    """
    if mask is None:
        return float(values.sum()), float(values.numel())
    weights = expand_mask(mask, values).to(values.dtype)
    return float((values * weights).sum()), float(weights.sum())


def masked_mae(
    prediction: torch.Tensor, target: torch.Tensor, mask: torch.Tensor | None = None
) -> float:
    """Mean absolute error over valid pixels.

    Args:
        prediction: Forecast, ``[B, K, C, H, W]``.
        target: Ground truth, same shape.
        mask: Validity mask, or ``None``.

    Returns:
        The MAE, or 0.0 when nothing is valid.
    """
    total, count = _masked_reduce((prediction - target).abs(), mask)
    return total / count if count > 0 else 0.0


def masked_rmse(
    prediction: torch.Tensor, target: torch.Tensor, mask: torch.Tensor | None = None
) -> float:
    """Root mean squared error over valid pixels.

    Args:
        prediction: Forecast, ``[B, K, C, H, W]``.
        target: Ground truth, same shape.
        mask: Validity mask, or ``None``.

    Returns:
        The RMSE, or 0.0 when nothing is valid.
    """
    total, count = _masked_reduce((prediction - target).pow(2), mask)
    return math.sqrt(total / count) if count > 0 else 0.0


def masked_psnr(
    prediction: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor | None = None,
    data_range: float = 1.0,
) -> float:
    """Peak signal-to-noise ratio over valid pixels.

    Args:
        prediction: Forecast, ``[B, K, C, H, W]``.
        target: Ground truth, same shape.
        mask: Validity mask, or ``None``.
        data_range: Peak signal value; 1.0 for reflectance in ``[0, 1]``.

    Returns:
        PSNR in dB. Returns ``inf`` for a perfect match, which callers should
        exclude from averages rather than propagate.
    """
    total, count = _masked_reduce((prediction - target).pow(2), mask)
    if count <= 0:
        return 0.0
    mse = total / count
    if mse <= _EPS:
        return float("inf")
    return 10.0 * math.log10(data_range**2 / mse)


def _gaussian_window(window_size: int, sigma: float, device: torch.device) -> torch.Tensor:
    """Build a normalised 2-D Gaussian kernel.

    Args:
        window_size: Kernel size; should be odd.
        sigma: Standard deviation in pixels.
        device: Device to allocate on.

    Returns:
        ``[1, 1, window_size, window_size]``.
    """
    coords = torch.arange(window_size, dtype=torch.float32, device=device)
    coords = coords - (window_size - 1) / 2.0
    gaussian = torch.exp(-coords.pow(2) / (2 * sigma**2))
    gaussian = gaussian / gaussian.sum()
    kernel = gaussian.outer(gaussian)
    return kernel.unsqueeze(0).unsqueeze(0)


def masked_ssim(
    prediction: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor | None = None,
    data_range: float = 1.0,
    window_size: int = 11,
    sigma: float = 1.5,
) -> float:
    """Structural similarity over valid pixels.

    Follows Wang et al. (2004): Gaussian-weighted local statistics with the
    standard stabilising constants.

    Masking is applied to the **SSIM map** rather than to the inputs. Masking
    the inputs first would let the fill value bleed into neighbouring windows
    through the Gaussian, corrupting scores for pixels that are themselves
    perfectly valid.

    Args:
        prediction: Forecast, ``[B, K, C, H, W]``.
        target: Ground truth, same shape.
        mask: Validity mask, ``[B, K, 1, H, W]``, or ``None``.
        data_range: Peak signal value.
        window_size: Gaussian window size.
        sigma: Gaussian standard deviation.

    Returns:
        Mean SSIM in ``[-1, 1]``; higher is better. Returns 0.0 when the frames
        are smaller than the window, since SSIM is undefined there.
    """
    if prediction.shape != target.shape:
        raise ValueError(
            f"SSIM: prediction {tuple(prediction.shape)} != target {tuple(target.shape)}."
        )

    batch, steps, channels, height, width = prediction.shape
    if min(height, width) < window_size:
        return 0.0

    flat_pred = prediction.reshape(batch * steps, channels, height, width)
    flat_target = target.reshape(batch * steps, channels, height, width)

    kernel = _gaussian_window(window_size, sigma, prediction.device)
    kernel = kernel.expand(channels, 1, window_size, window_size).to(flat_pred.dtype)
    padding = window_size // 2

    def blur(tensor: torch.Tensor) -> torch.Tensor:
        return F.conv2d(tensor, kernel, padding=padding, groups=channels)

    mu_pred = blur(flat_pred)
    mu_target = blur(flat_target)
    mu_pred_sq, mu_target_sq = mu_pred.pow(2), mu_target.pow(2)
    mu_cross = mu_pred * mu_target

    sigma_pred = blur(flat_pred.pow(2)) - mu_pred_sq
    sigma_target = blur(flat_target.pow(2)) - mu_target_sq
    sigma_cross = blur(flat_pred * flat_target) - mu_cross

    c1 = (0.01 * data_range) ** 2
    c2 = (0.03 * data_range) ** 2
    ssim_map = ((2 * mu_cross + c1) * (2 * sigma_cross + c2)) / (
        (mu_pred_sq + mu_target_sq + c1) * (sigma_pred + sigma_target + c2)
    )
    ssim_map = ssim_map.reshape(batch, steps, channels, height, width)

    total, count = _masked_reduce(ssim_map, mask)
    return total / count if count > 0 else 0.0


def masked_sam(
    prediction: torch.Tensor, target: torch.Tensor, mask: torch.Tensor | None = None
) -> float:
    """Spectral angle mapper over valid pixels, in degrees.

    SAM measures the angle between the predicted and true spectral vectors at
    each pixel. Because it is invariant to vector magnitude, it isolates
    *spectral shape* from brightness -- which is what distinguishes a genuinely
    correct land-cover prediction from one that merely has the right average
    illumination. Lower is better; 0 is a perfect spectral match.

    Computed as ``2 * atan2(|â - b̂|, |â + b̂|)`` on the unit-normalised spectra
    rather than the textbook ``acos(â · b̂)``. The two are equivalent in exact
    arithmetic, but ``acos`` has unbounded derivative at 1, so in float32 an
    identical pair scores a spurious ~0.006 degrees instead of 0. That is small
    but it is noise in a reported metric, and it sits exactly where a good model
    lives. The half-angle form is exact near zero.

    Args:
        prediction: Forecast, ``[B, K, C, H, W]``.
        target: Ground truth, same shape.
        mask: Validity mask, ``[B, K, 1, H, W]``, or ``None``.

    Returns:
        Mean spectral angle in degrees.
    """
    if prediction.shape != target.shape:
        raise ValueError(
            f"SAM: prediction {tuple(prediction.shape)} != target {tuple(target.shape)}."
        )

    # Reduce over the channel axis: [B, K, C, H, W] -> [B, K, 1, H, W]
    pred_norm = prediction.norm(dim=2, keepdim=True)
    target_norm = target.norm(dim=2, keepdim=True)
    defined = (pred_norm > _EPS) & (target_norm > _EPS)

    unit_pred = prediction / pred_norm.clamp_min(_EPS)
    unit_target = target / target_norm.clamp_min(_EPS)

    difference = (unit_pred - unit_target).norm(dim=2, keepdim=True)
    total_ = (unit_pred + unit_target).norm(dim=2, keepdim=True)
    angles = torch.rad2deg(2.0 * torch.atan2(difference, total_))

    # Pixels with a zero-norm spectrum have no defined angle; score them 0.
    angles = torch.where(defined, angles, torch.zeros_like(angles))

    total, count = _masked_reduce(angles, mask)
    return total / count if count > 0 else 0.0


def forecast_metrics(
    prediction: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor | None = None,
    data_range: float = 1.0,
) -> dict[str, float]:
    """Compute all forecast-quality metrics for one batch.

    Args:
        prediction: Forecast, ``[B, K, C, H, W]``.
        target: Ground truth, same shape.
        mask: Validity mask, or ``None``.
        data_range: Peak signal value for PSNR and SSIM.

    Returns:
        Metrics keyed ``mae``, ``rmse``, ``psnr``, ``ssim`` and ``sam``.
    """
    return {
        "mae": masked_mae(prediction, target, mask),
        "rmse": masked_rmse(prediction, target, mask),
        "psnr": masked_psnr(prediction, target, mask, data_range),
        "ssim": masked_ssim(prediction, target, mask, data_range),
        "sam": masked_sam(prediction, target, mask),
    }


@dataclass
class MetricAccumulator:
    """Accumulates metrics across batches, weighted by sample count.

    Averaging per-batch means would weight a partial final batch as heavily as a
    full one. Weighting by sample count avoids that.

    Attributes:
        totals: Running weighted sums.
        count: Running total weight.
    """

    totals: dict[str, float] = field(default_factory=dict)
    count: float = 0.0

    def update(self, metrics: dict[str, float], weight: float = 1.0) -> None:
        """Add one batch's metrics.

        Args:
            metrics: Metric values for the batch.
            weight: Batch weight, normally the batch size.
        """
        if weight <= 0:
            return
        for key, value in metrics.items():
            if math.isinf(value) or math.isnan(value):
                # A perfect-match PSNR of inf would destroy the running mean.
                continue
            self.totals[key] = self.totals.get(key, 0.0) + value * weight
        self.count += weight

    def compute(self) -> dict[str, float]:
        """Return the weighted means.

        Returns:
            Averaged metrics; empty when nothing was accumulated.
        """
        if self.count <= 0:
            return {}
        return {key: total / self.count for key, total in self.totals.items()}

    def reset(self) -> None:
        """Clear all accumulated state."""
        self.totals.clear()
        self.count = 0.0
