"""Reference forecasts that use no learned parameters.

A forecast-quality number in isolation says nothing. ``MAE = 0.05`` is
meaningless until it sits beside what the same metric gives for a forecast that
did no work, because Earth surface imagery is dominated by a static component:
a river is in the same place in 100 days, and a field is roughly the colour it
was. A model can score well while having learned nothing but "return the input".

Two references bracket that, and both are free to compute:

``persistence``
    Repeat the last observed frame. The standard baseline in Earth surface
    forecasting and a notoriously strong one -- over a 100-day horizon most
    pixels genuinely do not change much, so this is the number a learned model
    has to beat to have justified itself.

``climatology``
    Repeat the mean of the observed context. Smoother than persistence, so it
    is favoured by squared-error metrics and penalised by perceptual ones. Where
    a model beats persistence but not climatology, it has learned to blur rather
    than to forecast -- which is the characteristic failure of an
    under-trained video predictor, and worth being able to detect.

Neither is a competitor. They are the axis the learned results are read against.
"""

from __future__ import annotations

import torch

__all__ = ["REFERENCE_FORECASTS", "climatology_forecast", "persistence_forecast"]


def persistence_forecast(
    images: torch.Tensor,
    horizon: int,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Repeat the last observed frame across the forecast horizon.

    Args:
        images: Observed history, ``[B, T, C, H, W]``.
        horizon: Frames to emit.
        mask: Optional validity mask for the history, ``[B, T, 1, H, W]``, where
            1 marks a usable pixel. When given, each pixel persists from the
            most recent frame in which it was actually *observed*, rather than
            from a frame where cloud had blanked it to zero. Without this the
            baseline is unfairly weak precisely where the data is cloudiest.

    Returns:
        Forecast frames, ``[B, K, C, H, W]``.

    Raises:
        ValueError: If ``images`` is not rank 5 or ``horizon`` is not positive.
    """
    _check(images, horizon)

    if mask is None:
        latest = images[:, -1]
    else:
        # Index of the most recent valid observation per pixel. Where a pixel
        # was never valid, argmax over an all-zero vector returns 0, which
        # falls back to the first frame -- the best available guess.
        steps = images.shape[1]
        weights = mask[:, :, 0] * torch.arange(steps, device=images.device).view(1, steps, 1, 1)
        newest = weights.argmax(dim=1, keepdim=True)  # [B, 1, H, W]
        index = newest.unsqueeze(2).expand(-1, -1, images.shape[2], -1, -1)
        latest = images.gather(1, index)[:, 0]

    return latest.unsqueeze(1).expand(-1, horizon, -1, -1, -1).contiguous()


def climatology_forecast(
    images: torch.Tensor,
    horizon: int,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Repeat the mean of the observed context across the forecast horizon.

    Args:
        images: Observed history, ``[B, T, C, H, W]``.
        horizon: Frames to emit.
        mask: Optional validity mask, ``[B, T, 1, H, W]``. When given the mean is
            taken over valid frames only, so cloud-blanked zeros do not drag
            the average toward black.

    Returns:
        Forecast frames, ``[B, K, C, H, W]``.

    Raises:
        ValueError: If ``images`` is not rank 5 or ``horizon`` is not positive.
    """
    _check(images, horizon)

    if mask is None:
        average = images.mean(dim=1)
    else:
        weights = mask.expand_as(images)
        observed = weights.sum(dim=1)
        # A pixel valid in no frame would divide by zero; fall back to the
        # unmasked mean there rather than emitting NaN.
        average = torch.where(
            observed > 0,
            (images * weights).sum(dim=1) / observed.clamp(min=1.0),
            images.mean(dim=1),
        )

    return average.unsqueeze(1).expand(-1, horizon, -1, -1, -1).contiguous()


def _check(images: torch.Tensor, horizon: int) -> None:
    """Validate the shared argument contract.

    Raises:
        ValueError: If the input rank is wrong or ``horizon`` is not positive.
    """
    if images.ndim != 5:
        raise ValueError(f"images must be [B, T, C, H, W], got shape {tuple(images.shape)}.")
    if horizon < 1:
        raise ValueError(f"horizon must be >= 1, got {horizon}.")


REFERENCE_FORECASTS = {
    "persistence": persistence_forecast,
    "climatology": climatology_forecast,
}
"""Parameter-free forecasts, selectable by name."""
