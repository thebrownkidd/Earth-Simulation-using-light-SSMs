"""Loss interface and masked reduction.

Adding a loss must be a new module plus a config entry, never an edit to the
training loop. Two things make that true:

1. Every loss implements :class:`ForecastLoss`, taking the same
   ``(prediction, target, mask)`` and returning a scalar.
2. :class:`CompositeLoss` combines any number of them with weights, so
   multi-term objectives need no special handling in the trainer.

Masking
-------
Every loss is mask-aware, and this is not optional here. A large share of
EarthNet2021 pixels are cloud-contaminated; optimising against them teaches the
model to predict cloud. :func:`masked_mean` performs the reduction so that
individual losses never reimplement it -- and never get it subtly wrong by, say,
dividing by the full pixel count instead of the valid one, which silently scales
the loss down in proportion to cloudiness and makes cloudy batches contribute
less gradient than they should.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import torch
from torch import nn

from tinyearth.utils.registry import Registry

__all__ = ["LOSSES", "CompositeLoss", "ForecastLoss", "expand_mask", "masked_mean"]

LOSSES: Registry[ForecastLoss] = Registry("loss")
"""Losses selectable by name from a config."""

_EPS = 1e-8


def expand_mask(mask: torch.Tensor | None, reference: torch.Tensor) -> torch.Tensor:
    """Broadcast a validity mask to the shape of ``reference``.

    Args:
        mask: Validity mask, ``[B, K, 1, H, W]`` where 1 means usable, or
            ``None`` for "everything is valid".
        reference: Tensor whose shape the mask must match.

    Returns:
        A mask of the same shape as ``reference``.

    Raises:
        ValueError: If the mask cannot broadcast.
    """
    if mask is None:
        return torch.ones_like(reference)
    if mask.shape == reference.shape:
        return mask
    try:
        return mask.expand_as(reference)
    except RuntimeError as error:
        raise ValueError(
            f"Mask of shape {tuple(mask.shape)} cannot broadcast to " f"{tuple(reference.shape)}."
        ) from error


def masked_mean(values: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
    """Average ``values`` over valid entries only.

    Args:
        values: Per-element values, e.g. absolute errors.
        mask: Validity mask where 1 means usable, or ``None``.

    Returns:
        A scalar. Returns 0 when nothing is valid, rather than NaN -- a fully
        cloudy batch should contribute no gradient, not poison the run.
    """
    if mask is None:
        return values.mean()

    weights = expand_mask(mask, values).to(values.dtype)
    total = weights.sum()
    if float(total) <= 0.0:
        return torch.zeros((), device=values.device, dtype=values.dtype)
    return (values * weights).sum() / total.clamp_min(_EPS)


class ForecastLoss(nn.Module, ABC):
    """A scalar training objective over a forecast.

    Implementations register themselves in :data:`LOSSES`.

    Attributes:
        name: Short identifier used when logging per-term values.
    """

    name: str = "loss"

    @abstractmethod
    def forward(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute the loss.

        Args:
            prediction: Forecast, ``[B, K, C, H, W]``.
            target: Ground truth, ``[B, K, C, H, W]``.
            mask: Validity mask, ``[B, K, 1, H, W]``, or ``None``.

        Returns:
            A scalar loss.
        """
        raise NotImplementedError

    def check_shapes(self, prediction: torch.Tensor, target: torch.Tensor) -> None:
        """Verify that prediction and target agree.

        Args:
            prediction: Forecast tensor.
            target: Ground-truth tensor.

        Raises:
            ValueError: If shapes differ.
        """
        if prediction.shape != target.shape:
            raise ValueError(
                f"{type(self).__name__}: prediction shape {tuple(prediction.shape)} "
                f"does not match target shape {tuple(target.shape)}."
            )


class CompositeLoss(ForecastLoss):
    """A weighted sum of losses.

    Per-term values are exposed through :attr:`last_terms` so the trainer can
    log them without recomputation. Reading the individual terms is how you tell
    a stalled multi-term objective from one where a single term dominates.

    Args:
        terms: Mapping from name to ``(loss, weight)``.

    Raises:
        ValueError: If ``terms`` is empty.
    """

    name = "composite"

    def __init__(self, terms: dict[str, tuple[ForecastLoss, float]]) -> None:
        super().__init__()
        if not terms:
            raise ValueError("CompositeLoss needs at least one term.")

        self.losses = nn.ModuleDict({name: loss for name, (loss, _) in terms.items()})
        self.weights = {name: float(weight) for name, (_, weight) in terms.items()}
        self.last_terms: dict[str, float] = {}

    def forward(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute the weighted sum of all terms.

        Args:
            prediction: Forecast, ``[B, K, C, H, W]``.
            target: Ground truth, ``[B, K, C, H, W]``.
            mask: Validity mask, or ``None``.

        Returns:
            The weighted total.
        """
        self.check_shapes(prediction, target)
        total = torch.zeros((), device=prediction.device, dtype=prediction.dtype)
        terms: dict[str, float] = {}

        for name, loss in self.losses.items():
            value = loss(prediction, target, mask)
            terms[name] = float(value.detach())
            total = total + self.weights[name] * value

        terms["total"] = float(total.detach())
        self.last_terms = terms
        return total

    def extra_repr(self) -> str:
        """Return the term weights."""
        return ", ".join(f"{name}={weight}" for name, weight in self.weights.items())
