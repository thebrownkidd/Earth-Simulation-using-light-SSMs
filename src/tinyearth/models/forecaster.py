"""The assembled forecasting model.

Composes the three components into the fixed architecture::

    encoder  ->  temporal backbone  ->  decoder  ->  forecast

The class holds no modelling logic of its own -- deliberately. Anything it did
would be shared machinery outside the component under study, and would blur the
attribution of a result to the backbone.

Its one substantive job is the **parameter breakdown**. Because the research
question is about parameter efficiency, knowing that a 10M-parameter model
spends 8M in the backbone and 2M in the encoder/decoder is not a diagnostic
nicety; it is the result.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from tinyearth.models.base import Decoder, Encoder, TemporalBackbone, check_latent_shape

__all__ = ["Forecaster", "ParameterBreakdown", "count_parameters"]


def count_parameters(module: nn.Module, trainable_only: bool = True) -> int:
    """Count parameters in a module.

    Args:
        module: Module to count.
        trainable_only: Count only parameters with ``requires_grad``. Set
            ``False`` to include frozen weights, which matters for the frozen
            foundation-encoder comparison, where the honest headline number
            includes weights the model uses but does not train.

    Returns:
        The parameter count.
    """
    return sum(p.numel() for p in module.parameters() if p.requires_grad or not trainable_only)


@dataclass(frozen=True)
class ParameterBreakdown:
    """Parameters attributed to each component.

    Attributes:
        encoder: Encoder parameters.
        backbone: Temporal backbone parameters.
        decoder: Decoder parameters.
        total: Sum of the three.
    """

    encoder: int
    backbone: int
    decoder: int
    total: int

    @property
    def backbone_fraction(self) -> float:
        """Share of parameters in the component under study, in ``[0, 1]``.

        Scaling is supposed to happen primarily inside the backbone. A low value
        here means a size sweep is mostly varying fixed machinery, and the
        scaling result would not mean what it appears to.
        """
        return self.backbone / self.total if self.total else 0.0

    def as_dict(self) -> dict[str, float]:
        """Flatten for logging and metric tracking.

        Returns:
            Counts plus the derived backbone fraction and millions-of-parameters.
        """
        return {
            "params/encoder": float(self.encoder),
            "params/backbone": float(self.backbone),
            "params/decoder": float(self.decoder),
            "params/total": float(self.total),
            "params/total_millions": self.total / 1e6,
            "params/backbone_fraction": self.backbone_fraction,
        }

    def format_table(self) -> str:
        """Render a human-readable breakdown.

        Returns:
            A multi-line table.
        """
        rows = [
            ("encoder", self.encoder),
            ("backbone", self.backbone),
            ("decoder", self.decoder),
            ("total", self.total),
        ]
        lines = [f"{'component':<12} {'params':>12} {'share':>8}"]
        for name, value in rows:
            share = value / self.total if self.total else 0.0
            lines.append(f"{name:<12} {value:>12,} {share:>7.1%}")
        return "\n".join(lines)


class Forecaster(nn.Module):
    """Encoder, temporal backbone and decoder assembled into a forecaster.

    Args:
        encoder: Maps frames to a latent grid.
        backbone: Maps a latent history to a latent forecast.
        decoder: Maps the latent forecast back to frames.
        horizon: Default number of frames to forecast. Can be overridden per
            call, which is what makes the horizon sweep possible without
            rebuilding the model.

    Raises:
        ValueError: If the components disagree on the latent dimension, or if
            the decoder's upsampling does not undo the encoder's downsampling.
            Both would otherwise surface as confusing shape errors mid-training.
    """

    def __init__(
        self,
        encoder: Encoder,
        backbone: TemporalBackbone,
        decoder: Decoder,
        horizon: int = 1,
    ) -> None:
        super().__init__()
        if not (encoder.latent_dim == backbone.latent_dim == decoder.latent_dim):
            raise ValueError(
                "Components disagree on latent_dim: "
                f"encoder={encoder.latent_dim}, backbone={backbone.latent_dim}, "
                f"decoder={decoder.latent_dim}."
            )
        if encoder.downsample != decoder.upsample:
            raise ValueError(
                f"Encoder downsamples by {encoder.downsample} but decoder upsamples by "
                f"{decoder.upsample}; the forecast would not match the input resolution."
            )
        if horizon < 1:
            raise ValueError(f"horizon must be >= 1, got {horizon}.")

        self.encoder = encoder
        self.backbone = backbone
        self.decoder = decoder
        self.horizon = horizon

    def forward(self, images: torch.Tensor, horizon: int | None = None) -> torch.Tensor:
        """Forecast future frames.

        Args:
            images: History frames, ``[B, T, C, H, W]``.
            horizon: Frames to forecast. Defaults to :attr:`horizon`.

        Returns:
            Forecast frames, ``[B, K, C, H, W]``.
        """
        check_latent_shape(images, "images")
        steps = self.horizon if horizon is None else horizon
        latents, skips = self.encoder.forward_with_skips(images)
        forecast_latents: torch.Tensor = self.backbone(latents, steps)
        forecast: torch.Tensor = self.decoder(forecast_latents, skips)
        return forecast

    def parameter_breakdown(self, trainable_only: bool = True) -> ParameterBreakdown:
        """Attribute parameters to each component.

        Args:
            trainable_only: Count only trainable parameters.

        Returns:
            The breakdown.
        """
        encoder = count_parameters(self.encoder, trainable_only)
        backbone = count_parameters(self.backbone, trainable_only)
        decoder = count_parameters(self.decoder, trainable_only)
        return ParameterBreakdown(
            encoder=encoder,
            backbone=backbone,
            decoder=decoder,
            total=encoder + backbone + decoder,
        )

    def extra_repr(self) -> str:
        """Return a summary for ``print(model)``."""
        breakdown = self.parameter_breakdown()
        return f"horizon={self.horizon}, params={breakdown.total:,}"
