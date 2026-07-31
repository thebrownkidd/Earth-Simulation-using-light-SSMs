"""Convolutional image encoder.

Maps each frame independently to a latent grid. Frames are folded into the batch
dimension, so the encoder carries **no temporal parameters at all** -- every bit
of temporal modelling capacity lives in the backbone under study. Mixing time
here would make the comparison between backbones incoherent.

Downsampling reduces the spatial size the backbone must process, which matters
disproportionately: a ConvLSTM's cost is quadratic in the spatial extent, so a
factor-4 reduction is a factor-16 saving in the component being measured.
"""

from __future__ import annotations

import torch
from torch import nn

from tinyearth.models.base import Encoder, check_latent_shape
from tinyearth.models.layers import conv_block
from tinyearth.utils.registry import Registry

__all__ = ["ENCODERS", "CNNEncoder"]

ENCODERS: Registry[Encoder] = Registry("encoder")
"""Image encoders selectable by name from a config."""


@ENCODERS.register("cnn")
class CNNEncoder(Encoder):
    """A strided convolutional encoder applied per frame.

    Args:
        in_channels: Input imagery channels.
        latent_dim: Output channel count, ``D``.
        base_channels: Channels after the stem, doubled at each downsampling
            stage until ``latent_dim`` is projected.
        depth: Number of stride-2 stages, so ``downsample == 2 ** depth``.
        norm: Normalisation kind.
        activation: Activation name.

    Raises:
        ValueError: If ``depth`` is negative or channel counts are non-positive.
    """

    def __init__(
        self,
        in_channels: int = 4,
        latent_dim: int = 128,
        base_channels: int = 32,
        depth: int = 2,
        norm: str = "group",
        activation: str = "gelu",
    ) -> None:
        super().__init__()
        if depth < 0:
            raise ValueError(f"depth must be >= 0, got {depth}.")
        for name, value in (
            ("in_channels", in_channels),
            ("latent_dim", latent_dim),
            ("base_channels", base_channels),
        ):
            if value < 1:
                raise ValueError(f"{name} must be >= 1, got {value}.")

        self.in_channels = in_channels
        self.latent_dim = latent_dim
        self.downsample = 2**depth

        stages: list[nn.Module] = [
            conv_block(in_channels, base_channels, norm=norm, activation=activation)
        ]
        channels = base_channels
        for _ in range(depth):
            stages.append(
                conv_block(channels, channels * 2, stride=2, norm=norm, activation=activation)
            )
            channels *= 2

        self.stages = nn.Sequential(*stages)
        # A 1x1 projection to the latent dimension keeps `latent_dim`
        # independent of `base_channels` and `depth`, so the three can be swept
        # without one silently constraining another.
        self.project = nn.Conv2d(channels, latent_dim, kernel_size=1)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """Encode a sequence of frames.

        Args:
            images: ``[B, T, C, H, W]``.

        Returns:
            Latent grid, ``[B, T, D, h, w]``.

        Raises:
            ValueError: If the input rank or channel count is wrong.
        """
        check_latent_shape(images, "images")
        batch, time, channels, height, width = images.shape
        if channels != self.in_channels:
            raise ValueError(
                f"Encoder expects {self.in_channels} channels, got {channels}. "
                "Check data.channels against model.encoder.in_channels."
            )

        flat = images.reshape(batch * time, channels, height, width)
        latents: torch.Tensor = self.project(self.stages(flat))
        return latents.reshape(batch, time, self.latent_dim, *latents.shape[-2:])

    def extra_repr(self) -> str:
        """Return a summary for ``print(model)``."""
        return (
            f"in_channels={self.in_channels}, latent_dim={self.latent_dim}, "
            f"downsample={self.downsample}"
        )
