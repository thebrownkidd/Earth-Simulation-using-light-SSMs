"""Convolutional decoder.

Mirrors :class:`~tinyearth.models.encoders.cnn.CNNEncoder`, decoding each
forecast frame independently. As with the encoder, it holds no temporal
parameters.

Upsampling is nearest-neighbour followed by a convolution, rather than
:class:`~torch.nn.ConvTranspose2d`. Transposed convolutions produce
checkerboard artefacts at these kernel/stride ratios, and in a forecasting task
those artefacts are easy to mistake for genuine spatial structure when reading
result figures.

The output activation is a sigmoid, matching reflectance in ``[0, 1]``. Pair it
with ``normalization: identity``. Under ``standardize`` the targets are no
longer bounded, so set ``output_activation: none`` -- the mismatch would
otherwise clamp the model to a range it can never reach, and the loss would
plateau for reasons that look like an optimisation failure.
"""

from __future__ import annotations

import torch
from torch import nn

from tinyearth.models.base import Decoder, check_latent_shape
from tinyearth.models.layers import build_activation, build_norm, conv_block
from tinyearth.utils.registry import Registry

__all__ = ["DECODERS", "CNNDecoder"]

DECODERS: Registry[Decoder] = Registry("decoder")
"""Decoders selectable by name from a config."""

_OUTPUT_ACTIVATIONS = {"sigmoid", "none"}


@DECODERS.register("cnn")
class CNNDecoder(Decoder):
    """A nearest-neighbour upsampling decoder applied per frame.

    Args:
        out_channels: Output imagery channels.
        latent_dim: Input channel count, ``D``.
        base_channels: Channel count at full resolution.
        depth: Number of upsampling stages; must match the encoder's depth.
        norm: Normalisation kind.
        activation: Hidden activation name.
        output_activation: ``"sigmoid"`` for reflectance in ``[0, 1]``, or
            ``"none"`` for unbounded output under standardised targets.

    Raises:
        ValueError: If ``depth`` is negative or ``output_activation`` is unknown.
    """

    def __init__(
        self,
        out_channels: int = 4,
        latent_dim: int = 128,
        base_channels: int = 32,
        depth: int = 2,
        norm: str = "group",
        activation: str = "gelu",
        output_activation: str = "sigmoid",
    ) -> None:
        super().__init__()
        if depth < 0:
            raise ValueError(f"depth must be >= 0, got {depth}.")
        if output_activation not in _OUTPUT_ACTIVATIONS:
            raise ValueError(
                f"Unknown output_activation {output_activation!r}. "
                f"Expected one of {sorted(_OUTPUT_ACTIVATIONS)}."
            )

        self.out_channels = out_channels
        self.latent_dim = latent_dim
        self.upsample = 2**depth

        channels = base_channels * (2**depth)
        self.project = nn.Conv2d(latent_dim, channels, kernel_size=1)

        stages: list[nn.Module] = []
        for _ in range(depth):
            stages.extend(
                (
                    nn.Upsample(scale_factor=2, mode="nearest"),
                    conv_block(channels, channels // 2, norm=norm, activation=activation),
                )
            )
            channels //= 2
        self.stages = nn.Sequential(*stages)

        self.refine = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            build_norm(norm, channels),
            build_activation(activation),
        )
        self.head = nn.Conv2d(channels, out_channels, kernel_size=1)
        self.output_activation: nn.Module = (
            nn.Sigmoid() if output_activation == "sigmoid" else nn.Identity()
        )

    def forward(self, latents: torch.Tensor) -> torch.Tensor:
        """Decode a latent sequence to frames.

        Args:
            latents: ``[B, K, D, h, w]``.

        Returns:
            Frames, ``[B, K, C, H, W]``.

        Raises:
            ValueError: If the input rank or channel count is wrong.
        """
        check_latent_shape(latents, "latents")
        batch, steps, channels, height, width = latents.shape
        if channels != self.latent_dim:
            raise ValueError(
                f"Decoder expects latent_dim={self.latent_dim}, got {channels}. "
                "Encoder, backbone and decoder must agree on the latent dimension."
            )

        flat = latents.reshape(batch * steps, channels, height, width)
        decoded: torch.Tensor = self.head(self.refine(self.stages(self.project(flat))))
        decoded = self.output_activation(decoded)
        return decoded.reshape(batch, steps, self.out_channels, *decoded.shape[-2:])

    def extra_repr(self) -> str:
        """Return a summary for ``print(model)``."""
        return (
            f"out_channels={self.out_channels}, latent_dim={self.latent_dim}, "
            f"upsample={self.upsample}"
        )
