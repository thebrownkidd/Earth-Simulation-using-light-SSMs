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
        skip_channels: Channel counts of the paired encoder's skip features,
            finest-to-coarsest -- i.e. exactly
            ``encoder.skip_channels`` when built with matching
            ``skip_connections=True``. ``None`` or empty (the default) builds
            a decoder with no skip pathway, byte-identical to one built
            before skip connections existed. When given, one 1x1 fusion
            convolution is added per resolution level, each concatenating the
            decoder's own features with the matching skip feature and
            projecting back down to the decoder's channel count -- so nothing
            downstream (``refine``, ``head``) needs to change shape.

    Raises:
        ValueError: If ``depth`` is negative, ``output_activation`` is
            unknown, or ``skip_channels`` has the wrong number of entries for
            ``depth``.
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
        skip_channels: list[int] | None = None,
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
        self.skip_channels = list(skip_channels) if skip_channels else []
        if self.skip_channels and len(self.skip_channels) != depth + 1:
            raise ValueError(
                f"skip_channels has {len(self.skip_channels)} entries but depth={depth} "
                f"needs {depth + 1} (one per resolution level, from the latent grid's own "
                "resolution up to full resolution)."
            )

        channels = base_channels * (2**depth)
        self.project = nn.Conv2d(latent_dim, channels, kernel_size=1)

        # One fusion point per resolution level: entry (post-project, the
        # latent grid's own resolution) plus one after each upsample stage.
        # Built in DECODER consumption order -- coarsest first -- which is the
        # reverse of skip_channels' finest-first order, since the decoder
        # starts at the bottleneck and works up to full resolution. Left
        # empty (no parameters added) when there is no skip pathway.
        self.skip_fusion = nn.ModuleList()
        if self.skip_channels:
            level_channels = [channels // (2**stage) for stage in range(depth + 1)]
            self.skip_fusion = nn.ModuleList(
                nn.Conv2d(level + skip, level, kernel_size=1)
                for level, skip in zip(level_channels, reversed(self.skip_channels), strict=True)
            )

        stages: list[nn.Module] = []
        for _ in range(depth):
            stages.append(
                nn.Sequential(
                    nn.Upsample(scale_factor=2, mode="nearest"),
                    conv_block(channels, channels // 2, norm=norm, activation=activation),
                )
            )
            channels //= 2
        # A ModuleList rather than a Sequential so forward can fuse a skip
        # feature between stages; state_dict key paths are identical either
        # way, so this changes nothing when skip_channels is empty.
        self.stages = nn.ModuleList(stages)

        self.refine = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            build_norm(norm, channels),
            build_activation(activation),
        )
        self.head = nn.Conv2d(channels, out_channels, kernel_size=1)
        self.output_activation: nn.Module = (
            nn.Sigmoid() if output_activation == "sigmoid" else nn.Identity()
        )

    def _fuse(
        self, features: torch.Tensor, skip: torch.Tensor, fusion: nn.Module, steps: int
    ) -> torch.Tensor:
        """Broadcast one context-frame skip feature across K forecast steps and fuse.

        ``skip`` is ``[B, C, h, w]`` -- one feature map per batch item, with no
        forecast-step axis, since it comes from a single observed frame. The
        same features are concatenated into every one of the ``K`` steps'
        decoder features: expanded to ``[B, K, C, h, w]`` and flattened to
        ``[B*K, C, h, w]``, matching how ``features`` was itself flattened from
        ``[B, K, D, h, w]``.

        Args:
            features: Decoder features at this resolution, ``[B*K, C, h, w]``.
            skip: The skip feature at this resolution, ``[B, C_i, h, w]``.
            fusion: The 1x1 convolution that projects the concatenation back
                to ``features``'s channel count.
            steps: ``K``, the number of forecast steps.

        Returns:
            Fused features, same shape as ``features``.
        """
        batch = skip.shape[0]
        broadcast = skip.unsqueeze(1).expand(batch, steps, *skip.shape[1:])
        broadcast = broadcast.reshape(batch * steps, *skip.shape[1:])
        fused: torch.Tensor = fusion(torch.cat([features, broadcast], dim=1))
        return fused

    def forward(
        self, latents: torch.Tensor, skips: list[torch.Tensor] | None = None
    ) -> torch.Tensor:
        """Decode a latent sequence to frames.

        Args:
            latents: ``[B, K, D, h, w]``.
            skips: Skip features from ``Encoder.forward_with_skips``, finest-
                to-coarsest, or ``None``/empty when this decoder has no skip
                pathway.

        Returns:
            Frames, ``[B, K, C, H, W]``.

        Raises:
            ValueError: If the input rank or channel count is wrong, or if
                ``skips`` disagrees with how this decoder was built.
        """
        check_latent_shape(latents, "latents")
        batch, steps, channels, height, width = latents.shape
        if channels != self.latent_dim:
            raise ValueError(
                f"Decoder expects latent_dim={self.latent_dim}, got {channels}. "
                "Encoder, backbone and decoder must agree on the latent dimension."
            )

        has_skips = bool(skips)
        if self.skip_channels and not has_skips:
            raise ValueError(
                f"This decoder was built with {len(self.skip_channels)} skip levels but "
                "got none at forward(). Pass the skips returned by "
                "Encoder.forward_with_skips()."
            )
        if not self.skip_channels and has_skips:
            raise ValueError(
                "This decoder has no skip pathway (skip_channels was empty at construction) "
                "but forward() received skips."
            )

        flat = latents.reshape(batch * steps, channels, height, width)
        flat = self.project(flat)

        if has_skips:
            # skips arrives finest-first (the encoder's natural stage order);
            # the decoder starts at the coarsest resolution, so it consumes
            # them in reverse -- see skip_fusion's construction order.
            # `has_skips` (bool(skips)) already rules out None here, but mypy
            # cannot narrow through that, so assert it explicitly.
            assert skips is not None
            ordered = list(reversed(skips))
            flat = self._fuse(flat, ordered[0], self.skip_fusion[0], steps)

        for index, stage in enumerate(self.stages):
            flat = stage(flat)
            if has_skips:
                flat = self._fuse(flat, ordered[index + 1], self.skip_fusion[index + 1], steps)

        decoded: torch.Tensor = self.head(self.refine(flat))
        decoded = self.output_activation(decoded)
        return decoded.reshape(batch, steps, self.out_channels, *decoded.shape[-2:])

    def extra_repr(self) -> str:
        """Return a summary for ``print(model)``."""
        return (
            f"out_channels={self.out_channels}, latent_dim={self.latent_dim}, "
            f"upsample={self.upsample}, skip_channels={self.skip_channels}"
        )
