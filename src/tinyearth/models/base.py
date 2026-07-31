"""Component interfaces for the forecasting architecture.

The architecture is fixed::

    encoder  ->  temporal backbone  ->  decoder  ->  forecast

and only the temporal backbone varies across experiments. That is the controlled
comparison the whole project rests on, so the interfaces are defined here, once,
and every component implements them.

Shapes
------

===================  =========================  =========================
Component            Input                      Output
===================  =========================  =========================
:class:`Encoder`     ``[B, T, C, H, W]``        ``[B, T, D, h, w]``
:class:`TemporalBackbone`  ``[B, T, D, h, w]``  ``[B, K, D, h, w]``
:class:`Decoder`     ``[B, K, D, h, w]``        ``[B, K, C, H, W]``
===================  =========================  =========================

``T`` is the history length, ``K`` the forecast horizon, ``D`` the latent
channel count, and ``h, w`` the downsampled spatial size.

Why the backbone emits ``K`` steps rather than a single summary state: it is the
component under study, so the sequence-to-sequence mapping should live entirely
inside it. Anything the decoder did to expand one state into ``K`` frames would
be shared machinery that quietly differs in capacity between architectures, and
would contaminate the comparison.

Backbones are :class:`~torch.nn.Module` subclasses rather than
:class:`~typing.Protocol` implementations because they hold parameters and must
participate in ``state_dict``, ``.to(device)`` and optimiser construction.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import torch
from torch import nn

__all__ = ["Decoder", "Encoder", "TemporalBackbone", "check_latent_shape"]


def check_latent_shape(tensor: torch.Tensor, name: str, expected_rank: int = 5) -> None:
    """Validate that a tensor has the expected rank.

    Shape errors deep inside a backbone produce unreadable messages, so the
    boundaries are checked explicitly.

    Args:
        tensor: Tensor to check.
        name: Name used in the error message.
        expected_rank: Required number of dimensions.

    Raises:
        ValueError: If the rank does not match.
    """
    if tensor.ndim != expected_rank:
        raise ValueError(
            f"{name} must have {expected_rank} dimensions "
            f"[B, T, C, H, W], got shape {tuple(tensor.shape)}."
        )


class Encoder(nn.Module, ABC):
    """Maps a frame sequence to a latent grid.

    Held fixed across temporal-backbone experiments.

    Attributes:
        in_channels: Number of input imagery channels.
        latent_dim: Channel count of the latent grid, ``D``.
        downsample: Spatial reduction factor, so ``h = H // downsample``.
    """

    in_channels: int
    latent_dim: int
    downsample: int

    @abstractmethod
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """Encode a sequence of frames.

        Args:
            images: ``[B, T, C, H, W]``.

        Returns:
            Latent grid, ``[B, T, D, h, w]``.
        """
        raise NotImplementedError


class TemporalBackbone(nn.Module, ABC):
    """Maps a latent history to a latent forecast.

    **The only component that varies across experiments.** Everything else in
    the architecture is held fixed, so a difference in results is attributable
    to this module.

    Implementations register themselves in
    :data:`tinyearth.models.temporal.TEMPORAL_BACKBONES` so configs select them
    by name.

    Attributes:
        latent_dim: Channel count of the latent grid it consumes and produces.
    """

    latent_dim: int

    @abstractmethod
    def forward(self, latents: torch.Tensor, horizon: int) -> torch.Tensor:
        """Forecast ``horizon`` latent frames from a latent history.

        Args:
            latents: Latent history, ``[B, T, D, h, w]``.
            horizon: Number of latent frames to produce, ``K``.

        Returns:
            Latent forecast, ``[B, K, D, h, w]``.
        """
        raise NotImplementedError


class Decoder(nn.Module, ABC):
    """Maps a latent grid back to pixel space.

    Held fixed across temporal-backbone experiments.

    Attributes:
        out_channels: Number of output imagery channels.
        latent_dim: Channel count of the latent grid it consumes.
        upsample: Spatial expansion factor; must match the encoder's downsample.
    """

    out_channels: int
    latent_dim: int
    upsample: int

    @abstractmethod
    def forward(self, latents: torch.Tensor) -> torch.Tensor:
        """Decode a latent sequence to frames.

        Args:
            latents: ``[B, K, D, h, w]``.

        Returns:
            Frames, ``[B, K, C, H, W]``.
        """
        raise NotImplementedError
