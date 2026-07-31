"""Shared building blocks.

Small pieces used by more than one component. Keeping them here means the
encoder, decoder and backbones agree on normalisation and activation choices,
so a difference between two experiments cannot come from these.

Normalisation is :class:`~torch.nn.GroupNorm` by default rather than
:class:`~torch.nn.BatchNorm2d`. Batch statistics are unreliable at the small
batch sizes this project uses, and -- more importantly -- BatchNorm couples
samples within a batch, which makes single-sample latency measurement
unrepresentative of training behaviour. Efficiency is a primary result here, so
that coupling is worth avoiding.
"""

from __future__ import annotations

from torch import nn

__all__ = ["ACTIVATIONS", "build_activation", "build_norm", "conv_block"]

ACTIVATIONS: dict[str, type[nn.Module]] = {
    "relu": nn.ReLU,
    "gelu": nn.GELU,
    "silu": nn.SiLU,
    "leaky_relu": nn.LeakyReLU,
}
"""Activation functions selectable from config."""

_MAX_GROUPS = 8


def build_activation(name: str) -> nn.Module:
    """Construct an activation by name.

    Args:
        name: Key from :data:`ACTIVATIONS`.

    Returns:
        A new activation module.

    Raises:
        ValueError: If ``name`` is unknown.
    """
    key = name.strip().lower()
    if key not in ACTIVATIONS:
        available = ", ".join(sorted(ACTIVATIONS))
        raise ValueError(f"Unknown activation {name!r}. Available: {available}.")
    return ACTIVATIONS[key]()


def build_norm(kind: str, channels: int) -> nn.Module:
    """Construct a normalisation layer.

    Args:
        kind: ``"group"``, ``"batch"``, ``"instance"`` or ``"none"``.
        channels: Channel count to normalise.

    Returns:
        The normalisation module; :class:`~torch.nn.Identity` for ``"none"``.

    Raises:
        ValueError: If ``kind`` is unknown.
    """
    key = kind.strip().lower()
    if key == "none":
        return nn.Identity()
    if key == "group":
        # Group count must divide the channel count; fall back toward 1 rather
        # than failing, so a config sweep over latent_dim cannot crash here.
        groups = next((g for g in range(min(_MAX_GROUPS, channels), 0, -1) if channels % g == 0), 1)
        return nn.GroupNorm(groups, channels)
    if key == "batch":
        return nn.BatchNorm2d(channels)
    if key == "instance":
        return nn.InstanceNorm2d(channels, affine=True)
    raise ValueError(f"Unknown norm {kind!r}. Expected 'group', 'batch', 'instance' or 'none'.")


def conv_block(
    in_channels: int,
    out_channels: int,
    *,
    stride: int = 1,
    kernel_size: int = 3,
    norm: str = "group",
    activation: str = "gelu",
) -> nn.Sequential:
    """Build a ``conv -> norm -> activation`` block.

    Args:
        in_channels: Input channels.
        out_channels: Output channels.
        stride: Convolution stride; 2 halves the spatial size.
        kernel_size: Convolution kernel size.
        norm: Normalisation kind, see :func:`build_norm`.
        activation: Activation name, see :func:`build_activation`.

    Returns:
        The assembled block. Convolution bias is omitted when a normalisation
        layer follows, since the norm's shift makes it redundant.
    """
    normed = norm.strip().lower() != "none"
    return nn.Sequential(
        nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=kernel_size // 2,
            bias=not normed,
        ),
        build_norm(norm, out_channels),
        build_activation(activation),
    )
