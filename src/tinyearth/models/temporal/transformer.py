"""Temporal transformer backbone.

Attention over the **time axis only**. Each spatial location in the latent grid
is treated as an independent sequence: the grid is folded into the batch
dimension, attention runs over ``T`` positions, and the grid is restored.

Why not spatiotemporal attention? Two reasons, both about keeping the comparison
honest:

1. It would no longer be a *temporal* backbone. The encoder and decoder own
   spatial modelling and are held fixed; a backbone that also attends spatially
   would add capacity the other backbones do not have, and any difference in
   results would be unattributable.
2. Cost. Full spatiotemporal attention is ``O((T·h·w)^2)``. At ``T=8`` on a
   32x32 grid that is 8192 tokens -- roughly 67M attention entries per head, per
   layer. Temporal-only attention is ``O(T^2)`` per location, which at ``T=8``
   is 64.

The forecast is produced from ``horizon`` learned query embeddings that
cross-attend to the encoded history. This keeps the model **non-autoregressive**:
all ``K`` steps are emitted in one pass. That is a genuine architectural
difference from the ConvLSTM baseline, and a favourable one for latency, so
efficiency comparisons between the two should be read with it in mind rather
than attributed to attention alone.
"""

from __future__ import annotations

import math
from typing import cast

import torch
from torch import nn

from tinyearth.models.base import TemporalBackbone, check_latent_shape
from tinyearth.models.temporal.convlstm import TEMPORAL_BACKBONES

__all__ = ["SinusoidalPositionalEncoding", "TemporalTransformerBackbone"]

_MAX_POSITIONS = 512
"""Ample for this project: history lengths sweep to 8 and horizons to 8."""


class SinusoidalPositionalEncoding(nn.Module):
    """Fixed sinusoidal position encodings.

    Fixed rather than learned so that a model trained at one history length can
    be evaluated at another. The history-length sweep (2, 4, 6, 8) makes that
    directly useful, and learned embeddings would silently cap the usable range.

    Args:
        dim: Embedding dimension; must be even.
        max_positions: Largest sequence length supported.

    Raises:
        ValueError: If ``dim`` is odd.
    """

    def __init__(self, dim: int, max_positions: int = _MAX_POSITIONS) -> None:
        super().__init__()
        if dim % 2 != 0:
            raise ValueError(f"dim must be even for sinusoidal encoding, got {dim}.")

        position = torch.arange(max_positions).unsqueeze(1).float()
        scale = torch.exp(torch.arange(0, dim, 2).float() * (-math.log(10000.0) / dim))
        encoding = torch.zeros(max_positions, dim)
        encoding[:, 0::2] = torch.sin(position * scale)
        encoding[:, 1::2] = torch.cos(position * scale)
        # A buffer, not a parameter: it must move with `.to(device)` and appear
        # in the state_dict, but must never receive gradient.
        self.register_buffer("encoding", encoding, persistent=False)

    def forward(self, length: int, offset: int = 0) -> torch.Tensor:
        """Return encodings for a span of positions.

        Args:
            length: Number of positions.
            offset: Index of the first position. Forecast queries use
                ``offset=T`` so they continue the history's positions rather
                than restarting, which is what tells the model *how far ahead*
                each query is.

        Returns:
            ``[length, dim]``.

        Raises:
            ValueError: If the requested span exceeds ``max_positions``.
        """
        encoding = cast("torch.Tensor", self.encoding)
        if offset + length > encoding.shape[0]:
            raise ValueError(
                f"Requested positions [{offset}, {offset + length}) but only "
                f"{encoding.shape[0]} are available."
            )
        return encoding[offset : offset + length]


@TEMPORAL_BACKBONES.register("transformer")
class TemporalTransformerBackbone(TemporalBackbone):
    """Non-autoregressive transformer over the temporal axis.

    Args:
        latent_dim: Latent channel count, ``D``.
        hidden_dim: Model width. **The main capacity knob.**
        n_layers: Number of encoder and decoder layers (each gets this many).
        n_heads: Attention heads; must divide ``hidden_dim``.
        ffn_multiplier: Feed-forward width as a multiple of ``hidden_dim``.
        dropout: Dropout probability.
        activation: ``"gelu"`` or ``"relu"``.

    Raises:
        ValueError: If ``hidden_dim`` is not divisible by ``n_heads``, or if any
            count is non-positive.
    """

    def __init__(
        self,
        latent_dim: int = 128,
        hidden_dim: int = 128,
        n_layers: int = 2,
        n_heads: int = 4,
        ffn_multiplier: int = 4,
        dropout: float = 0.0,
        activation: str = "gelu",
    ) -> None:
        super().__init__()
        if n_layers < 1:
            raise ValueError(f"n_layers must be >= 1, got {n_layers}.")
        if n_heads < 1:
            raise ValueError(f"n_heads must be >= 1, got {n_heads}.")
        if hidden_dim % n_heads != 0:
            raise ValueError(f"hidden_dim={hidden_dim} must be divisible by n_heads={n_heads}.")

        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.n_heads = n_heads

        self.input_projection = nn.Linear(latent_dim, hidden_dim)
        self.positions = SinusoidalPositionalEncoding(hidden_dim)

        layer_kwargs = {
            "d_model": hidden_dim,
            "nhead": n_heads,
            "dim_feedforward": hidden_dim * ffn_multiplier,
            "dropout": dropout,
            "activation": activation,
            "batch_first": True,
            "norm_first": True,  # pre-norm: markedly more stable without warmup
        }
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(**layer_kwargs),  # type: ignore[arg-type]
            num_layers=n_layers,
            # The nested-tensor fast path does not apply with norm_first=True and
            # would otherwise warn on every construction. Sequences here are
            # short and unpadded, so it would buy nothing anyway.
            enable_nested_tensor=False,
        )
        self.decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(**layer_kwargs),  # type: ignore[arg-type]
            num_layers=n_layers,
        )
        self.output_projection = nn.Linear(hidden_dim, latent_dim)

    def forward(self, latents: torch.Tensor, horizon: int) -> torch.Tensor:
        """Forecast ``horizon`` latent frames.

        Args:
            latents: Latent history, ``[B, T, D, h, w]``.
            horizon: Number of frames to forecast.

        Returns:
            Latent forecast, ``[B, K, D, h, w]``.

        Raises:
            ValueError: If the input rank is wrong or ``horizon`` is not positive.
        """
        check_latent_shape(latents, "latents")
        if horizon < 1:
            raise ValueError(f"horizon must be >= 1, got {horizon}.")

        batch, steps, channels, height, width = latents.shape

        # [B, T, D, h, w] -> [B*h*w, T, D]: every spatial location becomes an
        # independent sequence in the batch.
        tokens = latents.permute(0, 3, 4, 1, 2).reshape(batch * height * width, steps, channels)

        memory = self.input_projection(tokens)
        memory = memory + self.positions(steps).unsqueeze(0)
        memory = self.encoder(memory)

        # Learned-free queries: position encodings alone identify each forecast
        # step, continuing the history's positions.
        queries = self.positions(horizon, offset=steps)
        queries = queries.unsqueeze(0).expand(memory.shape[0], horizon, self.hidden_dim)

        decoded = self.decoder(queries, memory)
        outputs: torch.Tensor = self.output_projection(decoded)

        # [B*h*w, K, D] -> [B, K, D, h, w]
        outputs = outputs.reshape(batch, height, width, horizon, self.latent_dim)
        return outputs.permute(0, 3, 4, 1, 2).contiguous()

    def extra_repr(self) -> str:
        """Return a summary for ``print(model)``."""
        return (
            f"latent_dim={self.latent_dim}, hidden_dim={self.hidden_dim}, "
            f"n_layers={self.n_layers}, n_heads={self.n_heads}"
        )
