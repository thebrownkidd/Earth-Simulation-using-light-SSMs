"""ConvLSTM temporal backbone.

Shi et al., *Convolutional LSTM Network: A Machine Learning Approach for
Precipitation Nowcasting*, NeurIPS 2015.

The reference point for spatiotemporal forecasting. It replaces the LSTM's
matrix multiplications with convolutions, so the hidden state keeps its spatial
layout and locality is preserved.

Structure: an **encoder-decoder** (seq2seq) arrangement. The cells consume the
history to build up state, then run ``horizon`` further steps to emit the
forecast. During the forecast steps the input is the previous latent output,
which is the standard formulation and keeps the model autoregressive.

Cost characteristics -- the reason it is the baseline to beat:

* Time is **strictly sequential**. Step ``t`` needs step ``t-1``, so no
  parallelism is available over the temporal axis, and latency grows linearly
  in ``T + K``.
* Each step performs a convolution over the full spatial grid, so compute is
  linear in ``h * w`` with a large constant.

Both are precisely what a state space model is expected to improve on, so the
efficiency metrics in :mod:`tinyearth.evaluation` are what this baseline exists
to be measured against.
"""

from __future__ import annotations

from typing import cast

import torch
from torch import nn

from tinyearth.models.base import TemporalBackbone, check_latent_shape
from tinyearth.utils.registry import Registry

__all__ = ["TEMPORAL_BACKBONES", "ConvLSTMBackbone", "ConvLSTMCell"]

TEMPORAL_BACKBONES: Registry[TemporalBackbone] = Registry("temporal_backbone")
"""Temporal backbones selectable by name from a config.

The registry that makes "only the temporal backbone changes" a configuration
edit rather than a code edit. Phase 4 adds State Space Model entries here.
"""

_N_GATES = 4
"""input, forget, cell and output gates, computed in one convolution."""


class ConvLSTMCell(nn.Module):
    """A single ConvLSTM cell.

    All four gates are produced by one convolution over the concatenated input
    and hidden state, which is both faster and closer to the reference
    implementation than four separate convolutions.

    Args:
        input_dim: Channels of the input at each step.
        hidden_dim: Channels of the hidden and cell states.
        kernel_size: Convolution kernel size; must be odd so padding preserves
            the spatial size.
        bias: Include a convolution bias.

    Raises:
        ValueError: If ``kernel_size`` is even.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        kernel_size: int = 3,
        bias: bool = True,
    ) -> None:
        super().__init__()
        if kernel_size % 2 == 0:
            raise ValueError(f"kernel_size must be odd, got {kernel_size}.")

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.conv = nn.Conv2d(
            input_dim + hidden_dim,
            _N_GATES * hidden_dim,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            bias=bias,
        )
        self._init_forget_gate_bias()

    def _init_forget_gate_bias(self) -> None:
        """Initialise the forget-gate bias to 1.

        A standard LSTM trick: it keeps the cell state from decaying before the
        model has learned anything, which matters here because the sequences are
        short and there is little opportunity to recover from early forgetting.
        """
        if self.conv.bias is None:
            return
        with torch.no_grad():
            # Gate order below is [input, forget, cell, output].
            self.conv.bias[self.hidden_dim : 2 * self.hidden_dim].fill_(1.0)

    def forward(
        self,
        inputs: torch.Tensor,
        state: tuple[torch.Tensor, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Advance the cell by one step.

        Args:
            inputs: ``[B, input_dim, h, w]``.
            state: ``(hidden, cell)``, each ``[B, hidden_dim, h, w]``.

        Returns:
            The updated ``(hidden, cell)``.
        """
        hidden, cell = state
        gates = self.conv(torch.cat([inputs, hidden], dim=1))
        input_gate, forget_gate, candidate, output_gate = gates.chunk(_N_GATES, dim=1)

        input_gate = torch.sigmoid(input_gate)
        forget_gate = torch.sigmoid(forget_gate)
        output_gate = torch.sigmoid(output_gate)
        candidate = torch.tanh(candidate)

        next_cell = forget_gate * cell + input_gate * candidate
        next_hidden = output_gate * torch.tanh(next_cell)
        return next_hidden, next_cell

    def initial_state(
        self, batch: int, spatial: tuple[int, int], device: torch.device, dtype: torch.dtype
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Create a zero-initialised state.

        Args:
            batch: Batch size.
            spatial: ``(h, w)``.
            device: Device to allocate on.
            dtype: Tensor dtype.

        Returns:
            ``(hidden, cell)``, both zeros.
        """
        shape = (batch, self.hidden_dim, *spatial)
        zeros = torch.zeros(shape, device=device, dtype=dtype)
        return zeros, zeros.clone()


@TEMPORAL_BACKBONES.register("convlstm")
class ConvLSTMBackbone(TemporalBackbone):
    """Stacked ConvLSTM in an encoder-decoder arrangement.

    Args:
        latent_dim: Latent channel count, ``D``. Also the output channel count.
        hidden_dim: Hidden channels per cell. **The main capacity knob** -- cost
            grows roughly quadratically in it, since the gate convolution reads
            ``input_dim + hidden_dim`` channels and writes ``4 * hidden_dim``.
        n_layers: Number of stacked cells.
        kernel_size: Convolution kernel size within each cell.
        bias: Include convolution biases.

    Raises:
        ValueError: If ``n_layers`` or ``hidden_dim`` is not positive.
    """

    def __init__(
        self,
        latent_dim: int = 128,
        hidden_dim: int = 128,
        n_layers: int = 2,
        kernel_size: int = 3,
        bias: bool = True,
    ) -> None:
        super().__init__()
        if n_layers < 1:
            raise ValueError(f"n_layers must be >= 1, got {n_layers}.")
        if hidden_dim < 1:
            raise ValueError(f"hidden_dim must be >= 1, got {hidden_dim}.")

        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers

        self.cells: nn.ModuleList = nn.ModuleList(
            ConvLSTMCell(
                input_dim=latent_dim if layer == 0 else hidden_dim,
                hidden_dim=hidden_dim,
                kernel_size=kernel_size,
                bias=bias,
            )
            for layer in range(n_layers)
        )
        # Projects the top hidden state back to the latent dimension, so the
        # forecast steps can be fed their own output.
        self.output: nn.Conv2d = nn.Conv2d(hidden_dim, latent_dim, kernel_size=1)

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

        batch, steps, _, height, width = latents.shape
        spatial = (height, width)
        cells = cast("list[ConvLSTMCell]", list(self.cells))
        states = [
            cell.initial_state(batch, spatial, latents.device, latents.dtype) for cell in cells
        ]

        # Encode the history.
        for step in range(steps):
            states = self._step(latents[:, step], states)

        # Decode `horizon` steps, feeding each output back in.
        outputs: list[torch.Tensor] = []
        current: torch.Tensor = self.output(states[-1][0])
        for _ in range(horizon):
            outputs.append(current)
            states = self._step(current, states)
            current = self.output(states[-1][0])

        return torch.stack(outputs, dim=1)

    def _step(
        self,
        inputs: torch.Tensor,
        states: list[tuple[torch.Tensor, torch.Tensor]],
    ) -> list[tuple[torch.Tensor, torch.Tensor]]:
        """Advance every layer by one step.

        Args:
            inputs: ``[B, D, h, w]`` for the first layer.
            states: Per-layer ``(hidden, cell)``.

        Returns:
            The updated per-layer states.
        """
        updated: list[tuple[torch.Tensor, torch.Tensor]] = []
        signal = inputs
        cells = cast("list[ConvLSTMCell]", list(self.cells))
        for cell, state in zip(cells, states, strict=True):
            hidden, memory = cell(signal, state)
            updated.append((hidden, memory))
            signal = hidden
        return updated

    def extra_repr(self) -> str:
        """Return a summary for ``print(model)``."""
        return (
            f"latent_dim={self.latent_dim}, hidden_dim={self.hidden_dim}, "
            f"n_layers={self.n_layers}"
        )
