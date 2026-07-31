"""Selective state space model (Mamba-style) temporal backbone.

Gu and Dao, *Mamba: Linear-Time Sequence Modeling with Selective State
Spaces*, 2023.

The second SSM in the study. Where :class:`~tinyearth.models.temporal.ssm.S4DBackbone`
uses a **fixed** state space system -- the same ``Δ``, ``B`` and ``C`` at every
timestep -- this one makes them **functions of the input**. The model can then
decide, per step, how much of the incoming observation to write into state and
how much history to retain.

For Earth observation that is a substantive difference rather than a decorative
one: a cloudy frame carries almost no surface information, and a selective model
can learn to hold its state through one rather than overwriting it. A fixed SSM
integrates every frame with the same weight regardless of content.

The cost of selectivity
-----------------------
Input-dependent parameters make the system **time-varying**, so it can no longer
be written as a single convolution. The state must be advanced step by step.
This module uses a plain sequential loop, which is the honest choice at this
project's sequence lengths: the history sweep tops out at ``T=8``, so the loop
runs at most 16 iterations including forecast queries, and the parallel
associative scan that Mamba needs at ``T=4096`` would be machinery with nothing
to do here.

That trade-off is real and worth stating when reporting results: **the selective
model gives up the parallel-in-time property that the S4D kernel has.** At
``T≤8`` this costs little; at long sequence lengths it would need the scan.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn

from tinyearth.models.base import TemporalBackbone, check_latent_shape
from tinyearth.models.temporal.convlstm import TEMPORAL_BACKBONES

__all__ = ["MambaBackbone", "SelectiveSSMBlock", "selective_scan"]

_MIN_DT = 1e-4


def selective_scan(
    signal: torch.Tensor,
    dt: torch.Tensor,
    a_matrix: torch.Tensor,
    b_matrix: torch.Tensor,
    c_matrix: torch.Tensor,
) -> torch.Tensor:
    """Advance a time-varying diagonal state space system.

    Implements, per step::

        h_t = exp(Δ_t A) · h_{t-1} + Δ_t B_t x_t
        y_t = (C_t · h_t).sum(state)

    Sequential by necessity: with input-dependent coefficients the recurrence is
    time-varying, so no single convolution kernel exists. See the module
    docstring on why a sequential loop is the right choice at these lengths.

    **The discretised coefficients are built inside the loop, one step at a
    time.** Precomputing them for all steps means materialising three
    ``[B, L, C, N]`` tensors -- 346M elements each at the ``large`` tier -- which
    makes the block memory-bound. Per-step construction holds only ``[B, C, N]``,
    an exact ``L``-fold reduction in peak activation memory, and measured about
    1.4x faster end to end (13.8s to 9.8s for a large-tier forward pass on CPU).

    It is still slow. The remaining cost is the Python-level loop dispatching
    many small kernels, which is precisely what Mamba's fused CUDA kernel exists
    to avoid; see ``docs/phase-4.md`` for the honest accounting.

    Args:
        signal: Input ``x``, ``[B, L, C]``.
        dt: Per-step step size ``Δ``, ``[B, L, C]``.
        a_matrix: State matrix ``A``, ``[C, N]``. Must be negative for stability.
        b_matrix: Per-step input matrix ``B``, ``[B, L, N]``.
        c_matrix: Per-step readout matrix ``C``, ``[B, L, N]``.

    Returns:
        ``[B, L, C]``.
    """
    batch, length, channels = signal.shape
    state_dim = a_matrix.shape[-1]
    state = torch.zeros(batch, channels, state_dim, device=signal.device, dtype=signal.dtype)

    outputs = []
    for step in range(length):
        step_dt = dt[:, step].unsqueeze(-1)  # [B, C, 1]
        decay = torch.exp(step_dt * a_matrix)  # [B, C, N]
        drive = step_dt * b_matrix[:, step].unsqueeze(1) * signal[:, step].unsqueeze(-1)
        state = decay * state + drive
        outputs.append((c_matrix[:, step].unsqueeze(1) * state).sum(dim=-1))
    return torch.stack(outputs, dim=1)


class SelectiveSSMBlock(nn.Module):
    """A Mamba block: gated selective SSM with a short causal convolution.

    Structure::

        x -> norm -> in_proj -> (signal, gate)
                     signal -> depthwise causal conv -> SiLU
                            -> selective SSM
                            -> * SiLU(gate)
                            -> out_proj -> + x

    The short depthwise convolution supplies local context before the SSM, which
    is what lets the selection projections condition on a small neighbourhood
    rather than a single timestep.

    Args:
        hidden_dim: Channel count, ``H``.
        state_dim: State size per channel, ``N``.
        expand: Inner width multiplier; inner width is ``expand * hidden_dim``.
        conv_kernel: Width of the causal depthwise convolution.
        dt_rank: Rank of the low-rank ``Δ`` projection. ``None`` uses
            ``ceil(hidden_dim / 16)``, following the reference implementation.
        dropout: Dropout probability.

    Raises:
        ValueError: If any size is non-positive.
    """

    def __init__(
        self,
        hidden_dim: int,
        state_dim: int = 16,
        expand: int = 2,
        conv_kernel: int = 4,
        dt_rank: int | None = None,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if min(hidden_dim, state_dim, expand, conv_kernel) < 1:
            raise ValueError("hidden_dim, state_dim, expand and conv_kernel must all be >= 1.")

        self.hidden_dim = hidden_dim
        self.state_dim = state_dim
        self.inner_dim = expand * hidden_dim
        self.dt_rank = dt_rank if dt_rank is not None else math.ceil(hidden_dim / 16)
        self.conv_kernel = conv_kernel

        self.norm = nn.LayerNorm(hidden_dim)
        self.in_projection = nn.Linear(hidden_dim, 2 * self.inner_dim, bias=False)

        self.conv = nn.Conv1d(
            self.inner_dim,
            self.inner_dim,
            kernel_size=conv_kernel,
            groups=self.inner_dim,  # depthwise: no channel mixing here
            padding=0,  # padded manually on the left, to stay causal
        )

        # Produces the selective (input-dependent) parameters.
        self.x_projection = nn.Linear(self.inner_dim, self.dt_rank + 2 * state_dim, bias=False)
        self.dt_projection = nn.Linear(self.dt_rank, self.inner_dim, bias=True)
        self._init_dt_bias()

        # A is negative by construction, initialised so state dimension n decays
        # at rate n+1: a spread of timescales from the start.
        a_init = torch.arange(1, state_dim + 1, dtype=torch.float32)
        self.a_log = nn.Parameter(torch.log(a_init).unsqueeze(0).repeat(self.inner_dim, 1))
        self.feedthrough = nn.Parameter(torch.ones(self.inner_dim))

        self.out_projection = nn.Linear(self.inner_dim, hidden_dim, bias=False)
        self.dropout = nn.Dropout(dropout)

    def _init_dt_bias(self) -> None:
        """Bias ``Δ`` so ``softplus`` starts it in a useful range.

        Without this the initial ``Δ`` sits wherever the linear layer's default
        bias puts it, which is typically far too small -- the state barely
        updates and the block looks broken for the first few thousand steps.
        """
        dt = torch.exp(
            torch.rand(self.inner_dim) * (math.log(0.1) - math.log(1e-3)) + math.log(1e-3)
        )
        dt = dt.clamp_min(_MIN_DT)
        # Invert softplus so that softplus(bias) == dt.
        inverse = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            self.dt_projection.bias.copy_(inverse)

    def forward(self, sequence: torch.Tensor) -> torch.Tensor:
        """Apply the block.

        Args:
            sequence: ``[B, L, H]``.

        Returns:
            ``[B, L, H]``.
        """
        residual = sequence

        projected = self.in_projection(self.norm(sequence))
        signal, gate = projected.chunk(2, dim=-1)

        # Causal depthwise convolution: pad on the left only, so position t
        # never sees t+1.
        conv_in = signal.transpose(1, 2)
        conv_in = F.pad(conv_in, (self.conv_kernel - 1, 0))
        signal = F.silu(self.conv(conv_in).transpose(1, 2))

        # Selective parameters, all functions of the input.
        parameters = self.x_projection(signal)
        dt, b_matrix, c_matrix = torch.split(
            parameters, [self.dt_rank, self.state_dim, self.state_dim], dim=-1
        )
        dt = F.softplus(self.dt_projection(dt))  # [B, L, C]

        a_matrix = -torch.exp(self.a_log)  # [C, N]
        scanned = selective_scan(signal, dt, a_matrix, b_matrix, c_matrix)
        scanned = scanned + signal * self.feedthrough

        gated = scanned * F.silu(gate)
        projected_out: torch.Tensor = self.out_projection(gated)
        dropped: torch.Tensor = self.dropout(projected_out)
        return residual + dropped

    def extra_repr(self) -> str:
        """Return a summary for ``print(model)``."""
        return (
            f"hidden_dim={self.hidden_dim}, state_dim={self.state_dim}, "
            f"inner_dim={self.inner_dim}"
        )


@TEMPORAL_BACKBONES.register("mamba")
class MambaBackbone(TemporalBackbone):
    """Stacked selective SSM blocks over the temporal axis.

    Non-autoregressive, matching :class:`~tinyearth.models.temporal.ssm.S4DBackbone`
    and the transformer baseline: ``horizon`` learned query positions are
    appended and the last ``K`` outputs taken.

    Args:
        latent_dim: Latent channel count, ``D``.
        hidden_dim: Model width, ``H``. **The primary capacity knob.**
        n_layers: Number of stacked blocks.
        state_dim: State size per channel, ``N``. Mamba uses a small ``N`` (16)
            because selectivity, not state size, carries the capacity.
        expand: Inner width multiplier.
        conv_kernel: Width of the causal depthwise convolution.
        dt_rank: Rank of the ``Δ`` projection; ``None`` derives it.
        dropout: Dropout probability.

    Raises:
        ValueError: If ``n_layers`` is not positive.
    """

    def __init__(
        self,
        latent_dim: int = 128,
        hidden_dim: int = 128,
        n_layers: int = 4,
        state_dim: int = 16,
        expand: int = 2,
        conv_kernel: int = 4,
        dt_rank: int | None = None,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if n_layers < 1:
            raise ValueError(f"n_layers must be >= 1, got {n_layers}.")

        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.state_dim = state_dim

        self.input_projection = nn.Linear(latent_dim, hidden_dim)
        self.blocks: nn.ModuleList = nn.ModuleList(
            SelectiveSSMBlock(
                hidden_dim=hidden_dim,
                state_dim=state_dim,
                expand=expand,
                conv_kernel=conv_kernel,
                dt_rank=dt_rank,
                dropout=dropout,
            )
            for _ in range(n_layers)
        )
        self.norm_out = nn.LayerNorm(hidden_dim)
        self.output_projection = nn.Linear(hidden_dim, latent_dim)

        self.max_horizon = 32
        self.queries = nn.Parameter(torch.randn(self.max_horizon, hidden_dim) * 0.02)

    def forward(self, latents: torch.Tensor, horizon: int) -> torch.Tensor:
        """Forecast ``horizon`` latent frames.

        Args:
            latents: Latent history, ``[B, T, D, h, w]``.
            horizon: Number of frames to forecast, ``K``.

        Returns:
            Latent forecast, ``[B, K, D, h, w]``.

        Raises:
            ValueError: If the input rank is wrong, ``horizon`` is not positive,
                or ``horizon`` exceeds the query budget.
        """
        check_latent_shape(latents, "latents")
        if horizon < 1:
            raise ValueError(f"horizon must be >= 1, got {horizon}.")
        if horizon > self.max_horizon:
            raise ValueError(f"horizon={horizon} exceeds max_horizon={self.max_horizon}.")

        batch, steps, channels, height, width = latents.shape

        tokens = latents.permute(0, 3, 4, 1, 2).reshape(batch * height * width, steps, channels)
        sequence = self.input_projection(tokens)

        queries = self.queries[:horizon].unsqueeze(0).expand(sequence.shape[0], horizon, -1)
        sequence = torch.cat([sequence, queries], dim=1)

        for block in self.blocks:
            sequence = block(sequence)

        forecast: torch.Tensor = self.output_projection(self.norm_out(sequence[:, steps:]))
        forecast = forecast.reshape(batch, height, width, horizon, self.latent_dim)
        return forecast.permute(0, 3, 4, 1, 2).contiguous()

    def extra_repr(self) -> str:
        """Return a summary for ``print(model)``."""
        return (
            f"latent_dim={self.latent_dim}, hidden_dim={self.hidden_dim}, "
            f"n_layers={self.n_layers}, state_dim={self.state_dim}"
        )
