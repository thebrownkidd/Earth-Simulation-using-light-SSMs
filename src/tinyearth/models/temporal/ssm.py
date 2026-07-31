"""Diagonal state space model (S4D) temporal backbone.

Gu, Gupta, Goel and Ré, *On the Parameterization and Initialization of
Diagonal State Space Models*, NeurIPS 2022.

The backbone this project exists to study.

The model
---------
A continuous linear system, one per channel::

    x'(t) = A x(t) + B u(t)
    y(t)  = C x(t) + D u(t)

discretised with a learned step size ``Δ``. Restricting ``A`` to be **diagonal**
is what makes this cheap: the state update decouples across the ``N`` state
dimensions, so the whole sequence can be computed as a single convolution with
the kernel::

    K_k = C · Ā^k · B̄,    Ā = exp(ΔA),    B̄ = (Ā - 1)/A · B

Following S4D-Lin, ``B`` is fixed to 1 -- it is redundant with ``C`` under a
diagonal ``A``, and fixing it removes parameters without removing capacity.

Why this is the interesting architecture here
---------------------------------------------
**Parameter efficiency**, not asymptotics. The SSM carries roughly ``4HN``
parameters for its temporal mixing (``A`` real and imaginary, complex ``C``),
against ``4H²`` for attention's four projection matrices. At ``H=256, N=64``
that is 65k versus 262k -- a 4x saving in the component under study.

The asymptotic story does **not** apply at this project's sequence lengths. The
history sweep tops out at ``T=8``, where linear-time recurrence versus quadratic
attention is 8 operations against 64. Any measured speed difference at these
lengths comes from constant factors and parameter count, not from complexity
class, and ``docs/models.md`` says so where the numbers are reported.

Forecasting
-----------
``horizon`` learned query embeddings are appended to the history and the SSM is
run over ``T + K`` positions; the last ``K`` outputs are the forecast. This keeps
the model **non-autoregressive**, matching the transformer baseline, so the two
differ only in how they mix over time. The ConvLSTM is the odd one out.
"""

from __future__ import annotations

import math

import torch
from torch import nn

from tinyearth.models.base import TemporalBackbone, check_latent_shape
from tinyearth.models.layers import build_activation
from tinyearth.models.temporal.convlstm import TEMPORAL_BACKBONES

__all__ = ["S4DBackbone", "S4DBlock", "S4DKernel"]

_DT_MIN = 1e-3
_DT_MAX = 1e-1


class S4DKernel(nn.Module):
    """Generates the convolution kernel of a diagonal SSM.

    Parameterisation choices, each load-bearing for stability:

    * ``A``'s real part is stored as ``-exp(log_real)``, so it is **negative by
      construction**. A positive real part makes ``Ā^k`` grow without bound and
      the kernel diverges; constraining it in the parameterisation is safer than
      clamping after the fact.
    * ``Δ`` is stored as ``log_dt``, keeping it positive and letting the model
      span timescales multiplicatively. Initialised log-uniformly over
      ``[1e-3, 1e-1]`` so different channels start at different timescales --
      the mechanism by which an SSM covers short and long dependencies at once.
    * ``A``'s imaginary part is initialised to ``πn`` (S4D-Lin), giving each
      state dimension a distinct oscillation frequency.

    Args:
        channels: Number of independent channels, ``H``.
        state_dim: State size per channel, ``N``. **The knob this project's
            state-dimension sweep varies.**
        dt_min: Lower end of the timescale initialisation range.
        dt_max: Upper end of the timescale initialisation range.

    Raises:
        ValueError: If any size is non-positive or the ``dt`` range is invalid.
    """

    def __init__(
        self,
        channels: int,
        state_dim: int = 64,
        dt_min: float = _DT_MIN,
        dt_max: float = _DT_MAX,
    ) -> None:
        super().__init__()
        if channels < 1:
            raise ValueError(f"channels must be >= 1, got {channels}.")
        if state_dim < 1:
            raise ValueError(f"state_dim must be >= 1, got {state_dim}.")
        if not 0 < dt_min < dt_max:
            raise ValueError(f"Require 0 < dt_min < dt_max, got {dt_min} and {dt_max}.")

        self.channels = channels
        self.state_dim = state_dim

        # Log-uniform over [dt_min, dt_max]: channels start at varied timescales.
        log_dt = torch.rand(channels) * (math.log(dt_max) - math.log(dt_min)) + math.log(dt_min)
        self.log_dt = nn.Parameter(log_dt)

        # A = -exp(log_real) + i * imag, so Re(A) < 0 always.
        self.a_log_real = nn.Parameter(torch.full((channels, state_dim), math.log(0.5)))
        imag = math.pi * torch.arange(state_dim, dtype=torch.float32)
        self.a_imag = nn.Parameter(imag.unsqueeze(0).repeat(channels, 1).clone())

        # C is complex; stored as a trailing real/imaginary axis so that
        # optimisers and checkpoints see plain real tensors.
        self.c = nn.Parameter(torch.randn(channels, state_dim, 2) / math.sqrt(state_dim))

    def forward(self, length: int) -> torch.Tensor:
        """Materialise the kernel for a sequence of ``length`` steps.

        Args:
            length: Sequence length, ``L``.

        Returns:
            Real kernel of shape ``[H, L]``.

        Raises:
            ValueError: If ``length`` is not positive.
        """
        if length < 1:
            raise ValueError(f"length must be >= 1, got {length}.")

        dt = torch.exp(self.log_dt).unsqueeze(-1)  # [H, 1]
        a = -torch.exp(self.a_log_real) + 1j * self.a_imag  # [H, N]
        c = torch.view_as_complex(self.c)  # [H, N]

        dt_a = a * dt  # [H, N]
        # B̄ = (exp(ΔA) - 1)/A with B fixed to 1 (S4D-Lin).
        weight = c * (torch.exp(dt_a) - 1.0) / a  # [H, N]

        steps = torch.arange(length, device=dt_a.device, dtype=torch.float32)
        powers = torch.exp(dt_a.unsqueeze(-1) * steps)  # [H, N, L]

        # The state is a conjugate-symmetric pair, so twice the real part
        # recovers the full response while storing only half the coefficients.
        return 2.0 * torch.einsum("hn,hnl->hl", weight, powers).real

    def extra_repr(self) -> str:
        """Return a summary for ``print(model)``."""
        return f"channels={self.channels}, state_dim={self.state_dim}"


def causal_fft_conv(inputs: torch.Tensor, kernel: torch.Tensor) -> torch.Tensor:
    """Causal depthwise convolution along time, computed by FFT.

    Zero-padding to ``2L`` makes the circular convolution the FFT computes agree
    with the linear one; without it the sequence tail wraps into the head, which
    is a causality violation that produces plausible-looking output.

    Args:
        inputs: ``[B, H, L]``.
        kernel: ``[H, L]``.

    Returns:
        ``[B, H, L]``.
    """
    length = inputs.shape[-1]
    padded = 2 * length

    kernel_f = torch.fft.rfft(kernel.to(torch.float32), n=padded)
    inputs_f = torch.fft.rfft(inputs.to(torch.float32), n=padded)
    convolved: torch.Tensor = torch.fft.irfft(inputs_f * kernel_f.unsqueeze(0), n=padded)
    return convolved[..., :length]


class S4DBlock(nn.Module):
    """One residual SSM block: mix over time, then over channels.

    Structure, pre-norm throughout::

        u -> norm -> SSM(time) -> activation -> channel mix -> + u
          -> norm -> feed-forward                            -> + u

    The two mixing directions are deliberately separate. The SSM mixes **only
    over time** -- it is depthwise, one independent system per channel -- and the
    linear layers mix **only over channels**. Keeping them apart is what makes
    the state-dimension sweep interpretable: ``state_dim`` changes temporal
    capacity alone.

    Args:
        hidden_dim: Channel count, ``H``.
        state_dim: SSM state size, ``N``.
        ffn_multiplier: Feed-forward width as a multiple of ``hidden_dim``.
            ``0`` omits the feed-forward branch entirely.
        dropout: Dropout probability.
        activation: Activation name.
        dt_min: Lower end of the timescale initialisation range.
        dt_max: Upper end of the timescale initialisation range.
    """

    def __init__(
        self,
        hidden_dim: int,
        state_dim: int = 64,
        ffn_multiplier: int = 2,
        dropout: float = 0.0,
        activation: str = "gelu",
        dt_min: float = _DT_MIN,
        dt_max: float = _DT_MAX,
    ) -> None:
        super().__init__()
        if ffn_multiplier < 0:
            raise ValueError(f"ffn_multiplier must be >= 0, got {ffn_multiplier}.")

        self.hidden_dim = hidden_dim
        self.state_dim = state_dim

        self.norm_ssm = nn.LayerNorm(hidden_dim)
        self.kernel = S4DKernel(hidden_dim, state_dim, dt_min=dt_min, dt_max=dt_max)
        # The direct feedthrough D of the state space system: a per-channel skip.
        self.feedthrough = nn.Parameter(torch.ones(hidden_dim))
        self.activation = build_activation(activation)
        self.mix = nn.Linear(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)

        self.ffn: nn.Module | None = None
        if ffn_multiplier > 0:
            inner = hidden_dim * ffn_multiplier
            self.norm_ffn = nn.LayerNorm(hidden_dim)
            self.ffn = nn.Sequential(
                nn.Linear(hidden_dim, inner),
                build_activation(activation),
                nn.Dropout(dropout),
                nn.Linear(inner, hidden_dim),
            )

    def forward(self, sequence: torch.Tensor) -> torch.Tensor:
        """Apply the block.

        Args:
            sequence: ``[B, L, H]``.

        Returns:
            ``[B, L, H]``.
        """
        residual = sequence
        normed = self.norm_ssm(sequence)

        # [B, L, H] -> [B, H, L] for the depthwise time convolution.
        signal = normed.transpose(1, 2)
        kernel = self.kernel(signal.shape[-1])
        mixed = causal_fft_conv(signal, kernel) + signal * self.feedthrough.unsqueeze(-1)
        mixed = mixed.transpose(1, 2)

        mixed = self.dropout(self.mix(self.activation(mixed)))
        sequence = residual + mixed

        if self.ffn is not None:
            sequence = sequence + self.dropout(self.ffn(self.norm_ffn(sequence)))
        return sequence

    def extra_repr(self) -> str:
        """Return a summary for ``print(model)``."""
        return f"hidden_dim={self.hidden_dim}, state_dim={self.state_dim}"


@TEMPORAL_BACKBONES.register("s4d")
class S4DBackbone(TemporalBackbone):
    """Stacked diagonal SSM blocks over the temporal axis.

    Like the transformer baseline, each spatial location is treated as an
    independent sequence: the latent grid folds into the batch dimension, the
    SSM runs over time, and the grid is restored. The encoder and decoder own
    spatial modelling and are held fixed.

    Args:
        latent_dim: Latent channel count, ``D``.
        hidden_dim: Model width, ``H``. **The primary capacity knob.**
        n_layers: Number of stacked blocks.
        state_dim: SSM state size, ``N``.
        ffn_multiplier: Feed-forward width multiple; ``0`` disables it.
        dropout: Dropout probability.
        activation: Activation name.
        dt_min: Lower end of the timescale initialisation range.
        dt_max: Upper end of the timescale initialisation range.

    Raises:
        ValueError: If ``n_layers`` is not positive.
    """

    def __init__(
        self,
        latent_dim: int = 128,
        hidden_dim: int = 128,
        n_layers: int = 4,
        state_dim: int = 64,
        ffn_multiplier: int = 2,
        dropout: float = 0.0,
        activation: str = "gelu",
        dt_min: float = _DT_MIN,
        dt_max: float = _DT_MAX,
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
            S4DBlock(
                hidden_dim=hidden_dim,
                state_dim=state_dim,
                ffn_multiplier=ffn_multiplier,
                dropout=dropout,
                activation=activation,
                dt_min=dt_min,
                dt_max=dt_max,
            )
            for _ in range(n_layers)
        )
        self.norm_out = nn.LayerNorm(hidden_dim)
        self.output_projection = nn.Linear(hidden_dim, latent_dim)

        # Query embeddings for the forecast positions. One per step up to a
        # generous cap, so a model trained at one horizon can be evaluated at
        # another without reshaping parameters.
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
            raise ValueError(
                f"horizon={horizon} exceeds max_horizon={self.max_horizon}. "
                "Raise max_horizon if a longer forecast is genuinely needed."
            )

        batch, steps, channels, height, width = latents.shape

        # [B, T, D, h, w] -> [B*h*w, T, D]: one sequence per spatial location.
        tokens = latents.permute(0, 3, 4, 1, 2).reshape(batch * height * width, steps, channels)
        sequence = self.input_projection(tokens)

        # Append forecast queries; the SSM is causal, so these positions see the
        # history but never each other's futures.
        queries = self.queries[:horizon].unsqueeze(0).expand(sequence.shape[0], horizon, -1)
        sequence = torch.cat([sequence, queries], dim=1)

        for block in self.blocks:
            sequence = block(sequence)

        forecast: torch.Tensor = self.output_projection(self.norm_out(sequence[:, steps:]))

        # [B*h*w, K, D] -> [B, K, D, h, w]
        forecast = forecast.reshape(batch, height, width, horizon, self.latent_dim)
        return forecast.permute(0, 3, 4, 1, 2).contiguous()

    def extra_repr(self) -> str:
        """Return a summary for ``print(model)``."""
        return (
            f"latent_dim={self.latent_dim}, hidden_dim={self.hidden_dim}, "
            f"n_layers={self.n_layers}, state_dim={self.state_dim}"
        )
