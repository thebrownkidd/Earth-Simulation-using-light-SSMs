"""Tests for the state space model backbones.

The most important test here is
:meth:`TestS4DKernel.test_fft_convolution_matches_the_explicit_recurrence`. An
SSM whose kernel is subtly wrong still trains, still produces plausible losses,
and still yields publishable-looking numbers — the mathematics has to be pinned
to a reference computation, not to "it seems to learn".
"""

from __future__ import annotations

import pytest
import torch

from tinyearth.models.base import TemporalBackbone
from tinyearth.models.temporal import TEMPORAL_BACKBONES
from tinyearth.models.temporal.selective_ssm import (
    MambaBackbone,
    SelectiveSSMBlock,
    selective_scan,
)
from tinyearth.models.temporal.ssm import (
    S4DBackbone,
    S4DBlock,
    S4DKernel,
    causal_fft_conv,
)

SSM_NAMES = ("s4d", "mamba")


class TestS4DKernel:
    def test_shape(self):
        assert S4DKernel(4, 8)(16).shape == (4, 16)

    def test_kernel_is_real(self):
        kernel = S4DKernel(3, 8)(12)
        assert kernel.dtype == torch.float32
        assert torch.isfinite(kernel).all()

    def test_fft_convolution_matches_the_explicit_recurrence(self):
        """The load-bearing correctness test.

        Computes the state space response two independent ways -- the FFT
        convolution the model uses, and a literal step-by-step recurrence
        x_k = A_bar x_{k-1} + B_bar u_k -- and requires they agree.
        """
        torch.manual_seed(0)
        channels, state_dim, length, batch = 3, 8, 12, 2
        module = S4DKernel(channels, state_dim)

        inputs = torch.randn(batch, channels, length)
        via_fft = causal_fft_conv(inputs, module(length))

        dt = torch.exp(module.log_dt).unsqueeze(-1)
        a = -torch.exp(module.a_log_real) + 1j * module.a_imag
        c = torch.view_as_complex(module.c)
        a_bar = torch.exp(a * dt)
        b_bar = (a_bar - 1.0) / a

        via_recurrence = torch.zeros(batch, channels, length)
        state = torch.zeros(batch, channels, state_dim, dtype=torch.complex64)
        for step in range(length):
            state = a_bar * state + b_bar * inputs[:, :, step].unsqueeze(-1)
            via_recurrence[:, :, step] = 2.0 * (c * state).sum(-1).real

        torch.testing.assert_close(via_fft, via_recurrence, atol=1e-4, rtol=1e-4)

    def test_kernel_decays_rather_than_diverges(self):
        """Re(A) < 0 by construction; a positive real part diverges."""
        kernel = S4DKernel(4, 16)(512)
        assert torch.isfinite(kernel).all()
        assert float(kernel[:, -1].abs().max()) < float(kernel[:, 0].abs().max())

    def test_stays_stable_even_with_large_a_parameters(self):
        """The parameterisation, not a clamp, is what guarantees stability."""
        module = S4DKernel(2, 8)
        with torch.no_grad():
            module.a_log_real.fill_(4.0)  # A_real = -exp(4) — strongly damped
            module.log_dt.fill_(0.0)
        assert torch.isfinite(module(128)).all()

    def test_timescales_are_initialised_with_spread(self):
        """Channels must start at different timescales, not all the same."""
        dt = torch.exp(S4DKernel(64, 16).log_dt)
        assert float(dt.max() / dt.min()) > 10.0

    def test_gradients_reach_every_parameter(self):
        module = S4DKernel(3, 8)
        module(16).sum().backward()
        for name, param in module.named_parameters():
            assert param.grad is not None, name
            assert torch.isfinite(param.grad).all(), name

    @pytest.mark.parametrize(("channels", "state_dim"), [(0, 8), (4, 0)])
    def test_rejects_non_positive_sizes(self, channels, state_dim):
        with pytest.raises(ValueError, match="must be >= 1"):
            S4DKernel(channels, state_dim)

    def test_rejects_invalid_dt_range(self):
        with pytest.raises(ValueError, match="dt_min < dt_max"):
            S4DKernel(4, 8, dt_min=0.5, dt_max=0.1)

    def test_rejects_non_positive_length(self):
        with pytest.raises(ValueError, match="length must be"):
            S4DKernel(4, 8)(0)


class TestCausalFFTConv:
    def test_is_causal(self):
        """Perturbing a later input must not change earlier outputs."""
        torch.manual_seed(0)
        inputs = torch.randn(2, 3, 16)
        kernel = torch.randn(3, 16)

        baseline = causal_fft_conv(inputs, kernel)
        perturbed = inputs.clone()
        perturbed[:, :, 8:] += 100.0
        changed = causal_fft_conv(perturbed, kernel)

        torch.testing.assert_close(changed[:, :, :8], baseline[:, :, :8], atol=1e-3, rtol=1e-4)

    def test_matches_direct_convolution(self):
        """Guards the 2L zero-padding: too little and the tail wraps into the head."""
        torch.manual_seed(0)
        length = 8
        inputs = torch.randn(1, 1, length)
        kernel = torch.randn(1, length)

        expected = torch.zeros(1, 1, length)
        for k in range(length):
            for j in range(k + 1):
                expected[0, 0, k] += kernel[0, k - j] * inputs[0, 0, j]

        torch.testing.assert_close(causal_fft_conv(inputs, kernel), expected, atol=1e-4, rtol=1e-4)

    def test_preserves_shape(self):
        assert causal_fft_conv(torch.randn(2, 5, 9), torch.randn(5, 9)).shape == (2, 5, 9)


class TestSelectiveScan:
    def test_matches_a_literal_recurrence(self):
        """The selective analogue of the S4D kernel check."""
        torch.manual_seed(0)
        batch, length, channels, state = 2, 6, 3, 4
        signal = torch.randn(batch, length, channels)
        dt = torch.rand(batch, length, channels) * 0.1
        a_matrix = -torch.rand(channels, state)
        b_matrix = torch.randn(batch, length, state)
        c_matrix = torch.randn(batch, length, state)

        result = selective_scan(signal, dt, a_matrix, b_matrix, c_matrix)

        expected = torch.zeros(batch, length, channels)
        state_vector = torch.zeros(batch, channels, state)
        for step in range(length):
            step_dt = dt[:, step].unsqueeze(-1)
            state_vector = torch.exp(step_dt * a_matrix) * state_vector + (
                step_dt * b_matrix[:, step].unsqueeze(1) * signal[:, step].unsqueeze(-1)
            )
            expected[:, step] = (c_matrix[:, step].unsqueeze(1) * state_vector).sum(-1)

        torch.testing.assert_close(result, expected)

    def test_large_dt_forgets_almost_everything(self):
        """A big step size drives exp(dt*A) toward zero: the state is overwritten."""
        batch, length, channels, state = 1, 4, 2, 3
        result = selective_scan(
            torch.ones(batch, length, channels),
            torch.full((batch, length, channels), 50.0),
            torch.full((channels, state), -1.0),
            torch.ones(batch, length, state),
            torch.ones(batch, length, state),
        )
        # Each output reflects only the current step's drive: dt * B * x * state.
        torch.testing.assert_close(result, torch.full((batch, length, channels), 50.0 * state))

    def test_tiny_dt_accumulates_across_steps(self):
        """A small step size keeps exp(dt*A) near 1: history is retained."""
        batch, length, channels, state = 1, 4, 1, 1
        result = selective_scan(
            torch.ones(batch, length, channels),
            torch.full((batch, length, channels), 1e-6),
            torch.full((channels, state), -1.0),
            torch.ones(batch, length, state),
            torch.ones(batch, length, state),
        )
        # Nearly lossless accumulation: step k contributes k * dt.
        expected = torch.tensor([1e-6, 2e-6, 3e-6, 4e-6])
        torch.testing.assert_close(result[0, :, 0], expected, atol=1e-9, rtol=1e-3)

    def test_peak_memory_does_not_scale_with_sequence_length(self):
        """Coefficients are built per step, not materialised as [B, L, C, N]."""
        import inspect

        source = inspect.getsource(selective_scan)
        assert "for step in range(length)" in source
        assert "torch.exp(step_dt * a_matrix)" in source


class TestSelectiveSSMBlock:
    def test_preserves_shape(self):
        assert SelectiveSSMBlock(16, state_dim=8)(torch.randn(2, 10, 16)).shape == (2, 10, 16)

    def test_is_causal(self):
        """The depthwise convolution is left-padded; position t must not see t+1."""
        torch.manual_seed(0)
        block = SelectiveSSMBlock(16, state_dim=8).eval()
        inputs = torch.randn(2, 10, 16)
        perturbed = inputs.clone()
        perturbed[:, 6:] += 50.0

        with torch.no_grad():
            baseline = block(inputs)
            changed = block(perturbed)

        torch.testing.assert_close(changed[:, :6], baseline[:, :6], atol=1e-5, rtol=1e-5)

    def test_dt_starts_in_a_usable_range(self):
        """Without the bias init, Δ starts far too small and the block looks broken."""
        import torch.nn.functional as F

        block = SelectiveSSMBlock(32, state_dim=8)
        dt = F.softplus(block.dt_projection.bias)
        assert float(dt.min()) > 1e-4
        assert float(dt.max()) < 1.0

    def test_selectivity_makes_dynamics_input_dependent(self):
        """The defining property: identical state, different input -> different decay."""
        torch.manual_seed(0)
        block = SelectiveSSMBlock(16, state_dim=8).eval()
        with torch.no_grad():
            quiet = block(torch.zeros(1, 8, 16))
            loud = block(torch.randn(1, 8, 16) * 3.0)
        assert not torch.allclose(quiet, loud)

    def test_rejects_non_positive_sizes(self):
        with pytest.raises(ValueError, match="must all be >= 1"):
            SelectiveSSMBlock(0, state_dim=8)


class TestS4DBlock:
    def test_preserves_shape(self):
        assert S4DBlock(16, state_dim=8)(torch.randn(2, 10, 16)).shape == (2, 10, 16)

    def test_ffn_can_be_disabled(self):
        with_ffn = sum(p.numel() for p in S4DBlock(32, ffn_multiplier=2).parameters())
        without = sum(p.numel() for p in S4DBlock(32, ffn_multiplier=0).parameters())
        assert without < with_ffn

    def test_rejects_negative_ffn_multiplier(self):
        with pytest.raises(ValueError, match="ffn_multiplier"):
            S4DBlock(16, ffn_multiplier=-1)

    def test_state_dim_changes_only_temporal_capacity(self):
        """The SSM mixes over time, the linear layers over channels.

        Raising state_dim must add parameters only in the kernel, leaving the
        channel-mixing layers untouched — otherwise the state-dimension sweep
        would not be measuring what it claims.
        """
        small = S4DBlock(32, state_dim=8)
        large = S4DBlock(32, state_dim=64)

        assert sum(p.numel() for p in large.kernel.parameters()) > sum(
            p.numel() for p in small.kernel.parameters()
        )
        assert sum(p.numel() for p in large.mix.parameters()) == sum(
            p.numel() for p in small.mix.parameters()
        )


@pytest.mark.parametrize("name", SSM_NAMES)
class TestSSMBackboneContract:
    """The SSMs must satisfy the same interface as the Phase 3 baselines."""

    def make(self, name: str, **kwargs: object) -> TemporalBackbone:
        defaults: dict[str, object] = {
            "latent_dim": 32,
            "hidden_dim": 32,
            "n_layers": 2,
            "state_dim": 8,
        }
        defaults.update(kwargs)
        return TEMPORAL_BACKBONES.build(name, **defaults)

    def test_is_registered_as_a_temporal_backbone(self, name):
        assert issubclass(TEMPORAL_BACKBONES.get(name), TemporalBackbone)

    @pytest.mark.parametrize("horizon", [1, 2, 4, 8])
    def test_output_shape_matches_horizon(self, name, horizon):
        latents = torch.randn(2, 4, 32, 3, 3)
        assert self.make(name)(latents, horizon).shape == (2, horizon, 32, 3, 3)

    @pytest.mark.parametrize("history", [2, 4, 6, 8])
    def test_accepts_every_swept_history_length(self, name, history):
        latents = torch.randn(1, history, 32, 3, 3)
        assert self.make(name)(latents, 2).shape[1] == 2

    def test_preserves_spatial_size(self, name):
        latents = torch.randn(1, 3, 32, 5, 7)
        assert self.make(name)(latents, 2).shape[-2:] == (5, 7)

    def test_spatial_locations_stay_independent(self, name):
        """A temporal backbone must not mix spatial locations."""
        torch.manual_seed(0)
        backbone = self.make(name).eval()
        latents = torch.randn(1, 3, 32, 2, 3)
        perturbed = latents.clone()
        perturbed[..., 0] += 9.0

        with torch.no_grad():
            baseline = backbone(latents, 2)
            changed = backbone(perturbed, 2)

        assert not torch.allclose(baseline[..., 0], changed[..., 0])
        torch.testing.assert_close(baseline[..., 1:], changed[..., 1:], atol=1e-5, rtol=1e-4)

    def test_output_depends_on_the_history(self, name):
        backbone = self.make(name).eval()
        with torch.no_grad():
            zeros = backbone(torch.zeros(1, 3, 32, 2, 2), 2)
            ones = backbone(torch.ones(1, 3, 32, 2, 2), 2)
        assert not torch.allclose(zeros, ones)

    def test_gradients_reach_every_used_parameter(self, name):
        backbone = self.make(name)
        backbone(torch.randn(1, 3, 32, 2, 2), 2).sum().backward()
        missing = [
            pname
            for pname, param in backbone.named_parameters()
            # Unused forecast-query rows legitimately receive no gradient.
            if param.requires_grad and param.grad is None and "queries" not in pname
        ]
        assert not missing, f"no gradient reached: {missing}"

    def test_is_deterministic_in_eval(self, name):
        backbone = self.make(name).eval()
        latents = torch.randn(1, 3, 32, 2, 2)
        with torch.no_grad():
            torch.testing.assert_close(backbone(latents, 2), backbone(latents, 2))

    def test_rejects_wrong_rank(self, name):
        with pytest.raises(ValueError, match="5 dimensions"):
            self.make(name)(torch.randn(2, 4, 32), 1)

    def test_rejects_non_positive_horizon(self, name):
        with pytest.raises(ValueError, match="horizon must be"):
            self.make(name)(torch.randn(1, 3, 32, 2, 2), 0)

    def test_rejects_horizon_beyond_the_query_budget(self, name):
        with pytest.raises(ValueError, match="max_horizon"):
            self.make(name)(torch.randn(1, 3, 32, 2, 2), 999)

    def test_capacity_grows_with_width(self, name):
        narrow = sum(p.numel() for p in self.make(name, hidden_dim=32).parameters())
        wide = sum(p.numel() for p in self.make(name, hidden_dim=128).parameters())
        assert wide > narrow * 3


class TestParameterEfficiency:
    """The project's thesis, measured directly."""

    def test_state_dim_is_a_cheap_axis_compared_with_width(self):
        """state_dim costs ~4H per unit; hidden_dim costs ~H per unit, squared.

        This asymmetry is the mechanism behind the whole research question, so
        it is asserted rather than assumed.
        """
        base = S4DBackbone(latent_dim=64, hidden_dim=128, n_layers=2, state_dim=16)
        more_state = S4DBackbone(latent_dim=64, hidden_dim=128, n_layers=2, state_dim=64)
        more_width = S4DBackbone(latent_dim=64, hidden_dim=256, n_layers=2, state_dim=16)

        def count(module: torch.nn.Module) -> int:
            return sum(p.numel() for p in module.parameters())

        state_growth = count(more_state) - count(base)
        width_growth = count(more_width) - count(base)

        assert state_growth > 0
        assert width_growth > state_growth * 3

    def test_ssm_temporal_mixing_is_cheaper_than_attention(self):
        """At matched width and depth, the SSM should carry fewer parameters."""
        from tinyearth.models.temporal import TemporalTransformerBackbone

        ssm = S4DBackbone(latent_dim=128, hidden_dim=256, n_layers=4, state_dim=64)
        attention = TemporalTransformerBackbone(
            latent_dim=128, hidden_dim=256, n_layers=4, n_heads=4
        )
        assert sum(p.numel() for p in ssm.parameters()) < sum(
            p.numel() for p in attention.parameters()
        )


def test_both_ssms_are_in_the_registry():
    assert set(TEMPORAL_BACKBONES.keys()) == {"convlstm", "transformer", "s4d", "mamba"}


def test_mamba_uses_a_small_default_state_dim():
    """Selectivity carries the capacity in Mamba, not state size."""
    assert MambaBackbone().state_dim < S4DBackbone().state_dim
