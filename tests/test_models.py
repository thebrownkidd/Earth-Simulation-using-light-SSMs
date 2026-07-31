"""Tests for model components and the assembled forecaster.

The most important tests here are the ones asserting that encoder and decoder
parameter counts are **identical across backbones**. That is the controlled
comparison the whole project rests on: if the fixed components differ between
two runs, a difference in results is not attributable to the backbone.
"""

from __future__ import annotations

import pytest
import torch

from tinyearth.models.base import Decoder, Encoder, TemporalBackbone
from tinyearth.models.decoders import DECODERS, CNNDecoder
from tinyearth.models.encoders import ENCODERS, CNNEncoder
from tinyearth.models.forecaster import Forecaster, count_parameters
from tinyearth.models.layers import build_activation, build_norm
from tinyearth.models.temporal import (
    TEMPORAL_BACKBONES,
    ConvLSTMBackbone,
    ConvLSTMCell,
    SinusoidalPositionalEncoding,
    TemporalTransformerBackbone,
)

BATCH, HISTORY, CHANNELS, SIZE, LATENT = 2, 4, 4, 16, 16
BACKBONE_NAMES = ("convlstm", "transformer")


def make_backbone(name: str, latent_dim: int = LATENT, **kwargs: object) -> TemporalBackbone:
    """Build a small backbone by registry name."""
    defaults: dict[str, object] = {"hidden_dim": 16, "n_layers": 1}
    if name == "transformer":
        defaults["n_heads"] = 2
    defaults.update(kwargs)
    return TEMPORAL_BACKBONES.build(name, latent_dim=latent_dim, **defaults)


def make_forecaster(name: str, horizon: int = 2, depth: int = 2) -> Forecaster:
    """Assemble a small forecaster with the named backbone."""
    return Forecaster(
        encoder=CNNEncoder(CHANNELS, LATENT, base_channels=8, depth=depth),
        backbone=make_backbone(name),
        decoder=CNNDecoder(CHANNELS, LATENT, base_channels=8, depth=depth),
        horizon=horizon,
    )


@pytest.fixture
def images() -> torch.Tensor:
    torch.manual_seed(0)
    return torch.rand(BATCH, HISTORY, CHANNELS, SIZE, SIZE)


class TestLayers:
    def test_build_activation(self):
        assert isinstance(build_activation("gelu"), torch.nn.GELU)

    def test_unknown_activation_lists_options(self):
        with pytest.raises(ValueError, match="Unknown activation"):
            build_activation("swish9000")

    def test_group_norm_picks_a_divisor(self):
        norm = build_norm("group", 12)
        assert isinstance(norm, torch.nn.GroupNorm)
        assert 12 % norm.num_groups == 0

    def test_group_norm_handles_awkward_channel_counts(self):
        """A latent_dim sweep must not crash on a prime channel count."""
        assert isinstance(build_norm("group", 7), torch.nn.GroupNorm)

    def test_none_norm_is_identity(self):
        assert isinstance(build_norm("none", 8), torch.nn.Identity)

    def test_unknown_norm(self):
        with pytest.raises(ValueError, match="Unknown norm"):
            build_norm("layer2d", 8)


class TestEncoder:
    def test_output_shape(self, images):
        encoder = CNNEncoder(CHANNELS, LATENT, base_channels=8, depth=2)
        latents = encoder(images)
        assert latents.shape == (BATCH, HISTORY, LATENT, SIZE // 4, SIZE // 4)

    @pytest.mark.parametrize("depth", [0, 1, 2, 3])
    def test_downsample_matches_depth(self, images, depth):
        encoder = CNNEncoder(CHANNELS, LATENT, base_channels=8, depth=depth)
        assert encoder.downsample == 2**depth
        assert encoder(images).shape[-1] == SIZE // 2**depth

    def test_latent_dim_is_independent_of_base_channels(self, images):
        for base in (8, 16, 32):
            encoder = CNNEncoder(CHANNELS, 24, base_channels=base, depth=1)
            assert encoder(images).shape[2] == 24

    def test_frames_are_encoded_independently(self):
        """No temporal parameters: shuffling frames must permute the output."""
        encoder = CNNEncoder(CHANNELS, LATENT, base_channels=8, depth=1).eval()
        torch.manual_seed(0)
        images = torch.rand(1, 3, CHANNELS, SIZE, SIZE)

        with torch.no_grad():
            straight = encoder(images)
            reversed_ = encoder(images.flip(dims=[1]))

        torch.testing.assert_close(reversed_, straight.flip(dims=[1]))

    def test_rejects_wrong_channel_count(self):
        encoder = CNNEncoder(in_channels=4, latent_dim=LATENT, base_channels=8, depth=1)
        with pytest.raises(ValueError, match="expects 4 channels"):
            encoder(torch.rand(1, 2, 3, SIZE, SIZE))

    def test_rejects_wrong_rank(self):
        encoder = CNNEncoder(CHANNELS, LATENT, base_channels=8, depth=1)
        with pytest.raises(ValueError, match="5 dimensions"):
            encoder(torch.rand(2, CHANNELS, SIZE, SIZE))

    def test_rejects_negative_depth(self):
        with pytest.raises(ValueError, match="depth must be"):
            CNNEncoder(CHANNELS, LATENT, depth=-1)

    def test_is_registered(self):
        assert "cnn" in ENCODERS
        assert issubclass(ENCODERS.get("cnn"), Encoder)


class TestDecoder:
    def test_restores_the_input_resolution(self):
        decoder = CNNDecoder(CHANNELS, LATENT, base_channels=8, depth=2)
        latents = torch.rand(BATCH, 3, LATENT, SIZE // 4, SIZE // 4)
        assert decoder(latents).shape == (BATCH, 3, CHANNELS, SIZE, SIZE)

    def test_sigmoid_bounds_the_output(self):
        decoder = CNNDecoder(CHANNELS, LATENT, base_channels=8, depth=1)
        output = decoder(torch.randn(1, 2, LATENT, SIZE // 2, SIZE // 2) * 50)
        assert float(output.min()) >= 0.0
        assert float(output.max()) <= 1.0

    def test_none_activation_allows_unbounded_output(self):
        decoder = CNNDecoder(CHANNELS, LATENT, base_channels=8, depth=1, output_activation="none")
        output = decoder(torch.randn(1, 2, LATENT, SIZE // 2, SIZE // 2) * 50)
        assert float(output.max()) > 1.0 or float(output.min()) < 0.0

    def test_rejects_unknown_output_activation(self):
        with pytest.raises(ValueError, match="Unknown output_activation"):
            CNNDecoder(CHANNELS, LATENT, output_activation="softmax")

    def test_rejects_wrong_latent_dim(self):
        decoder = CNNDecoder(CHANNELS, latent_dim=32, base_channels=8, depth=1)
        with pytest.raises(ValueError, match="expects latent_dim=32"):
            decoder(torch.rand(1, 2, 16, SIZE // 2, SIZE // 2))

    def test_is_registered(self):
        assert "cnn" in DECODERS
        assert issubclass(DECODERS.get("cnn"), Decoder)


class TestConvLSTMCell:
    def test_preserves_spatial_size(self):
        cell = ConvLSTMCell(8, 12)
        state = cell.initial_state(2, (SIZE, SIZE), torch.device("cpu"), torch.float32)
        hidden, memory = cell(torch.rand(2, 8, SIZE, SIZE), state)
        assert hidden.shape == memory.shape == (2, 12, SIZE, SIZE)

    def test_rejects_even_kernel(self):
        with pytest.raises(ValueError, match="kernel_size must be odd"):
            ConvLSTMCell(4, 4, kernel_size=2)

    def test_forget_gate_bias_starts_at_one(self):
        """Guards the standard LSTM initialisation trick against a silent regression."""
        cell = ConvLSTMCell(4, 6)
        bias = cell.conv.bias
        assert bias is not None
        torch.testing.assert_close(bias[6:12], torch.ones(6))

    def test_initial_state_is_zero(self):
        cell = ConvLSTMCell(4, 6)
        hidden, memory = cell.initial_state(1, (4, 4), torch.device("cpu"), torch.float32)
        assert float(hidden.abs().sum()) == 0.0
        assert float(memory.abs().sum()) == 0.0

    def test_initial_state_tensors_are_not_aliased(self):
        """Hidden and cell must be separate tensors, or updates corrupt each other."""
        cell = ConvLSTMCell(4, 6)
        hidden, memory = cell.initial_state(1, (4, 4), torch.device("cpu"), torch.float32)
        hidden.add_(1.0)
        assert float(memory.abs().sum()) == 0.0


class TestPositionalEncoding:
    def test_shape(self):
        assert SinusoidalPositionalEncoding(16)(8).shape == (8, 16)

    def test_offset_selects_later_positions(self):
        encoding = SinusoidalPositionalEncoding(16)
        torch.testing.assert_close(encoding(4, offset=4), encoding(8)[4:])

    def test_positions_are_distinct(self):
        values = SinusoidalPositionalEncoding(16)(8)
        assert not torch.allclose(values[0], values[1])

    def test_rejects_odd_dim(self):
        with pytest.raises(ValueError, match="dim must be even"):
            SinusoidalPositionalEncoding(15)

    def test_rejects_out_of_range_request(self):
        encoding = SinusoidalPositionalEncoding(16, max_positions=10)
        with pytest.raises(ValueError, match="only 10 are available"):
            encoding(4, offset=8)

    def test_is_a_buffer_not_a_parameter(self):
        """It must move with .to(device) but never receive gradient."""
        encoding = SinusoidalPositionalEncoding(16)
        assert not list(encoding.parameters())
        assert "encoding" in dict(encoding.named_buffers())


@pytest.mark.parametrize("name", BACKBONE_NAMES)
class TestBackboneContract:
    """Every backbone must satisfy the same interface, or swapping breaks."""

    def test_is_registered_as_a_temporal_backbone(self, name):
        assert issubclass(TEMPORAL_BACKBONES.get(name), TemporalBackbone)

    @pytest.mark.parametrize("horizon", [1, 2, 4, 8])
    def test_output_shape_matches_horizon(self, name, horizon):
        backbone = make_backbone(name)
        latents = torch.rand(BATCH, HISTORY, LATENT, 4, 4)
        assert backbone(latents, horizon).shape == (BATCH, horizon, LATENT, 4, 4)

    @pytest.mark.parametrize("history", [2, 4, 6, 8])
    def test_accepts_every_swept_history_length(self, name, history):
        backbone = make_backbone(name)
        latents = torch.rand(1, history, LATENT, 4, 4)
        assert backbone(latents, 2).shape[1] == 2

    def test_preserves_spatial_size(self, name):
        backbone = make_backbone(name)
        latents = torch.rand(1, 3, LATENT, 6, 8)
        assert backbone(latents, 2).shape[-2:] == (6, 8)

    def test_rejects_wrong_rank(self, name):
        with pytest.raises(ValueError, match="5 dimensions"):
            make_backbone(name)(torch.rand(2, 4, LATENT), 1)

    def test_rejects_non_positive_horizon(self, name):
        with pytest.raises(ValueError, match="horizon must be"):
            make_backbone(name)(torch.rand(1, 3, LATENT, 4, 4), 0)

    def test_output_depends_on_the_input(self, name):
        """A backbone ignoring its history would still produce right-shaped output."""
        backbone = make_backbone(name).eval()
        with torch.no_grad():
            first = backbone(torch.zeros(1, 3, LATENT, 4, 4), 2)
            second = backbone(torch.ones(1, 3, LATENT, 4, 4), 2)
        assert not torch.allclose(first, second)

    def test_gradients_reach_every_parameter(self, name):
        backbone = make_backbone(name)
        backbone(torch.rand(1, 3, LATENT, 4, 4), 2).sum().backward()
        missing = [
            pname
            for pname, param in backbone.named_parameters()
            if param.requires_grad and (param.grad is None or float(param.grad.abs().sum()) == 0)
        ]
        assert not missing, f"no gradient reached: {missing}"

    def test_is_deterministic_in_eval(self, name):
        backbone = make_backbone(name).eval()
        latents = torch.rand(1, 3, LATENT, 4, 4)
        with torch.no_grad():
            torch.testing.assert_close(backbone(latents, 2), backbone(latents, 2))


class TestBackboneSpecifics:
    def test_convlstm_rejects_zero_layers(self):
        with pytest.raises(ValueError, match="n_layers must be"):
            ConvLSTMBackbone(latent_dim=LATENT, n_layers=0)

    def test_transformer_requires_heads_to_divide_width(self):
        with pytest.raises(ValueError, match="divisible by n_heads"):
            TemporalTransformerBackbone(latent_dim=LATENT, hidden_dim=18, n_heads=4)

    def test_transformer_treats_spatial_positions_independently(self):
        """Attention is over time only; locations must not mix.

        Changing one spatial column must leave the others' outputs untouched.
        """
        backbone = TemporalTransformerBackbone(
            latent_dim=LATENT, hidden_dim=16, n_layers=1, n_heads=2
        ).eval()
        latents = torch.rand(1, 3, LATENT, 2, 3)
        perturbed = latents.clone()
        perturbed[..., 0] += 5.0

        with torch.no_grad():
            base = backbone(latents, 2)
            other = backbone(perturbed, 2)

        assert not torch.allclose(base[..., 0], other[..., 0])
        torch.testing.assert_close(base[..., 1:], other[..., 1:])

    def test_convlstm_capacity_grows_with_hidden_dim(self):
        small = count_parameters(ConvLSTMBackbone(latent_dim=LATENT, hidden_dim=16))
        large = count_parameters(ConvLSTMBackbone(latent_dim=LATENT, hidden_dim=64))
        assert large > small * 3

    def test_registry_holds_the_baselines_and_the_ssms(self):
        """Phase 4 added s4d and mamba; the baselines remain."""
        assert set(BACKBONE_NAMES) <= set(TEMPORAL_BACKBONES.keys())
        assert set(TEMPORAL_BACKBONES.keys()) == {"convlstm", "transformer", "s4d", "mamba"}


@pytest.mark.parametrize("name", BACKBONE_NAMES)
class TestForecaster:
    def test_round_trips_to_the_input_resolution(self, name, images):
        model = make_forecaster(name, horizon=2)
        assert model(images).shape == (BATCH, 2, CHANNELS, SIZE, SIZE)

    def test_horizon_can_be_overridden_per_call(self, name, images):
        """The horizon sweep must not require rebuilding the model."""
        model = make_forecaster(name, horizon=2)
        assert model(images, horizon=5).shape[1] == 5

    def test_output_is_bounded_by_the_sigmoid_decoder(self, name, images):
        output = make_forecaster(name)(images)
        assert float(output.min()) >= 0.0
        assert float(output.max()) <= 1.0

    def test_gradients_reach_all_three_components(self, name, images):
        model = make_forecaster(name)
        model(images).sum().backward()
        for component in ("encoder", "backbone", "decoder"):
            grads = [
                p.grad is not None
                for p in getattr(model, component).parameters()
                if p.requires_grad
            ]
            assert grads and all(grads), f"{component} received no gradient"

    def test_parameter_breakdown_sums_to_total(self, name):
        breakdown = make_forecaster(name).parameter_breakdown()
        assert breakdown.encoder + breakdown.backbone + breakdown.decoder == breakdown.total

    def test_backbone_holds_a_meaningful_share(self, name):
        """Scaling happens in the backbone; it must be a substantial fraction."""
        assert make_forecaster(name).parameter_breakdown().backbone_fraction > 0.2


class TestControlledComparison:
    """The fixed components must be identical across backbones."""

    def test_encoder_and_decoder_sizes_match_across_backbones(self):
        breakdowns = {name: make_forecaster(name).parameter_breakdown() for name in BACKBONE_NAMES}
        encoders = {b.encoder for b in breakdowns.values()}
        decoders = {b.decoder for b in breakdowns.values()}
        assert len(encoders) == 1, f"encoder sizes differ across backbones: {breakdowns}"
        assert len(decoders) == 1, f"decoder sizes differ across backbones: {breakdowns}"

    def test_only_the_backbone_total_differs(self):
        first, second = (make_forecaster(name).parameter_breakdown() for name in BACKBONE_NAMES)
        assert first.backbone != second.backbone
        assert first.total - first.backbone == second.total - second.backbone


class TestForecasterValidation:
    def test_rejects_latent_dim_mismatch(self):
        with pytest.raises(ValueError, match="disagree on latent_dim"):
            Forecaster(
                encoder=CNNEncoder(CHANNELS, 16, base_channels=8, depth=1),
                backbone=make_backbone("convlstm", latent_dim=32),
                decoder=CNNDecoder(CHANNELS, 16, base_channels=8, depth=1),
            )

    def test_rejects_depth_mismatch(self):
        with pytest.raises(ValueError, match="downsamples by"):
            Forecaster(
                encoder=CNNEncoder(CHANNELS, LATENT, base_channels=8, depth=2),
                backbone=make_backbone("convlstm"),
                decoder=CNNDecoder(CHANNELS, LATENT, base_channels=8, depth=1),
            )

    def test_rejects_non_positive_horizon(self):
        with pytest.raises(ValueError, match="horizon must be"):
            make_forecaster("convlstm", horizon=0)


def test_count_parameters_respects_trainable_only():
    encoder = CNNEncoder(CHANNELS, LATENT, base_channels=8, depth=1)
    total = count_parameters(encoder, trainable_only=False)
    for param in encoder.parameters():
        param.requires_grad = False

    assert count_parameters(encoder, trainable_only=True) == 0
    assert count_parameters(encoder, trainable_only=False) == total
