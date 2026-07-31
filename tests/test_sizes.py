"""Tests for the calibrated size tiers.

The scaling study compares backbones at matched parameter budgets. If the stored
widths drift from what the models actually build — because a block structure
changed — the "2M model" and the "20M model" stop meaning what they say, and the
scaling curve silently becomes uninterpretable.

These tests rebuild every tier and check it against its target, so that drift
fails the suite instead of quietly corrupting a result.
"""

from __future__ import annotations

import pytest

from tinyearth.models.decoders import CNNDecoder
from tinyearth.models.encoders import CNNEncoder
from tinyearth.models.forecaster import count_parameters
from tinyearth.models.sizes import (
    CALIBRATION_LATENT_DIM,
    CALIBRATION_LAYERS,
    SIZE_TIERS,
    TARGET_PARAMETERS,
    available_sizes,
    resolve_hidden_dim,
)
from tinyearth.models.temporal import TEMPORAL_BACKBONES

# The calibration was performed with the standard encoder/decoder.
FIXED_KWARGS: dict[str, dict[str, object]] = {
    "s4d": {"state_dim": 64},
    "mamba": {"state_dim": 16},
    "convlstm": {},
    "transformer": {"n_heads": 4},
}
TOLERANCE = 0.12
"""Allowed relative deviation from the nominal target.

Generous on purpose: hitting 2,000,000 exactly would require a width the search
grid does not offer. What matters is that the tiers are *comparable across
backbones*, not that they are exact.
"""


def build_total(backbone: str, hidden_dim: int) -> int:
    """Count total parameters for a backbone at a given width."""
    encoder = CNNEncoder(4, CALIBRATION_LATENT_DIM, base_channels=32, depth=2)
    decoder = CNNDecoder(4, CALIBRATION_LATENT_DIM, base_channels=32, depth=2)
    module = TEMPORAL_BACKBONES.build(
        backbone,
        latent_dim=CALIBRATION_LATENT_DIM,
        hidden_dim=hidden_dim,
        n_layers=CALIBRATION_LAYERS,
        **FIXED_KWARGS[backbone],
    )
    return count_parameters(encoder) + count_parameters(decoder) + count_parameters(module)


class TestTierTable:
    def test_every_registered_backbone_is_calibrated(self):
        """A new backbone without a calibration row cannot join the scaling study."""
        assert set(SIZE_TIERS) == set(TEMPORAL_BACKBONES.keys())

    def test_every_backbone_covers_every_tier(self):
        for backbone, tiers in SIZE_TIERS.items():
            assert set(tiers) == set(TARGET_PARAMETERS), backbone

    def test_tiers_are_ordered_smallest_first(self):
        targets = list(TARGET_PARAMETERS.values())
        assert targets == sorted(targets)

    def test_widths_increase_with_tier(self):
        for backbone, tiers in SIZE_TIERS.items():
            widths = [tiers[size] for size in available_sizes()]
            assert widths == sorted(widths), backbone


@pytest.mark.parametrize("backbone", sorted(SIZE_TIERS))
@pytest.mark.parametrize("size", available_sizes())
class TestCalibrationAccuracy:
    def test_tier_lands_near_its_target(self, backbone, size):
        target = TARGET_PARAMETERS[size]
        total = build_total(backbone, resolve_hidden_dim(backbone, size))
        relative = abs(total - target) / target

        assert relative <= TOLERANCE, (
            f"{backbone}/{size}: {total:,} parameters is {100 * relative:.1f}% from the "
            f"{target:,} target. Rerun `python scripts/calibrate_sizes.py` and update "
            "SIZE_TIERS."
        )

    def test_backbone_holds_most_of_the_budget(self, backbone, size):
        """Scaling must happen in the component under study, not in fixed parts."""
        encoder = count_parameters(CNNEncoder(4, CALIBRATION_LATENT_DIM, base_channels=32, depth=2))
        decoder = count_parameters(CNNDecoder(4, CALIBRATION_LATENT_DIM, base_channels=32, depth=2))
        total = build_total(backbone, resolve_hidden_dim(backbone, size))

        assert (total - encoder - decoder) / total > 0.8


class TestTiersAreComparable:
    @pytest.mark.parametrize("size", available_sizes())
    def test_all_backbones_land_within_a_narrow_band_at_each_tier(self, size):
        """The point of matched budgets: a tier must mean the same everywhere.

        If one backbone's "small" were 4M and another's 6M, a quality comparison
        at that tier would be measuring parameter count, not architecture.
        """
        totals = [
            build_total(backbone, resolve_hidden_dim(backbone, size)) for backbone in SIZE_TIERS
        ]
        spread = (max(totals) - min(totals)) / min(totals)
        assert spread < 0.20, f"{size}: totals span {spread:.0%} — tiers are not comparable"


class TestResolveHiddenDim:
    def test_returns_the_stored_width(self):
        assert resolve_hidden_dim("s4d", "base") == SIZE_TIERS["s4d"]["base"]

    def test_unknown_backbone_points_at_the_calibration_script(self):
        with pytest.raises(KeyError, match="calibrate_sizes"):
            resolve_hidden_dim("mystery_net", "tiny")

    def test_unknown_size_lists_the_options(self):
        with pytest.raises(KeyError, match="Available"):
            resolve_hidden_dim("s4d", "enormous")

    def test_ssms_afford_more_width_than_convlstm_at_the_same_budget(self):
        """The parameter-efficiency claim, visible before any training.

        A ConvLSTM spends 4*9*(2H)*H on its gate convolution; a diagonal SSM
        spends roughly 5H^2 + 4HN. At a matched budget the SSM buys far more
        width — which is the mechanism the whole project is testing.
        """
        for size in available_sizes():
            assert resolve_hidden_dim("s4d", size) > 2 * resolve_hidden_dim("convlstm", size)
