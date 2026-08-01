"""Tests for forecast rendering.

The property that matters most is that a stretch computed from the truth is
applied unchanged to the prediction. Restretching each panel independently is
the standard way a figure flatters a model, so it is worth a test rather than a
comment.
"""

from __future__ import annotations

import pytest
import torch

from tinyearth.evaluation.visualization import (
    NDVI_RANGE,
    composite_rgb,
    ndvi,
    stretch_limits,
)


class TestStretchLimits:
    def test_returns_one_limit_per_channel(self):
        low, high = stretch_limits(torch.rand(5, 4, 8, 8))
        assert low.shape == (4,)
        assert high.shape == (4,)

    def test_limits_are_ordered(self):
        low, high = stretch_limits(torch.rand(5, 4, 8, 8))
        assert (high > low).all()

    def test_percentiles_ignore_outliers(self):
        """One specular pixel must not set the white point for the scene."""
        torch.manual_seed(0)
        images = torch.rand(1, 1, 32, 32) * 0.2
        images[0, 0, 0, 0] = 1000.0
        _, high = stretch_limits(images, 2.0, 98.0)
        assert high.item() < 1.0

    def test_a_constant_channel_does_not_divide_by_zero(self):
        images = torch.zeros(3, 4, 8, 8)
        low, high = stretch_limits(images)
        assert (high > low).all()
        assert torch.isfinite(composite_rgb(images, low, high)).all()

    def test_a_constant_channel_renders_mid_grey(self):
        """Rendering it black would read as real dark ground."""
        images = torch.full((3, 4, 8, 8), 0.3)
        rgb = composite_rgb(images, *stretch_limits(images), gamma=1.0)
        assert rgb.mean().item() == pytest.approx(0.5, abs=0.01)

    @pytest.mark.parametrize("lower,upper", [(50.0, 50.0), (-1.0, 90.0), (10.0, 101.0)])
    def test_rejects_bad_percentiles(self, lower: float, upper: float):
        with pytest.raises(ValueError, match="Percentiles must satisfy"):
            stretch_limits(torch.rand(2, 3, 4, 4), lower, upper)


class TestComposite:
    def test_shape_and_range(self):
        images = torch.rand(6, 4, 16, 16)
        rgb = composite_rgb(images, *stretch_limits(images))
        assert rgb.shape == (6, 16, 16, 3)
        assert rgb.min() >= 0.0
        assert rgb.max() <= 1.0

    def test_channels_are_reordered_to_true_colour(self):
        """Source order is (blue, green, red, nir); display order is (r, g, b)."""
        images = torch.zeros(1, 4, 2, 2)
        images[0, 2] = 1.0  # red band
        low = torch.zeros(4)
        high = torch.ones(4)
        rgb = composite_rgb(images, low, high, gamma=1.0)
        assert rgb[0, 0, 0, 0] == pytest.approx(1.0)  # lands in the red display channel
        assert rgb[0, 0, 0, 1] == pytest.approx(0.0)
        assert rgb[0, 0, 0, 2] == pytest.approx(0.0)

    def test_one_stretch_applies_unchanged_to_a_second_sequence(self):
        """The anti-flattery property: a dimmer prediction must render dimmer."""
        truth = torch.full((3, 4, 8, 8), 0.4)
        prediction = torch.full((3, 4, 8, 8), 0.2)
        low, high = stretch_limits(truth)
        assert composite_rgb(prediction, low, high).mean() < composite_rgb(truth, low, high).mean()

    def test_values_outside_the_stretch_are_clipped_not_wrapped(self):
        images = torch.full((1, 4, 2, 2), 5.0)
        rgb = composite_rgb(images, torch.zeros(4), torch.ones(4))
        assert rgb.max() <= 1.0

    def test_rejects_too_few_channels(self):
        with pytest.raises(ValueError, match="at least 3 channels"):
            composite_rgb(torch.rand(2, 2, 4, 4), torch.zeros(2), torch.ones(2))


class TestNDVI:
    def test_shape_drops_the_channel_axis(self):
        assert ndvi(torch.rand(5, 4, 16, 16)).shape == (5, 16, 16)

    def test_dense_vegetation_is_high(self):
        """High NIR against low red is the signature of a healthy canopy."""
        images = torch.zeros(1, 4, 1, 1)
        images[0, 2] = 0.05  # red
        images[0, 3] = 0.45  # nir
        assert ndvi(images).item() == pytest.approx(0.8, abs=0.01)

    def test_bare_ground_is_near_zero(self):
        images = torch.zeros(1, 4, 1, 1)
        images[0, 2] = 0.3
        images[0, 3] = 0.3
        assert ndvi(images).item() == pytest.approx(0.0, abs=0.01)

    def test_a_blanked_pixel_stays_finite(self):
        """Cloud-masked pixels are zero in both bands; 0/0 must not be NaN."""
        assert torch.isfinite(ndvi(torch.zeros(2, 4, 4, 4))).all()

    def test_stays_within_bounds(self):
        values = ndvi(torch.rand(20, 4, 8, 8))
        assert values.min() >= -1.0
        assert values.max() <= 1.0

    def test_rejects_missing_nir(self):
        with pytest.raises(ValueError, match="needs the NIR channel"):
            ndvi(torch.rand(2, 3, 4, 4))

    def test_display_range_brackets_real_land_cover(self):
        low, high = NDVI_RANGE
        assert low < 0.0 < high <= 1.0
