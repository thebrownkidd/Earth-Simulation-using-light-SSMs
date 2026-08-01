"""Tests for the parameter-free reference forecasts.

These exist to make a learned MAE interpretable, so they have to be right in
the same way a metric has to be right: an over-strong or over-weak baseline
would silently distort every comparison drawn against it.
"""

from __future__ import annotations

import pytest
import torch

from tinyearth.evaluation.references import (
    REFERENCE_FORECASTS,
    climatology_forecast,
    persistence_forecast,
)


@pytest.fixture
def images() -> torch.Tensor:
    torch.manual_seed(0)
    return torch.rand(2, 4, 3, 8, 8)


class TestPersistence:
    def test_shape(self, images: torch.Tensor):
        assert persistence_forecast(images, 5).shape == (2, 5, 3, 8, 8)

    def test_repeats_the_last_frame(self, images: torch.Tensor):
        forecast = persistence_forecast(images, 3)
        for step in range(3):
            assert torch.equal(forecast[:, step], images[:, -1])

    def test_mask_selects_the_last_observed_frame(self):
        """A pixel blanked by cloud must persist from when it was last seen."""
        images = torch.zeros(1, 3, 1, 1, 1)
        images[0, 0, 0, 0, 0] = 0.7  # observed
        images[0, 1, 0, 0, 0] = 0.4  # observed
        images[0, 2, 0, 0, 0] = 0.0  # clouded, blanked to zero
        mask = torch.tensor([1.0, 1.0, 0.0]).view(1, 3, 1, 1, 1)

        assert persistence_forecast(images, 2, mask=mask)[0, 0, 0, 0, 0] == pytest.approx(0.4)
        # Without the mask the baseline persists the cloud-blanked zero.
        assert persistence_forecast(images, 2)[0, 0, 0, 0, 0] == pytest.approx(0.0)

    def test_mask_is_applied_per_pixel(self):
        """Different pixels cloud over at different times."""
        images = torch.zeros(1, 2, 1, 1, 2)
        images[0, 0, 0, 0, :] = torch.tensor([0.1, 0.2])
        images[0, 1, 0, 0, :] = torch.tensor([0.8, 0.9])
        mask = torch.ones(1, 2, 1, 1, 2)
        mask[0, 1, 0, 0, 0] = 0.0  # first pixel clouded in the final frame

        forecast = persistence_forecast(images, 1, mask=mask)
        assert forecast[0, 0, 0, 0, 0] == pytest.approx(0.1)
        assert forecast[0, 0, 0, 0, 1] == pytest.approx(0.9)

    def test_a_never_valid_pixel_falls_back_rather_than_failing(self):
        images = torch.full((1, 3, 1, 1, 1), 0.5)
        forecast = persistence_forecast(images, 2, mask=torch.zeros(1, 3, 1, 1, 1))
        assert torch.isfinite(forecast).all()


class TestClimatology:
    def test_shape(self, images: torch.Tensor):
        assert climatology_forecast(images, 5).shape == (2, 5, 3, 8, 8)

    def test_repeats_the_context_mean(self, images: torch.Tensor):
        forecast = climatology_forecast(images, 3)
        assert torch.allclose(forecast[:, 0], images.mean(dim=1))

    def test_every_step_is_identical(self, images: torch.Tensor):
        forecast = climatology_forecast(images, 4)
        for step in range(1, 4):
            assert torch.equal(forecast[:, step], forecast[:, 0])

    def test_mask_excludes_blanked_frames_from_the_mean(self):
        images = torch.zeros(1, 3, 1, 1, 1)
        images[0, 0, 0, 0, 0] = 0.6
        images[0, 1, 0, 0, 0] = 0.4
        images[0, 2, 0, 0, 0] = 0.0  # clouded
        mask = torch.tensor([1.0, 1.0, 0.0]).view(1, 3, 1, 1, 1)

        assert climatology_forecast(images, 1, mask=mask)[0, 0, 0, 0, 0] == pytest.approx(0.5)
        # Unmasked, the blanked zero drags the mean down.
        assert climatology_forecast(images, 1)[0, 0, 0, 0, 0] == pytest.approx(1.0 / 3.0)

    def test_a_fully_masked_pixel_stays_finite(self):
        images = torch.full((1, 3, 1, 1, 1), 0.5)
        forecast = climatology_forecast(images, 2, mask=torch.zeros(1, 3, 1, 1, 1))
        assert torch.isfinite(forecast).all()
        assert forecast[0, 0, 0, 0, 0] == pytest.approx(0.5)


@pytest.mark.parametrize("name", sorted(REFERENCE_FORECASTS))
class TestContract:
    def test_rejects_a_non_positive_horizon(self, name: str, images: torch.Tensor):
        with pytest.raises(ValueError, match="horizon must be >= 1"):
            REFERENCE_FORECASTS[name](images, 0)

    def test_rejects_the_wrong_rank(self, name: str):
        with pytest.raises(ValueError, match=r"\[B, T, C, H, W\]"):
            REFERENCE_FORECASTS[name](torch.rand(4, 3, 8, 8), 2)

    def test_output_is_writable(self, name: str, images: torch.Tensor):
        """`expand` returns a read-only view; callers get a real tensor."""
        forecast = REFERENCE_FORECASTS[name](images, 3)
        forecast[0, 0, 0, 0, 0] = 0.123  # must not raise
        assert forecast.is_contiguous()

    def test_a_static_scene_is_predicted_exactly(self, name: str):
        """Both references are exact when nothing moves -- the sanity anchor."""
        frame = torch.rand(2, 1, 3, 8, 8)
        images = frame.expand(-1, 5, -1, -1, -1).contiguous()
        forecast = REFERENCE_FORECASTS[name](images, 4)
        assert torch.allclose(forecast, frame.expand(-1, 4, -1, -1, -1), atol=1e-6)
