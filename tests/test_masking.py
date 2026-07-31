"""Tests for cloud-mask policy."""

from __future__ import annotations

import pytest
import torch

from tinyearth.datasets.masking import (
    MaskPolicy,
    apply_mask_policy,
    masked_fraction,
    passes_validity_threshold,
)


@pytest.fixture
def images() -> torch.Tensor:
    return torch.full((2, 3, 4, 4), 0.5)


@pytest.fixture
def half_valid() -> torch.Tensor:
    mask = torch.ones(2, 1, 4, 4)
    mask[:, :, :2, :] = 0.0  # top half invalid
    return mask


class TestFillPolicies:
    def test_keep_leaves_values_untouched(self, images, half_valid):
        result = apply_mask_policy(images, half_valid, MaskPolicy.KEEP)
        torch.testing.assert_close(result, images)

    def test_keep_returns_a_copy_not_the_input(self, images, half_valid):
        result = apply_mask_policy(images, half_valid, MaskPolicy.KEEP)
        result[0, 0, 0, 0] = 99.0
        assert float(images[0, 0, 0, 0]) == 0.5

    def test_zero_clears_invalid_pixels_only(self, images, half_valid):
        result = apply_mask_policy(images, half_valid, MaskPolicy.ZERO)
        assert float(result[:, :, :2, :].max()) == 0.0
        assert float(result[:, :, 2:, :].min()) == 0.5

    def test_mean_fills_with_the_valid_mean(self, images, half_valid):
        images = images.clone()
        images[:, :, 2:, :] = 0.8  # valid region has a distinct value
        result = apply_mask_policy(images, half_valid, MaskPolicy.MEAN)
        assert float(result[0, 0, 0, 0]) == pytest.approx(0.8)

    def test_mean_preserves_valid_pixels(self, images, half_valid):
        result = apply_mask_policy(images, half_valid, MaskPolicy.MEAN)
        torch.testing.assert_close(result[:, :, 2:, :], images[:, :, 2:, :])

    def test_mean_is_per_channel(self):
        images = torch.zeros(1, 2, 2, 2)
        images[0, 0] = 0.2
        images[0, 1] = 0.9
        mask = torch.ones(1, 1, 2, 2)
        mask[0, 0, 0, 0] = 0.0

        result = apply_mask_policy(images, mask, MaskPolicy.MEAN)
        assert float(result[0, 0, 0, 0]) == pytest.approx(0.2)
        assert float(result[0, 1, 0, 0]) == pytest.approx(0.9)

    def test_mean_falls_back_to_zero_when_a_frame_is_fully_invalid(self, images):
        mask = torch.zeros(2, 1, 4, 4)
        result = apply_mask_policy(images, mask, MaskPolicy.MEAN)
        assert float(result.abs().max()) == 0.0

    def test_fully_valid_mask_is_a_no_op_for_every_policy(self, images):
        mask = torch.ones(2, 1, 4, 4)
        for policy in MaskPolicy:
            result = apply_mask_policy(images, mask, policy)
            torch.testing.assert_close(result, images, msg=f"policy={policy}")

    def test_policy_accepts_a_string(self, images, half_valid):
        result = apply_mask_policy(images, half_valid, MaskPolicy("zero"))
        assert float(result[:, :, :2, :].max()) == 0.0


class TestShapeValidation:
    def test_rejects_wrong_rank(self, half_valid):
        with pytest.raises(ValueError, match=r"images must be \[T, C, H, W\]"):
            apply_mask_policy(torch.zeros(3, 4), half_valid, MaskPolicy.ZERO)

    def test_rejects_mismatched_time(self, images):
        with pytest.raises(ValueError, match="not broadcastable"):
            apply_mask_policy(images, torch.ones(5, 1, 4, 4), MaskPolicy.ZERO)

    def test_rejects_mismatched_spatial_size(self, images):
        with pytest.raises(ValueError, match="not broadcastable"):
            apply_mask_policy(images, torch.ones(2, 1, 8, 8), MaskPolicy.ZERO)


class TestMaskedFraction:
    def test_all_valid(self):
        assert masked_fraction(torch.ones(2, 1, 4, 4)) == pytest.approx(0.0)

    def test_all_invalid(self):
        assert masked_fraction(torch.zeros(2, 1, 4, 4)) == pytest.approx(1.0)

    def test_half_invalid(self, half_valid):
        assert masked_fraction(half_valid) == pytest.approx(0.5)

    def test_empty_mask(self):
        assert masked_fraction(torch.empty(0)) == 0.0


class TestValidityThreshold:
    def test_zero_threshold_accepts_everything(self):
        assert passes_validity_threshold(torch.zeros(2, 1, 4, 4), 0.0)

    def test_accepts_when_above_threshold(self, half_valid):
        assert passes_validity_threshold(half_valid, 0.4)

    def test_rejects_when_below_threshold(self, half_valid):
        assert not passes_validity_threshold(half_valid, 0.6)

    def test_boundary_is_inclusive(self, half_valid):
        assert passes_validity_threshold(half_valid, 0.5)

    def test_empty_mask_fails_a_positive_threshold(self):
        assert not passes_validity_threshold(torch.empty(0), 0.1)

    @pytest.mark.parametrize("threshold", [-0.1, 1.1])
    def test_rejects_out_of_range_threshold(self, threshold):
        with pytest.raises(ValueError, match=r"must be in \[0, 1\]"):
            passes_validity_threshold(torch.ones(1, 1, 2, 2), threshold)
