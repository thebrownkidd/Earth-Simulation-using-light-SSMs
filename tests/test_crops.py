"""Tests for spatial cropping.

The properties that matter are alignment and determinism. A crop that moved
between the imagery and its mask, or between history and target, would corrupt
training silently -- the loss would still fall, just against the wrong pixels.
"""

from __future__ import annotations

import pytest
import torch

from tinyearth.datasets.crops import CropMode, SpatialCrop


def test_rejects_a_non_positive_size():
    with pytest.raises(ValueError, match="crop size must be >= 1"):
        SpatialCrop(size=0)


def test_rejects_a_crop_larger_than_the_frame():
    crop = SpatialCrop(size=64, mode=CropMode.CENTER)
    with pytest.raises(ValueError, match="does not fit"):
        crop.origin(32, 32)


def test_center_crop_is_centred():
    crop = SpatialCrop(size=64, mode=CropMode.CENTER)
    assert crop.origin(128, 128) == (32, 32)


def test_center_crop_is_stable_across_calls():
    """Validation must not move between epochs, or the metric becomes noisy."""
    crop = SpatialCrop(size=32, mode=CropMode.CENTER)
    assert {crop.origin(128, 128) for _ in range(50)} == {(48, 48)}


def test_random_crop_explores_the_frame():
    crop = SpatialCrop(size=32, mode=CropMode.RANDOM)
    torch.manual_seed(0)
    origins = {crop.origin(128, 128) for _ in range(200)}
    assert len(origins) > 50, "random origins should not collapse to a few positions"


def test_random_crop_stays_in_bounds():
    crop = SpatialCrop(size=32, mode=CropMode.RANDOM)
    torch.manual_seed(0)
    for _ in range(200):
        top, left = crop.origin(128, 100)
        assert 0 <= top <= 128 - 32
        assert 0 <= left <= 100 - 32


def test_random_crop_is_reproducible_under_a_seed():
    """Determinism is a project-wide guarantee; cropping must not break it."""
    crop = SpatialCrop(size=32, mode=CropMode.RANDOM)
    torch.manual_seed(7)
    first = [crop.origin(128, 128) for _ in range(20)]
    torch.manual_seed(7)
    assert [crop.origin(128, 128) for _ in range(20)] == first


def test_exact_fit_yields_the_only_origin():
    crop = SpatialCrop(size=128, mode=CropMode.RANDOM)
    assert crop.origin(128, 128) == (0, 0)


def test_apply_returns_the_requested_size():
    crop = SpatialCrop(size=16)
    images = torch.rand(4, 3, 128, 128)
    assert crop.apply(images, 10, 20).shape == (4, 3, 16, 16)


def test_apply_selects_the_requested_region():
    crop = SpatialCrop(size=8)
    images = torch.arange(128 * 128, dtype=torch.float32).reshape(1, 1, 128, 128)
    assert torch.equal(crop.apply(images, 5, 9), images[..., 5:13, 9:17])


def test_apply_is_rank_agnostic():
    """Imagery is [T, C, H, W] and masks are [T, 1, H, W]; both must work."""
    crop = SpatialCrop(size=16)
    assert crop.apply(torch.rand(4, 1, 64, 64), 0, 0).shape == (4, 1, 16, 16)
    assert crop.apply(torch.rand(2, 4, 3, 64, 64), 0, 0).shape == (2, 4, 3, 16, 16)


def test_for_evaluation_is_centred():
    assert SpatialCrop.for_evaluation(32).mode is CropMode.CENTER
