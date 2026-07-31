"""Tests for normalisation and channel statistics."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from tinyearth.datasets.normalization import (
    ChannelStandardizer,
    ChannelStatistics,
    IdentityNormalizer,
    Normalizer,
    build_normalizer,
    compute_channel_statistics,
)


class TestIdentity:
    def test_is_a_no_op(self):
        images = torch.rand(2, 4, 8, 8)
        torch.testing.assert_close(IdentityNormalizer()(images), images)

    def test_round_trips(self):
        images = torch.rand(2, 4, 8, 8)
        normalizer = IdentityNormalizer()
        torch.testing.assert_close(normalizer.invert(normalizer(images)), images)

    def test_satisfies_the_protocol(self):
        assert isinstance(IdentityNormalizer(), Normalizer)


class TestChannelStatistics:
    def test_rejects_mismatched_lengths(self):
        with pytest.raises(ValueError, match="channels but std has"):
            ChannelStatistics(mean=(0.1, 0.2), std=(1.0,))

    def test_rejects_empty(self):
        with pytest.raises(ValueError, match="at least one channel"):
            ChannelStatistics(mean=(), std=())

    def test_rejects_degenerate_std(self):
        with pytest.raises(ValueError, match="too small to standardise"):
            ChannelStatistics(mean=(0.1,), std=(0.0,))

    def test_round_trips_through_json(self, tmp_path: Path):
        original = ChannelStatistics(mean=(0.1, 0.2), std=(0.3, 0.4), n_pixels=99)
        path = original.save(tmp_path / "nested" / "stats.json")
        assert ChannelStatistics.load(path) == original

    def test_load_missing_file_suggests_the_script(self, tmp_path: Path):
        with pytest.raises(FileNotFoundError, match="compute_dataset_statistics"):
            ChannelStatistics.load(tmp_path / "absent.json")


class TestChannelStandardizer:
    @pytest.fixture
    def statistics(self) -> ChannelStatistics:
        return ChannelStatistics(mean=(0.1, 0.2, 0.3, 0.4), std=(0.5, 0.5, 0.5, 0.5))

    def test_produces_zero_mean_unit_variance(self):
        images = torch.randn(64, 4, 16, 16) * 3.0 + 7.0
        per_channel = images.movedim(1, 0).reshape(4, -1)
        statistics = ChannelStatistics(
            mean=tuple(per_channel.mean(dim=1).tolist()),
            std=tuple(per_channel.std(dim=1).tolist()),
        )

        result = ChannelStandardizer(statistics)(images)
        result_flat = result.movedim(1, 0).reshape(4, -1)
        torch.testing.assert_close(result_flat.mean(dim=1), torch.zeros(4), atol=1e-4, rtol=1e-3)
        torch.testing.assert_close(result_flat.std(dim=1), torch.ones(4), atol=1e-3, rtol=1e-3)

    def test_round_trips(self, statistics):
        images = torch.rand(3, 4, 8, 8)
        normalizer = ChannelStandardizer(statistics)
        torch.testing.assert_close(normalizer.invert(normalizer(images)), images, atol=1e-6, rtol=0)

    def test_applies_per_channel_not_globally(self, statistics):
        images = torch.zeros(1, 4, 2, 2)
        result = ChannelStandardizer(statistics)(images)
        expected = torch.tensor([-0.2, -0.4, -0.6, -0.8])
        torch.testing.assert_close(result[0, :, 0, 0], expected)

    def test_works_on_a_batched_five_dim_tensor(self, statistics):
        images = torch.rand(2, 3, 4, 8, 8)
        assert ChannelStandardizer(statistics)(images).shape == images.shape

    def test_rejects_channel_count_mismatch(self, statistics):
        with pytest.raises(ValueError, match="cover 4 channels but imagery has 2"):
            ChannelStandardizer(statistics)(torch.rand(1, 2, 8, 8))

    def test_satisfies_the_protocol(self, statistics):
        assert isinstance(ChannelStandardizer(statistics), Normalizer)


class TestBuildNormalizer:
    def test_identity_by_name(self):
        assert isinstance(build_normalizer("identity"), IdentityNormalizer)

    def test_standardize_by_name(self):
        statistics = ChannelStatistics(mean=(0.1,), std=(0.2,))
        assert isinstance(
            build_normalizer("standardize", statistics=statistics), ChannelStandardizer
        )

    def test_accepts_british_spelling(self):
        statistics = ChannelStatistics(mean=(0.1,), std=(0.2,))
        assert isinstance(
            build_normalizer("standardise", statistics=statistics), ChannelStandardizer
        )

    def test_standardize_without_statistics_explains_how_to_get_them(self):
        with pytest.raises(ValueError, match="compute_dataset_statistics"):
            build_normalizer("standardize")

    def test_unknown_kind(self):
        with pytest.raises(ValueError, match="Unknown normalization"):
            build_normalizer("whitening")


class TestComputeChannelStatistics:
    def test_recovers_known_moments(self):
        torch.manual_seed(0)
        images = torch.randn(200, 3, 8, 8) * 2.0 + 5.0
        statistics = compute_channel_statistics([(images, None)], n_channels=3)

        assert all(value == pytest.approx(5.0, abs=0.1) for value in statistics.mean)
        assert all(value == pytest.approx(2.0, abs=0.1) for value in statistics.std)

    def test_accumulates_across_batches(self):
        torch.manual_seed(0)
        batches = [(torch.randn(50, 2, 8, 8) + 3.0, None) for _ in range(4)]
        statistics = compute_channel_statistics(batches, n_channels=2)
        assert all(value == pytest.approx(3.0, abs=0.15) for value in statistics.mean)

    def test_mask_excludes_invalid_pixels(self):
        """Cloudy pixels are bright; counting them would inflate every mean."""
        images = torch.full((4, 2, 8, 8), 0.2)
        images[:, :, :4, :] = 0.95  # "cloud"
        mask = torch.ones(4, 1, 8, 8)
        mask[:, :, :4, :] = 0.0

        statistics = compute_channel_statistics([(images, mask)], n_channels=2)
        assert all(value == pytest.approx(0.2, abs=1e-5) for value in statistics.mean)

    def test_records_the_pixel_count(self):
        images = torch.rand(3, 2, 4, 4)
        statistics = compute_channel_statistics([(images, None)], n_channels=2)
        assert statistics.n_pixels == 3 * 4 * 4

    def test_all_masked_raises_with_a_polarity_hint(self):
        images = torch.rand(2, 2, 4, 4)
        with pytest.raises(ValueError, match="cloud-mask convention"):
            compute_channel_statistics([(images, torch.zeros(2, 1, 4, 4))], n_channels=2)

    def test_result_is_usable_by_the_standardizer(self):
        torch.manual_seed(0)
        images = torch.randn(100, 4, 8, 8) * 1.5 + 2.0
        statistics = compute_channel_statistics([(images, None)], n_channels=4)

        result = ChannelStandardizer(statistics)(images)
        flat = result.movedim(1, 0).reshape(4, -1)
        torch.testing.assert_close(flat.mean(dim=1), torch.zeros(4), atol=1e-4, rtol=1e-3)
