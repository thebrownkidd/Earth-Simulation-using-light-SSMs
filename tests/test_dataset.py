"""Tests for the EarthNet2021 dataset and the synthetic generator.

These run against synthetic cubes written in the real on-disk format, so the
production reader is what gets exercised -- not a parallel implementation.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from tinyearth.datasets.earthnet2021 import EarthNet2021Dataset
from tinyearth.datasets.masking import MaskPolicy
from tinyearth.datasets.normalization import ChannelStandardizer, ChannelStatistics
from tinyearth.datasets.splits import Split
from tinyearth.datasets.synthetic import (
    SyntheticEarthNet2021,
    SyntheticSpec,
    write_synthetic_dataset,
)
from tinyearth.datasets.types import SampleMetadata

SPEC = SyntheticSpec(n_cubes=6, n_frames=12, size=8, cloud_fraction=0.2, nan_fraction=0.02, seed=0)


@pytest.fixture
def dataset_root(tmp_path: Path) -> Path:
    write_synthetic_dataset(tmp_path, SPEC)
    return tmp_path


def make_dataset(root: Path, **kwargs: object) -> EarthNet2021Dataset:
    """Build a dataset over synthetic cubes with sensible test defaults."""
    defaults: dict[str, object] = {
        "history_length": 3,
        "horizon": 2,
        "val_fraction": 0.25,
        "expected_frames": SPEC.n_frames,
        "context_frames": 6,
    }
    defaults.update(kwargs)
    return EarthNet2021Dataset(root, **defaults)  # type: ignore[arg-type]


class TestSyntheticGeneration:
    def test_writes_the_expected_number_of_cubes(self, dataset_root: Path):
        assert len(list((dataset_root / "train").glob("*.npz"))) == SPEC.n_cubes

    def test_is_idempotent(self, tmp_path: Path):
        write_synthetic_dataset(tmp_path, SPEC)
        first = (tmp_path / "train" / "train_cube_0000.npz").stat().st_mtime_ns
        write_synthetic_dataset(tmp_path, SPEC)
        assert (tmp_path / "train" / "train_cube_0000.npz").stat().st_mtime_ns == first

    def test_regenerates_when_the_spec_changes(self, tmp_path: Path):
        write_synthetic_dataset(tmp_path, SPEC)
        write_synthetic_dataset(tmp_path, SyntheticSpec(n_cubes=3, n_frames=12, size=8))
        assert len(list((tmp_path / "train").glob("*.npz"))) == 3

    def test_content_is_reproducible_for_a_fixed_seed(self, tmp_path: Path):
        write_synthetic_dataset(tmp_path / "a", SPEC)
        write_synthetic_dataset(tmp_path / "b", SPEC)
        first = make_dataset(tmp_path / "a")[0]
        second = make_dataset(tmp_path / "b")[0]
        torch.testing.assert_close(first["images"], second["images"])

    @pytest.mark.parametrize(
        ("field", "value"),
        [
            ("n_cubes", 0),
            ("n_frames", 1),
            ("size", 0),
            ("cloud_fraction", 1.5),
            ("nan_fraction", -1),
        ],
    )
    def test_spec_validation(self, field, value):
        with pytest.raises(ValueError, match=field):
            SyntheticSpec(**{field: value})


class TestSampleContract:
    def test_returns_the_documented_keys(self, dataset_root: Path):
        sample = make_dataset(dataset_root)[0]
        assert set(sample) == {"images", "target", "metadata", "images_mask", "target_mask"}

    def test_images_shape_is_time_channel_height_width(self, dataset_root: Path):
        sample = make_dataset(dataset_root, history_length=3, horizon=2)[0]
        assert sample["images"].shape == (3, 4, SPEC.size, SPEC.size)

    def test_target_keeps_its_time_axis_at_horizon_one(self, dataset_root: Path):
        """Rank must not depend on a config value."""
        sample = make_dataset(dataset_root, history_length=3, horizon=1)[0]
        assert sample["target"].shape == (1, 4, SPEC.size, SPEC.size)
        assert sample["target"].ndim == sample["images"].ndim

    @pytest.mark.parametrize("horizon", [1, 2, 4])
    def test_target_length_matches_horizon(self, dataset_root: Path, horizon):
        sample = make_dataset(dataset_root, history_length=3, horizon=horizon)[0]
        assert sample["target"].shape[0] == horizon

    @pytest.mark.parametrize("history", [2, 4, 6])
    def test_images_length_matches_history(self, dataset_root: Path, history):
        sample = make_dataset(dataset_root, history_length=history, horizon=1)[0]
        assert sample["images"].shape[0] == history

    def test_masks_have_a_singleton_channel(self, dataset_root: Path):
        sample = make_dataset(dataset_root, history_length=3, horizon=2)[0]
        assert sample["images_mask"].shape == (3, 1, SPEC.size, SPEC.size)
        assert sample["target_mask"].shape == (2, 1, SPEC.size, SPEC.size)

    def test_masks_absent_when_masking_disabled(self, dataset_root: Path):
        sample = make_dataset(dataset_root, cloud_masking=False)[0]
        assert "images_mask" not in sample
        assert "target_mask" not in sample

    def test_metadata_is_populated(self, dataset_root: Path):
        sample = make_dataset(dataset_root, history_length=3, horizon=2)[0]
        metadata = sample["metadata"]
        assert isinstance(metadata, SampleMetadata)
        assert metadata.split == "train"
        assert metadata.history_length == 3
        assert metadata.horizon == 2
        assert Path(metadata.source).is_file()

    def test_values_stay_in_unit_range(self, dataset_root: Path):
        sample = make_dataset(dataset_root)[0]
        assert float(sample["images"].min()) >= 0.0
        assert float(sample["images"].max()) <= 1.0

    def test_no_nans_survive(self, dataset_root: Path):
        dataset = make_dataset(dataset_root)
        for index in range(min(len(dataset), 10)):
            sample = dataset[index]
            assert not torch.isnan(sample["images"]).any()
            assert not torch.isnan(sample["target"]).any()


class TestIndexing:
    def test_length_is_cubes_times_windows(self, dataset_root: Path):
        dataset = make_dataset(dataset_root, history_length=3, horizon=2, val_fraction=0.0)
        windows_per_cube = SPEC.n_frames - 5 + 1
        assert len(dataset) == SPEC.n_cubes * windows_per_cube

    def test_stride_reduces_the_index(self, dataset_root: Path):
        dense = make_dataset(dataset_root, stride=1, val_fraction=0.0)
        sparse = make_dataset(dataset_root, stride=3, val_fraction=0.0)
        assert len(sparse) < len(dense)

    def test_negative_indexing_works(self, dataset_root: Path):
        dataset = make_dataset(dataset_root)
        assert dataset[-1]["images"].shape == dataset[len(dataset) - 1]["images"].shape

    def test_out_of_range_raises(self, dataset_root: Path):
        dataset = make_dataset(dataset_root)
        with pytest.raises(IndexError, match="out of range"):
            dataset[len(dataset)]

    def test_successive_windows_advance_in_time(self, dataset_root: Path):
        dataset = make_dataset(dataset_root, val_fraction=0.0)
        starts = [dataset.index[i].window.start for i in range(4)]
        assert starts == [0, 1, 2, 3]

    def test_window_too_long_for_the_cube_is_rejected(self, dataset_root: Path):
        with pytest.raises(ValueError, match="does not fit in a cube"):
            make_dataset(dataset_root, history_length=20, horizon=20)

    def test_max_cubes_caps_the_index(self, dataset_root: Path):
        dataset = make_dataset(dataset_root, max_cubes=2, val_fraction=0.0)
        assert len(dataset.cubes) == 2


class TestTemporalCorrectness:
    def test_target_frames_immediately_follow_history_frames(self, dataset_root: Path):
        """History and target must be contiguous and non-overlapping in the cube."""
        from tinyearth.datasets.minicube import read_minicube

        dataset = make_dataset(
            dataset_root, history_length=3, horizon=2, cloud_masking=False, val_fraction=0.0
        )
        entry = dataset.index[4]
        sample = dataset[4]
        cube = read_minicube(entry.path, apply_mask=False)

        start = entry.window.start
        torch.testing.assert_close(sample["images"], cube.images[start : start + 3])
        torch.testing.assert_close(sample["target"], cube.images[start + 3 : start + 5])

    def test_history_does_not_contain_target_frames(self, dataset_root: Path):
        dataset = make_dataset(dataset_root, history_length=3, horizon=2, val_fraction=0.0)
        for index in range(min(len(dataset), 8)):
            window = dataset.index[index].window
            assert window.history_slice.stop == window.target_slice.start


class TestSplits:
    def test_train_and_val_use_disjoint_cubes(self, dataset_root: Path):
        train = make_dataset(dataset_root, split=Split.TRAIN, val_fraction=0.25)
        val = make_dataset(dataset_root, split=Split.VAL, val_fraction=0.25)
        assert not (set(train.cubes) & set(val.cubes))

    def test_split_partition_covers_all_cubes(self, dataset_root: Path):
        train = make_dataset(dataset_root, split=Split.TRAIN, val_fraction=0.25)
        val = make_dataset(dataset_root, split=Split.VAL, val_fraction=0.25)
        assert len(train.cubes) + len(val.cubes) == SPEC.n_cubes

    def test_empty_partition_gives_an_actionable_error(self, dataset_root: Path):
        with pytest.raises(ValueError, match="Raise val_fraction"):
            make_dataset(dataset_root, split=Split.VAL, val_fraction=0.001)

    def test_missing_split_directory(self, tmp_path: Path):
        from tinyearth.datasets.splits import SplitNotFoundError

        with pytest.raises(SplitNotFoundError):
            make_dataset(tmp_path, split=Split.OOD_TEST)

    def test_test_split_uses_a_single_anchored_window_per_cube(self, tmp_path: Path):
        write_synthetic_dataset(tmp_path, SPEC, splits=(Split.TRAIN, Split.IID_TEST))
        dataset = make_dataset(tmp_path, split=Split.IID_TEST, history_length=3, horizon=2)
        assert len(dataset) == len(dataset.cubes) == SPEC.n_cubes

    def test_anchored_window_ends_at_the_context_boundary(self, tmp_path: Path):
        write_synthetic_dataset(tmp_path, SPEC, splits=(Split.TRAIN, Split.IID_TEST))
        dataset = make_dataset(tmp_path, split=Split.IID_TEST, history_length=3, horizon=2)
        window = dataset.index[0].window
        assert window.history_slice.stop == 6  # context_frames


class TestMaskingIntegration:
    def test_zero_policy_blanks_cloudy_pixels(self, dataset_root: Path):
        dataset = make_dataset(dataset_root, mask_policy=MaskPolicy.ZERO)
        sample = dataset[0]
        invalid = sample["images_mask"] == 0
        if invalid.any():
            blanked = sample["images"] * invalid
            assert float(blanked.abs().max()) == 0.0

    def test_valid_fraction_is_recorded_in_metadata(self, dataset_root: Path):
        sample = make_dataset(dataset_root)[0]
        expected = float(sample["target_mask"].mean())
        assert sample["metadata"].valid_fraction == pytest.approx(expected)

    def test_valid_fraction_is_one_when_masking_disabled(self, dataset_root: Path):
        sample = make_dataset(dataset_root, cloud_masking=False)[0]
        assert sample["metadata"].valid_fraction == 1.0

    def test_strict_threshold_rejects_everything(self, dataset_root: Path):
        dataset = make_dataset(dataset_root, min_valid_fraction=1.0, nan_is_invalid=True)
        with pytest.raises(IndexError, match="min_valid_fraction"):
            dataset[0]

    def test_threshold_requires_masking(self, dataset_root: Path):
        with pytest.raises(ValueError, match="requires cloud_masking=True"):
            make_dataset(dataset_root, cloud_masking=False, min_valid_fraction=0.5)

    def test_lenient_threshold_keeps_samples(self, dataset_root: Path):
        dataset = make_dataset(dataset_root, min_valid_fraction=0.1)
        assert dataset[0]["images"].shape[0] == 3


class TestChannelSelection:
    def test_selects_a_channel_subset(self, dataset_root: Path):
        sample = make_dataset(dataset_root, channels=(0, 3))[0]
        assert sample["images"].shape[1] == 2

    def test_n_channels_property_reflects_selection(self, dataset_root: Path):
        assert make_dataset(dataset_root, channels=(0, 1, 2)).n_channels == 3
        assert make_dataset(dataset_root).n_channels == 4

    def test_rejects_empty_channel_tuple(self, dataset_root: Path):
        with pytest.raises(ValueError, match="channels must be non-empty"):
            make_dataset(dataset_root, channels=())

    def test_selected_channels_match_the_full_read(self, dataset_root: Path):
        full = make_dataset(dataset_root, cloud_masking=False, val_fraction=0.0)[0]
        subset = make_dataset(dataset_root, channels=(1, 3), cloud_masking=False, val_fraction=0.0)[
            0
        ]
        torch.testing.assert_close(subset["images"], full["images"][:, [1, 3]])


class TestNormalizationIntegration:
    def test_normalizer_is_applied_to_both_images_and_target(self, dataset_root: Path):
        statistics = ChannelStatistics(mean=(0.5,) * 4, std=(0.25,) * 4)
        plain = make_dataset(dataset_root, val_fraction=0.0)[0]
        scaled = make_dataset(
            dataset_root, normalizer=ChannelStandardizer(statistics), val_fraction=0.0
        )[0]

        torch.testing.assert_close(scaled["images"], (plain["images"] - 0.5) / 0.25)
        torch.testing.assert_close(scaled["target"], (plain["target"] - 0.5) / 0.25)

    def test_masks_are_not_normalized(self, dataset_root: Path):
        statistics = ChannelStatistics(mean=(0.5,) * 4, std=(0.25,) * 4)
        sample = make_dataset(
            dataset_root, normalizer=ChannelStandardizer(statistics), val_fraction=0.0
        )[0]
        assert set(torch.unique(sample["images_mask"]).tolist()) <= {0.0, 1.0}


class TestCaching:
    def test_repeated_reads_of_one_cube_hit_the_cache(self, dataset_root: Path):
        dataset = make_dataset(dataset_root, cache_size=2, val_fraction=0.0)
        for index in range(5):
            dataset[index]
        stats = dataset.cache_statistics()
        assert stats["hits"] > 0

    def test_cache_can_be_disabled(self, dataset_root: Path):
        dataset = make_dataset(dataset_root, cache_size=0, val_fraction=0.0)
        for index in range(5):
            dataset[index]
        assert dataset.cache_statistics()["size"] == 0

    def test_cache_respects_its_capacity(self, dataset_root: Path):
        dataset = make_dataset(dataset_root, cache_size=2, val_fraction=0.0)
        for index in range(len(dataset)):
            dataset[index]
        assert dataset.cache_statistics()["size"] <= 2

    def test_cached_and_uncached_reads_agree(self, dataset_root: Path):
        cached = make_dataset(dataset_root, cache_size=4, val_fraction=0.0)
        uncached = make_dataset(dataset_root, cache_size=0, val_fraction=0.0)
        for index in (0, 3, 7):
            torch.testing.assert_close(cached[index]["images"], uncached[index]["images"])


class TestRobustness:
    def test_corrupt_cube_is_skipped_rather_than_fatal(self, dataset_root: Path):
        """One bad file in a 32k-cube download must not kill an overnight run."""
        (dataset_root / "train" / "train_cube_0000.npz").write_bytes(b"not an npz")
        dataset = make_dataset(dataset_root, val_fraction=0.0)
        assert dataset[0]["images"].shape[0] == 3

    def test_repr_is_informative(self, dataset_root: Path):
        text = repr(make_dataset(dataset_root))
        assert "EarthNet2021Dataset" in text
        assert "split='train'" in text


class TestSyntheticDataset:
    def test_generates_on_first_use(self, tmp_path: Path):
        dataset = SyntheticEarthNet2021(tmp_path, spec=SPEC, history_length=3, horizon=2)
        assert len(dataset) > 0

    def test_infers_expected_frames_from_the_spec(self, tmp_path: Path):
        dataset = SyntheticEarthNet2021(tmp_path, spec=SPEC, history_length=3, horizon=2)
        assert dataset.expected_frames == SPEC.n_frames

    def test_describe_warns_that_results_are_meaningless(self, tmp_path: Path):
        dataset = SyntheticEarthNet2021(tmp_path, spec=SPEC, history_length=3, horizon=2)
        assert "not meaningful" in dataset.describe()

    def test_creates_test_track_directories_when_asked(self, tmp_path: Path):
        SyntheticEarthNet2021(
            tmp_path, Split.IID_TEST, spec=SPEC, history_length=3, horizon=2, context_frames=6
        )
        assert (tmp_path / "iid_test_split").is_dir()

    def test_is_a_real_earthnet_dataset(self, tmp_path: Path):
        dataset = SyntheticEarthNet2021(tmp_path, spec=SPEC, history_length=3, horizon=2)
        assert isinstance(dataset, EarthNet2021Dataset)


class TestCropping:
    """Cropping must shrink every tensor of a sample by the same window."""

    def test_crop_sets_the_spatial_size(self, dataset_root: Path):
        sample = make_dataset(dataset_root, crop_size=4)[0]
        assert sample["images"].shape[-2:] == (4, 4)
        assert sample["target"].shape[-2:] == (4, 4)

    def test_crop_applies_to_masks_too(self, dataset_root: Path):
        sample = make_dataset(dataset_root, crop_size=4, cloud_masking=True)[0]
        assert sample["images_mask"].shape[-2:] == (4, 4)
        assert sample["target_mask"].shape[-2:] == (4, 4)

    def test_no_crop_keeps_the_native_size(self, dataset_root: Path):
        assert make_dataset(dataset_root)[0]["images"].shape[-2:] == (SPEC.size, SPEC.size)

    def test_validation_crops_deterministically(self, dataset_root: Path):
        """A moving validation crop would make the metric noisy for no reason."""
        dataset = make_dataset(dataset_root, split=Split.VAL, crop_size=4)
        first = dataset[0]["images"]
        assert torch.equal(first, dataset[0]["images"])

    def test_training_crops_are_drawn_at_random(self, dataset_root: Path):
        dataset = make_dataset(dataset_root, split=Split.TRAIN, crop_size=4)
        torch.manual_seed(0)
        draws = [dataset[0]["images"].clone() for _ in range(20)]
        assert any(not torch.equal(draws[0], other) for other in draws[1:])

    def test_history_and_target_share_one_origin(self, dataset_root: Path):
        """The model must be asked to forecast the patch it was shown.

        Checked against the source cube directly rather than by comparing the
        two crops to each other: a shared but *wrong* origin would pass the
        latter, and would train every model on misaligned ground.
        """
        dataset = make_dataset(dataset_root, split=Split.VAL, crop_size=4, cloud_masking=False)
        entry = dataset.index[0]
        cube = dataset._load_cube(entry.path)
        assert dataset.crop is not None
        top, left = dataset.crop.origin(*cube.spatial_size)

        sample = dataset[0]
        window = entry.window
        expected_history = cube.images[window.history_slice][..., top : top + 4, left : left + 4]
        expected_target = cube.images[window.target_slice][..., top : top + 4, left : left + 4]
        assert torch.equal(sample["images"], expected_history)
        assert torch.equal(sample["target"], expected_target)

    def test_a_crop_larger_than_the_cube_fails_loudly(self, dataset_root: Path):
        dataset = make_dataset(dataset_root, crop_size=SPEC.size * 2)
        with pytest.raises(IndexError, match="No usable sample"):
            _ = dataset[0]
