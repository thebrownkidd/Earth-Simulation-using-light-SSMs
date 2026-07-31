"""Tests for collation, dataloader construction and the config factory."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from tinyearth.config.resolution import ResolvedPaths
from tinyearth.config.schema import DataConfig, LoaderConfig, SyntheticConfig
from tinyearth.datasets.factory import build_datamodule, build_dataset, resolve_dataset_root
from tinyearth.datasets.loaders import LoaderSettings, build_dataloader, describe_batch
from tinyearth.datasets.splits import Split
from tinyearth.datasets.synthetic import SyntheticEarthNet2021, SyntheticSpec
from tinyearth.datasets.types import SampleMetadata, collate_samples

SPEC = SyntheticSpec(n_cubes=6, n_frames=12, size=8, seed=0)


@pytest.fixture
def dataset(tmp_path: Path) -> SyntheticEarthNet2021:
    return SyntheticEarthNet2021(
        tmp_path, spec=SPEC, history_length=3, horizon=2, val_fraction=0.25, context_frames=6
    )


@pytest.fixture
def paths(tmp_path: Path) -> ResolvedPaths:
    return ResolvedPaths(
        root=tmp_path,
        data=tmp_path / "data",
        outputs=tmp_path / "outputs",
        cache=tmp_path / "cache",
        run_dir=tmp_path / "outputs" / "g" / "n",
    )


def make_data_config(**kwargs: object) -> DataConfig:
    """Build a DataConfig with small synthetic defaults."""
    cfg = DataConfig(
        name="synthetic",
        root="synth",
        history_length=3,
        horizon=2,
        val_fraction=0.25,
        context_frames=6,
        loader=LoaderConfig(batch_size=2, num_workers=0),
        synthetic=SyntheticConfig(n_cubes=6, n_frames=12, size=8, seed=0),
    )
    for key, value in kwargs.items():
        setattr(cfg, key, value)
    return cfg


class TestCollate:
    def test_stacks_tensors_with_a_batch_axis(self, dataset):
        batch = collate_samples([dataset[0], dataset[1]])
        assert batch["images"].shape == (2, 3, 4, 8, 8)
        assert batch["target"].shape == (2, 2, 4, 8, 8)

    def test_metadata_becomes_a_list(self, dataset):
        batch = collate_samples([dataset[0], dataset[1]])
        assert isinstance(batch["metadata"], list)
        assert len(batch["metadata"]) == 2
        assert all(isinstance(item, SampleMetadata) for item in batch["metadata"])

    def test_masks_are_collated_when_present(self, dataset):
        batch = collate_samples([dataset[0], dataset[1]])
        assert batch["images_mask"].shape == (2, 3, 1, 8, 8)
        assert batch["target_mask"].shape == (2, 2, 1, 8, 8)

    def test_rejects_an_empty_sequence(self):
        with pytest.raises(ValueError, match="empty sequence"):
            collate_samples([])

    def test_rejects_inconsistent_mask_keys(self, tmp_path: Path):
        masked = SyntheticEarthNet2021(
            tmp_path / "a", spec=SPEC, history_length=3, horizon=2, val_fraction=0.25
        )
        unmasked = SyntheticEarthNet2021(
            tmp_path / "b",
            spec=SPEC,
            history_length=3,
            horizon=2,
            val_fraction=0.25,
            cloud_masking=False,
        )
        with pytest.raises(ValueError, match="Inconsistent mask keys"):
            collate_samples([masked[0], unmasked[0]])

    def test_single_sample_batch(self, dataset):
        batch = collate_samples([dataset[0]])
        assert batch["images"].shape[0] == 1


class TestLoaderSettings:
    def test_rejects_zero_batch_size(self):
        with pytest.raises(ValueError, match="batch_size must be"):
            LoaderSettings(batch_size=0)

    def test_rejects_negative_workers(self):
        with pytest.raises(ValueError, match="num_workers must be"):
            LoaderSettings(num_workers=-1)

    def test_persistent_workers_requires_workers(self):
        with pytest.raises(ValueError, match="persistent_workers requires"):
            LoaderSettings(num_workers=0, persistent_workers=True)

    def test_prefetch_factor_requires_workers(self):
        with pytest.raises(ValueError, match="prefetch_factor requires"):
            LoaderSettings(num_workers=0, prefetch_factor=2)

    def test_accepts_worker_options_with_workers(self):
        LoaderSettings(num_workers=2, persistent_workers=True, prefetch_factor=2)


class TestBuildDataloader:
    def test_yields_batches_of_the_requested_size(self, dataset):
        loader = build_dataloader(dataset, LoaderSettings(batch_size=4, shuffle=False))
        batch = next(iter(loader))
        assert batch["images"].shape[0] == 4

    def test_same_seed_gives_the_same_shuffled_order(self, dataset):
        first = next(iter(build_dataloader(dataset, LoaderSettings(batch_size=4), seed=7)))
        second = next(iter(build_dataloader(dataset, LoaderSettings(batch_size=4), seed=7)))
        torch.testing.assert_close(first["images"], second["images"])

    def test_different_seeds_give_different_order(self, dataset):
        settings = LoaderSettings(batch_size=4, shuffle=True)
        first = next(iter(build_dataloader(dataset, settings, seed=1)))
        second = next(iter(build_dataloader(dataset, settings, seed=2)))
        ids_first = [item.cube_id for item in first["metadata"]]
        ids_second = [item.cube_id for item in second["metadata"]]
        starts_first = [item.start_index for item in first["metadata"]]
        starts_second = [item.start_index for item in second["metadata"]]
        assert (ids_first, starts_first) != (ids_second, starts_second)

    def test_shuffle_false_preserves_dataset_order(self, dataset):
        loader = build_dataloader(dataset, LoaderSettings(batch_size=2, shuffle=False))
        batch = next(iter(loader))
        assert [item.start_index for item in batch["metadata"]] == [0, 1]

    def test_drop_last_discards_the_partial_batch(self, dataset):
        keep = build_dataloader(dataset, LoaderSettings(batch_size=7, drop_last=False))
        drop = build_dataloader(dataset, LoaderSettings(batch_size=7, drop_last=True))
        assert len(drop) <= len(keep)

    def test_pin_memory_disabled_on_cpu(self, dataset):
        loader = build_dataloader(
            dataset,
            LoaderSettings(batch_size=2, pin_memory=True),
            device=torch.device("cpu"),
        )
        assert loader.pin_memory is False

    def test_iterating_covers_every_sample(self, dataset):
        loader = build_dataloader(dataset, LoaderSettings(batch_size=4, shuffle=False))
        total = sum(batch["images"].shape[0] for batch in loader)
        assert total == len(dataset)

    def test_describe_batch_reports_shapes(self, dataset):
        batch = next(iter(build_dataloader(dataset, LoaderSettings(batch_size=2))))
        text = describe_batch(batch)
        assert "images=" in text
        assert "target_valid=" in text


class TestFactory:
    def test_synthetic_root_resolves_under_cache(self, paths):
        cfg = make_data_config()
        assert resolve_dataset_root(cfg, paths) == paths.cache / "synth"

    def test_real_root_resolves_under_data(self, paths):
        cfg = make_data_config(name="earthnet2021", root="earthnet2021")
        assert resolve_dataset_root(cfg, paths) == paths.data / "earthnet2021"

    def test_absolute_root_is_used_as_given(self, paths, tmp_path: Path):
        external = tmp_path / "elsewhere"
        cfg = make_data_config(root=str(external))
        assert resolve_dataset_root(cfg, paths) == external

    def test_builds_a_synthetic_dataset(self, paths):
        dataset = build_dataset(make_data_config(), paths)
        assert isinstance(dataset, SyntheticEarthNet2021)
        assert len(dataset) > 0

    def test_unknown_dataset_name(self, paths):
        with pytest.raises(ValueError, match="Unknown dataset"):
            build_dataset(make_data_config(name="nope"), paths)

    def test_split_override_wins_over_config(self, paths):
        bundle = build_datamodule(make_data_config(), paths, split=Split.VAL)
        assert bundle.split is Split.VAL

    def test_shuffle_forced_off_for_validation(self, paths):
        cfg = make_data_config(loader=LoaderConfig(batch_size=2, shuffle=True))
        bundle = build_datamodule(cfg, paths, split=Split.VAL)
        assert bundle.loader.sampler is not None
        batch = next(iter(bundle.loader))
        assert [item.start_index for item in batch["metadata"]] == [0, 1]

    def test_shuffle_preserved_for_training(self, paths):
        cfg = make_data_config(loader=LoaderConfig(batch_size=2, shuffle=True))
        bundle = build_datamodule(cfg, paths, split=Split.TRAIN)
        assert len(bundle) > 0

    def test_bundle_length_matches_dataset(self, paths):
        bundle = build_datamodule(make_data_config(), paths)
        assert len(bundle) == len(bundle.dataset)

    def test_channels_config_is_applied(self, paths):
        dataset = build_dataset(make_data_config(channels=[0, 2]), paths)
        assert dataset[0]["images"].shape[1] == 2

    def test_mask_policy_config_is_applied(self, paths):
        dataset = build_dataset(make_data_config(mask_policy="mean"), paths)
        assert dataset[0]["images"].shape[0] == 3
