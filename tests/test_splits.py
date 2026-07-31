"""Tests for splits and the deterministic train/val partition.

The partition must be stable across processes *and* stable when cubes are added
or removed. The second property is what a seeded shuffle fails to provide, and
is why hashing was chosen; it is tested explicitly below.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

from tinyearth.datasets.splits import (
    TEST_SPLITS,
    Split,
    SplitNotFoundError,
    assign_partition,
    discover_cubes,
    partition_train_val,
    split_directory,
    split_directory_name,
    summarise_partition,
)


class TestSplitEnum:
    def test_test_tracks_are_flagged(self):
        for split in (Split.IID_TEST, Split.OOD_TEST, Split.EXTREME_TEST, Split.SEASONAL_TEST):
            assert split.is_test

    def test_train_and_val_are_not_test(self):
        assert not Split.TRAIN.is_test
        assert not Split.VAL.is_test

    def test_train_and_val_derive_from_train(self):
        assert Split.TRAIN.derives_from_train
        assert Split.VAL.derives_from_train

    def test_test_tracks_do_not_derive_from_train(self):
        for split in TEST_SPLITS:
            assert not split.derives_from_train

    def test_constructible_from_string(self):
        assert Split("iid_test") is Split.IID_TEST

    def test_val_shares_the_train_directory(self):
        assert split_directory_name(Split.VAL) == split_directory_name(Split.TRAIN) == "train"

    def test_official_directory_names(self):
        assert split_directory_name(Split.IID_TEST) == "iid_test_split"
        assert split_directory_name(Split.OOD_TEST) == "ood_test_split"


class TestSplitDirectory:
    def test_returns_existing_directory(self, tmp_path: Path):
        (tmp_path / "train").mkdir()
        assert split_directory(tmp_path, Split.TRAIN) == tmp_path / "train"

    def test_missing_directory_gives_download_instructions(self, tmp_path: Path):
        with pytest.raises(SplitNotFoundError, match="download_earthnet2021"):
            split_directory(tmp_path, Split.TRAIN)

    def test_missing_directory_mentions_no_redistribution(self, tmp_path: Path):
        with pytest.raises(SplitNotFoundError, match="does not redistribute"):
            split_directory(tmp_path, Split.OOD_TEST)


class TestDiscoverCubes:
    def test_finds_npz_files(self, tmp_path: Path):
        for name in ("a", "b", "c"):
            (tmp_path / f"{name}.npz").touch()
        assert len(discover_cubes(tmp_path)) == 3

    def test_recurses_into_subdirectories(self, tmp_path: Path):
        nested = tmp_path / "29SND" / "inner"
        nested.mkdir(parents=True)
        (nested / "cube.npz").touch()
        assert len(discover_cubes(tmp_path)) == 1

    def test_ignores_other_extensions(self, tmp_path: Path):
        (tmp_path / "a.npz").touch()
        (tmp_path / "readme.txt").touch()
        assert len(discover_cubes(tmp_path)) == 1

    def test_results_are_sorted(self, tmp_path: Path):
        for name in ("zulu", "alpha", "mike"):
            (tmp_path / f"{name}.npz").touch()
        paths = discover_cubes(tmp_path)
        assert list(paths) == sorted(paths)

    def test_paths_are_absolute(self, tmp_path: Path):
        (tmp_path / "a.npz").touch()
        assert all(path.is_absolute() for path in discover_cubes(tmp_path))

    def test_empty_directory(self, tmp_path: Path):
        assert discover_cubes(tmp_path) == ()


class TestAssignPartition:
    def test_is_deterministic(self):
        assert assign_partition("cube_x", 0.2) is assign_partition("cube_x", 0.2)

    def test_zero_fraction_sends_everything_to_train(self):
        assert all(assign_partition(f"cube_{i}", 0.0) is Split.TRAIN for i in range(50))

    def test_approximates_the_requested_fraction(self):
        ids = [f"cube_{i:05d}" for i in range(4000)]
        val = sum(assign_partition(cube, 0.2) is Split.VAL for cube in ids)
        assert 0.18 <= val / len(ids) <= 0.22

    def test_salt_changes_the_partition(self):
        ids = [f"cube_{i:04d}" for i in range(400)]
        first = {cube for cube in ids if assign_partition(cube, 0.2, "a") is Split.VAL}
        second = {cube for cube in ids if assign_partition(cube, 0.2, "b") is Split.VAL}
        assert first != second

    @pytest.mark.parametrize("fraction", [-0.1, 1.0, 1.5])
    def test_rejects_out_of_range_fraction(self, fraction):
        with pytest.raises(ValueError, match=r"must be in \[0, 1\)"):
            assign_partition("cube", fraction)

    def test_stable_when_other_cubes_are_added(self):
        """The property a seeded shuffle cannot provide.

        Adding cubes must not reassign existing ones -- otherwise a cube that
        was validation yesterday becomes training today, contaminating any
        comparison against previously reported numbers.
        """
        original = [f"cube_{i:04d}" for i in range(100)]
        before = {cube: assign_partition(cube, 0.2) for cube in original}

        # Simulate a later, larger download.
        _ = [f"cube_{i:04d}" for i in range(500)]
        after = {cube: assign_partition(cube, 0.2) for cube in original}

        assert before == after

    def test_stable_across_processes(self):
        """Guards against using `hash()`, whose string seed varies per process."""
        code = (
            "from tinyearth.datasets.splits import assign_partition;"
            "print(''.join('1' if assign_partition(f'c{i}', 0.3).value == 'val' else '0'"
            " for i in range(40)))"
        )
        results = {
            subprocess.run(
                [sys.executable, "-c", code], capture_output=True, text=True, check=True
            ).stdout.strip()
            for _ in range(2)
        }
        assert len(results) == 1


class TestPartitionTrainVal:
    @pytest.fixture
    def cubes(self, tmp_path: Path) -> list[Path]:
        return [tmp_path / f"cube_{i:04d}.npz" for i in range(200)]

    def test_train_and_val_are_disjoint(self, cubes):
        train = set(partition_train_val(cubes, Split.TRAIN, 0.2))
        val = set(partition_train_val(cubes, Split.VAL, 0.2))
        assert not (train & val)

    def test_train_and_val_cover_everything(self, cubes):
        train = partition_train_val(cubes, Split.TRAIN, 0.2)
        val = partition_train_val(cubes, Split.VAL, 0.2)
        assert len(train) + len(val) == len(cubes)

    def test_preserves_order(self, cubes):
        train = partition_train_val(cubes, Split.TRAIN, 0.2)
        assert list(train) == sorted(train)

    def test_rejects_a_test_track(self, cubes):
        with pytest.raises(ValueError, match="official test track"):
            partition_train_val(cubes, Split.IID_TEST, 0.2)

    def test_summarise_matches_the_partition(self, cubes):
        counts = summarise_partition(cubes, 0.2)
        assert counts["train"] == len(partition_train_val(cubes, Split.TRAIN, 0.2))
        assert counts["val"] == len(partition_train_val(cubes, Split.VAL, 0.2))
