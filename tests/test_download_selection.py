"""Tests for choosing which EarthNet2021 tarballs to download.

This is a small function guarding a large mistake. The archives are grouped by
Sentinel-2 tile, so a contiguous subset of the training split is one region of
the planet rather than a sample of the dataset -- and a study that trains and
validates on it has silently restricted itself to a single landscape without
anything failing or warning.
"""

from __future__ import annotations

import sys

import pytest

from tinyearth.utils.paths import project_root

sys.path.insert(0, str(project_root() / "scripts"))

from download_earthnet2021 import Tarball, select


def manifest(count: int) -> tuple[Tarball, ...]:
    """Build a stand-in manifest of ``count`` tarballs."""
    return tuple(
        Tarball(name=f"train_{index:03d}.tar.gz", url=f"https://example/{index}", sha256="0" * 64)
        for index in range(count)
    )


def names(chosen: tuple[Tarball, ...]) -> list[str]:
    """Return the selected archive names."""
    return [entry.name for entry in chosen]


def test_none_takes_everything():
    assert select(manifest(160), None, 1) == manifest(160)


def test_stride_one_is_a_prefix():
    assert names(select(manifest(160), 3, 1)) == [
        "train_000.tar.gz",
        "train_001.tar.gz",
        "train_002.tar.gz",
    ]


def test_stride_spreads_across_the_manifest():
    """The property the whole function exists for."""
    assert names(select(manifest(160), 4, 20)) == [
        "train_000.tar.gz",
        "train_020.tar.gz",
        "train_040.tar.gz",
        "train_060.tar.gz",
    ]


def test_count_caps_the_stride_walk():
    assert len(select(manifest(160), 5, 32)) == 5


def test_a_stride_that_overshoots_returns_what_exists():
    """Asking for more spread than the manifest holds must not raise."""
    chosen = select(manifest(160), 10, 32)
    assert names(chosen) == [f"train_{index:03d}.tar.gz" for index in range(0, 160, 32)]
    assert len(chosen) == 5


def test_requesting_more_than_available_is_not_an_error():
    assert len(select(manifest(4), 99, 1)) == 4


def test_selection_is_deterministic():
    """Reproducibility across machines depends on this."""
    assert select(manifest(160), 8, 20) == select(manifest(160), 8, 20)


@pytest.mark.parametrize("stride", [1, 2, 7, 20, 32])
def test_selection_preserves_manifest_order(stride: int):
    chosen = names(select(manifest(160), 6, stride))
    assert chosen == sorted(chosen)


def test_selection_never_repeats_an_archive():
    chosen = names(select(manifest(160), 8, 20))
    assert len(set(chosen)) == len(chosen)
