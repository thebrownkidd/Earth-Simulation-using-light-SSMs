"""Tests for minicube reading.

The mask-polarity test is the most important one in the suite. If ``cldmsk``
were interpreted backwards the model would train exclusively on cloud, and
nothing would fail -- losses would still decrease.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from tinyearth.datasets.minicube import (
    BAND_NAMES,
    N_IMAGE_CHANNELS,
    MinicubeFormatError,
    read_minicube,
)


def write_cube(
    path: Path,
    *,
    frames: int = 6,
    size: int = 4,
    channels: int = 7,
    fill: float = 0.5,
    cloudy: float = 0.0,
    nan_at: tuple[int, int, int, int] | None = None,
) -> Path:
    """Write a minicube in the real ``(H, W, C, T)`` layout."""
    cube = np.full((size, size, channels, frames), fill, dtype=np.float32)
    cube[:, :, -1, :] = cloudy
    if nan_at is not None:
        cube[nan_at] = np.nan
    np.savez_compressed(path, highresdynamic=cube)
    return path


class TestShapeAndLayout:
    def test_returns_time_major_tensor(self, tmp_path: Path):
        cube = read_minicube(write_cube(tmp_path / "c.npz", frames=6, size=4))
        assert cube.images.shape == (6, N_IMAGE_CHANNELS, 4, 4)

    def test_mask_has_a_singleton_channel_axis(self, tmp_path: Path):
        cube = read_minicube(write_cube(tmp_path / "c.npz", frames=6, size=4))
        assert cube.valid.shape == (6, 1, 4, 4)

    def test_only_the_first_four_channels_are_imagery(self, tmp_path: Path):
        cube = read_minicube(write_cube(tmp_path / "c.npz", channels=7))
        assert cube.n_channels == N_IMAGE_CHANNELS == len(BAND_NAMES)

    def test_dtype_is_float32(self, tmp_path: Path):
        cube = read_minicube(write_cube(tmp_path / "c.npz"))
        assert cube.images.dtype == torch.float32
        assert cube.valid.dtype == torch.float32

    def test_properties_report_geometry(self, tmp_path: Path):
        cube = read_minicube(write_cube(tmp_path / "c.npz", frames=8, size=5))
        assert cube.n_frames == 8
        assert cube.spatial_size == (5, 5)
        assert cube.cube_id == "c"

    def test_transpose_preserves_values(self, tmp_path: Path):
        """Guards against a wrong axis order silently scrambling the data."""
        size, frames = 3, 4
        raw = np.arange(size * size * 7 * frames, dtype=np.float32).reshape(size, size, 7, frames)
        raw = raw / raw.max()
        raw[:, :, -1, :] = 0.0
        np.savez_compressed(tmp_path / "c.npz", highresdynamic=raw)

        cube = read_minicube(tmp_path / "c.npz")
        for t in range(frames):
            for c in range(N_IMAGE_CHANNELS):
                expected = torch.from_numpy(raw[:, :, c, t])
                torch.testing.assert_close(cube.images[t, c], expected)


class TestMaskPolarity:
    def test_cldmsk_one_means_cloudy_so_valid_is_zero(self, tmp_path: Path):
        """The single highest-consequence convention in the pipeline."""
        cube = read_minicube(write_cube(tmp_path / "c.npz", cloudy=1.0))
        assert float(cube.valid.max()) == 0.0

    def test_cldmsk_zero_means_clear_so_valid_is_one(self, tmp_path: Path):
        cube = read_minicube(write_cube(tmp_path / "c.npz", cloudy=0.0))
        assert float(cube.valid.min()) == 1.0

    def test_partial_cloud_is_reflected_in_valid_fraction(self, tmp_path: Path):
        size, frames = 4, 2
        raw = np.full((size, size, 7, frames), 0.5, dtype=np.float32)
        raw[:, :, -1, :] = 0.0  # start fully clear
        raw[:2, :, -1, :] = 1.0  # top half cloudy
        np.savez_compressed(tmp_path / "c.npz", highresdynamic=raw)

        cube = read_minicube(tmp_path / "c.npz")
        assert cube.valid_fraction() == pytest.approx(0.5)

    def test_valid_fraction_accepts_a_time_slice(self, tmp_path: Path):
        size, frames = 4, 4
        raw = np.full((size, size, 7, frames), 0.5, dtype=np.float32)
        raw[:, :, -1, :] = 0.0  # start fully clear
        raw[:, :, -1, :2] = 1.0  # first two frames fully cloudy
        np.savez_compressed(tmp_path / "c.npz", highresdynamic=raw)

        cube = read_minicube(tmp_path / "c.npz")
        assert cube.valid_fraction(slice(0, 2)) == pytest.approx(0.0)
        assert cube.valid_fraction(slice(2, 4)) == pytest.approx(1.0)

    def test_apply_mask_false_marks_everything_valid(self, tmp_path: Path):
        cube = read_minicube(write_cube(tmp_path / "c.npz", cloudy=1.0), apply_mask=False)
        assert float(cube.valid.min()) == 1.0

    def test_nan_in_the_mask_channel_is_treated_as_unusable(self, tmp_path: Path):
        raw = np.full((3, 3, 7, 2), 0.5, dtype=np.float32)
        raw[:, :, -1, :] = np.nan
        np.savez_compressed(tmp_path / "c.npz", highresdynamic=raw)

        cube = read_minicube(tmp_path / "c.npz")
        assert float(cube.valid.max()) == 0.0


class TestCleaning:
    def test_nan_imagery_is_zero_filled(self, tmp_path: Path):
        path = write_cube(tmp_path / "c.npz", nan_at=(0, 0, 0, 0))
        cube = read_minicube(path)
        assert not torch.isnan(cube.images).any()
        assert float(cube.images[0, 0, 0, 0]) == 0.0

    def test_nan_imagery_is_marked_invalid_by_default(self, tmp_path: Path):
        path = write_cube(tmp_path / "c.npz", nan_at=(0, 0, 0, 0), cloudy=0.0)
        cube = read_minicube(path)
        assert float(cube.valid[0, 0, 0, 0]) == 0.0

    def test_nan_is_invalid_can_be_disabled_for_official_scoring(self, tmp_path: Path):
        path = write_cube(tmp_path / "c.npz", nan_at=(0, 0, 0, 0), cloudy=0.0)
        cube = read_minicube(path, nan_is_invalid=False)
        assert float(cube.valid[0, 0, 0, 0]) == 1.0

    def test_values_are_clipped_into_unit_range(self, tmp_path: Path):
        raw = np.full((3, 3, 7, 2), 0.5, dtype=np.float32)
        raw[0, 0, 0, 0] = 5.0
        raw[1, 1, 1, 0] = -3.0
        raw[:, :, -1, :] = 0.0
        np.savez_compressed(tmp_path / "c.npz", highresdynamic=raw)

        cube = read_minicube(tmp_path / "c.npz")
        assert float(cube.images.max()) <= 1.0
        assert float(cube.images.min()) >= 0.0


class TestErrors:
    def test_missing_file(self, tmp_path: Path):
        with pytest.raises(FileNotFoundError, match="No minicube at"):
            read_minicube(tmp_path / "absent.npz")

    def test_missing_highresdynamic_lists_what_was_found(self, tmp_path: Path):
        np.savez_compressed(tmp_path / "c.npz", something_else=np.zeros((2, 2)))
        with pytest.raises(MinicubeFormatError, match="something_else"):
            read_minicube(tmp_path / "c.npz")

    def test_wrong_rank_is_rejected(self, tmp_path: Path):
        np.savez_compressed(tmp_path / "c.npz", highresdynamic=np.zeros((4, 4, 7)))
        with pytest.raises(MinicubeFormatError, match="4 dimensions"):
            read_minicube(tmp_path / "c.npz")

    def test_too_few_channels_for_masking_is_rejected(self, tmp_path: Path):
        write_cube(tmp_path / "c.npz", channels=4)
        with pytest.raises(MinicubeFormatError, match="at least 5 channels"):
            read_minicube(tmp_path / "c.npz")

    def test_four_channel_cube_readable_without_masking(self, tmp_path: Path):
        write_cube(tmp_path / "c.npz", channels=4)
        cube = read_minicube(tmp_path / "c.npz", apply_mask=False)
        assert cube.n_channels == 4

    def test_five_channel_test_cube_uses_last_channel_as_mask(self, tmp_path: Path):
        """Test cubes may carry fewer channels; indexing [-1] must still work."""
        raw = np.full((3, 3, 5, 2), 0.5, dtype=np.float32)
        raw[:, :, -1, :] = 1.0
        np.savez_compressed(tmp_path / "c.npz", highresdynamic=raw)

        cube = read_minicube(tmp_path / "c.npz")
        assert cube.n_channels == 4
        assert float(cube.valid.max()) == 0.0
