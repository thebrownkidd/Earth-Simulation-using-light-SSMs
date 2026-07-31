"""Tests for temporal window generation.

Window indexing is the place where a silent off-by-one leaks future frames into
the history, which would invalidate every forecasting result without failing
anything. These tests are correspondingly picky.
"""

from __future__ import annotations

import pytest

from tinyearth.datasets.windows import Window, WindowMode, WindowSpec, generate_windows


class TestWindowSpec:
    def test_total_length(self):
        assert WindowSpec(history_length=4, horizon=2).total_length == 6

    @pytest.mark.parametrize(
        ("field", "value"),
        [("history_length", 0), ("horizon", 0), ("stride", 0), ("history_length", -1)],
    )
    def test_rejects_non_positive(self, field, value):
        kwargs = {"history_length": 4, "horizon": 1, "stride": 1, field: value}
        with pytest.raises(ValueError, match=f"{field} must be"):
            WindowSpec(**kwargs)

    def test_is_frozen(self):
        spec = WindowSpec(history_length=4, horizon=1)
        with pytest.raises(AttributeError):
            spec.history_length = 8  # type: ignore[misc]


class TestWindowSlices:
    def test_history_and_target_are_adjacent_and_disjoint(self):
        window = Window(start=3, history_length=4, horizon=2)
        assert window.history_slice == slice(3, 7)
        assert window.target_slice == slice(7, 9)
        assert window.stop == 9

    def test_target_never_overlaps_history(self):
        """The failure this guards is invisible: the model would see the answer."""
        for start in range(6):
            for history in range(1, 5):
                for horizon in range(1, 4):
                    window = Window(start, history, horizon)
                    history_frames = set(range(*window.history_slice.indices(100)))
                    target_frames = set(range(*window.target_slice.indices(100)))
                    assert not (history_frames & target_frames)

    def test_slices_cover_exactly_total_length(self):
        window = Window(start=2, history_length=5, horizon=3)
        assert window.stop - window.start == 8


class TestSlidingWindows:
    def test_count_matches_formula(self):
        spec = WindowSpec(history_length=4, horizon=1)
        windows = generate_windows(30, spec)
        assert len(windows) == 30 - 5 + 1 == 26

    def test_starts_are_contiguous_with_stride_one(self):
        windows = generate_windows(10, WindowSpec(history_length=2, horizon=1))
        assert [window.start for window in windows] == list(range(8))

    def test_stride_skips_starts(self):
        windows = generate_windows(10, WindowSpec(history_length=2, horizon=1, stride=3))
        assert [window.start for window in windows] == [0, 3, 6]

    def test_last_window_fits_inside_the_cube(self):
        spec = WindowSpec(history_length=4, horizon=2)
        windows = generate_windows(17, spec)
        assert windows[-1].stop <= 17

    def test_returns_empty_when_cube_is_too_short(self):
        assert generate_windows(4, WindowSpec(history_length=4, horizon=1)) == ()

    def test_exact_fit_yields_one_window(self):
        windows = generate_windows(5, WindowSpec(history_length=4, horizon=1))
        assert len(windows) == 1
        assert windows[0].start == 0

    def test_starts_are_strictly_increasing(self):
        windows = generate_windows(30, WindowSpec(history_length=3, horizon=2, stride=2))
        starts = [window.start for window in windows]
        assert starts == sorted(starts)
        assert len(set(starts)) == len(starts)

    @pytest.mark.parametrize("horizon", [1, 2, 4, 8])
    def test_all_swept_horizons_fit_a_real_cube(self, horizon):
        """The horizon sweep in the experiment plan must be runnable on 30 frames."""
        windows = generate_windows(30, WindowSpec(history_length=8, horizon=horizon))
        assert windows
        assert windows[-1].stop <= 30

    @pytest.mark.parametrize("history", [2, 4, 6, 8])
    def test_all_swept_history_lengths_fit_a_real_cube(self, history):
        windows = generate_windows(30, WindowSpec(history_length=history, horizon=1))
        assert windows


class TestAnchoredWindows:
    def test_history_ends_exactly_at_the_context_boundary(self):
        spec = WindowSpec(history_length=4, horizon=2)
        (window,) = generate_windows(30, spec, mode=WindowMode.ANCHORED, context_length=10)
        assert window.history_slice == slice(6, 10)
        assert window.target_slice == slice(10, 12)

    def test_official_protocol_uses_all_ten_context_frames(self):
        spec = WindowSpec(history_length=10, horizon=20)
        (window,) = generate_windows(30, spec, mode=WindowMode.ANCHORED, context_length=10)
        assert window.start == 0
        assert window.stop == 30

    def test_yields_exactly_one_window(self):
        windows = generate_windows(
            30, WindowSpec(history_length=4, horizon=1), mode=WindowMode.ANCHORED, context_length=10
        )
        assert len(windows) == 1

    def test_requires_context_length(self):
        with pytest.raises(ValueError, match="context_length is required"):
            generate_windows(30, WindowSpec(history_length=4, horizon=1), mode=WindowMode.ANCHORED)

    def test_rejects_history_longer_than_context(self):
        with pytest.raises(ValueError, match="exceeds context_length"):
            generate_windows(
                30,
                WindowSpec(history_length=12, horizon=1),
                mode=WindowMode.ANCHORED,
                context_length=10,
            )

    def test_rejects_horizon_running_past_the_cube(self):
        with pytest.raises(ValueError, match="but the cube has only"):
            generate_windows(
                30,
                WindowSpec(history_length=4, horizon=25),
                mode=WindowMode.ANCHORED,
                context_length=10,
            )
