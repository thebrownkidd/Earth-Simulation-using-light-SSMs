"""Temporal window generation.

A minicube is a long sequence; a training sample is a short
``(history, forecast)`` window cut from it. This module owns that cutting.

It is deliberately free of I/O and of PyTorch: windows are computed from frame
*counts*, not from data. That keeps the indexing logic -- the part most likely
to harbour an off-by-one that silently leaks future frames into the history --
cheap to test exhaustively.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

__all__ = ["Window", "WindowMode", "WindowSpec", "generate_windows"]


class WindowMode(StrEnum):
    """How windows are cut from a cube.

    Attributes:
        SLIDING: Every valid start offset, advanced by ``stride``. Used for
            training, where each cube should yield many samples.
        ANCHORED: A single window whose history ends exactly at a given
            boundary. Used for evaluation, so that the model always forecasts
            from the same point and results are comparable across runs and
            against the published EarthNet2021 protocol.
    """

    SLIDING = "sliding"
    ANCHORED = "anchored"


@dataclass(frozen=True)
class WindowSpec:
    """Window geometry.

    Attributes:
        history_length: Number of context frames fed to the model.
        horizon: Number of frames to forecast.
        stride: Step between consecutive sliding window starts. ``1`` yields
            maximally overlapping samples; larger values trade sample count for
            decorrelation between samples.

    Raises:
        ValueError: If any field is non-positive.
    """

    history_length: int
    horizon: int
    stride: int = 1

    def __post_init__(self) -> None:
        """Validate the geometry."""
        if self.history_length < 1:
            raise ValueError(f"history_length must be >= 1, got {self.history_length}.")
        if self.horizon < 1:
            raise ValueError(f"horizon must be >= 1, got {self.horizon}.")
        if self.stride < 1:
            raise ValueError(f"stride must be >= 1, got {self.stride}.")

    @property
    def total_length(self) -> int:
        """Total frames consumed by one window."""
        return self.history_length + self.horizon


@dataclass(frozen=True)
class Window:
    """A concrete window into a cube.

    Attributes:
        start: Index of the first history frame.
        history_length: Number of context frames.
        horizon: Number of forecast frames.
    """

    start: int
    history_length: int
    horizon: int

    @property
    def history_slice(self) -> slice:
        """Frames used as model input."""
        return slice(self.start, self.start + self.history_length)

    @property
    def target_slice(self) -> slice:
        """Frames the model must forecast."""
        boundary = self.start + self.history_length
        return slice(boundary, boundary + self.horizon)

    @property
    def stop(self) -> int:
        """Exclusive end index of the window."""
        return self.start + self.history_length + self.horizon


def generate_windows(
    n_frames: int,
    spec: WindowSpec,
    *,
    mode: WindowMode = WindowMode.SLIDING,
    context_length: int | None = None,
) -> tuple[Window, ...]:
    """Enumerate the windows a cube of ``n_frames`` frames yields.

    Args:
        n_frames: Number of frames available in the cube.
        spec: Window geometry.
        mode: See :class:`WindowMode`.
        context_length: Required for :attr:`WindowMode.ANCHORED`. The history
            ends here, so the window starts at ``context_length - history_length``.
            For EarthNet2021 test cubes this is 10.

    Returns:
        Windows in increasing start order. Empty when the cube is too short --
        callers should skip such cubes rather than treat this as an error, since
        a shorter-than-usual cube is a property of the data, not a bug.

    Raises:
        ValueError: If ``mode`` is ``ANCHORED`` and ``context_length`` is absent,
            or if the anchored window does not fit within the cube.
    """
    if mode is WindowMode.ANCHORED:
        if context_length is None:
            raise ValueError("context_length is required for WindowMode.ANCHORED.")
        start = context_length - spec.history_length
        if start < 0:
            raise ValueError(
                f"history_length={spec.history_length} exceeds context_length="
                f"{context_length}; the window would need frames from before the "
                "start of the cube."
            )
        if start + spec.total_length > n_frames:
            raise ValueError(
                f"An anchored window needs frames [{start}, {start + spec.total_length}) "
                f"but the cube has only {n_frames}. Reduce horizon or history_length."
            )
        return (Window(start, spec.history_length, spec.horizon),)

    last_start = n_frames - spec.total_length
    if last_start < 0:
        return ()
    return tuple(
        Window(start, spec.history_length, spec.horizon)
        for start in range(0, last_start + 1, spec.stride)
    )
