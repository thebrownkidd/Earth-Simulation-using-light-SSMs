"""Normalisation of reflectance imagery.

Sentinel-2 reflectance already lies in ``[0, 1]``, so normalisation is a
modelling choice rather than a necessity. Both options are provided because
they interact with the decoder's output activation:

* :class:`IdentityNormalizer` keeps data in ``[0, 1]``, which pairs with a
  sigmoid output and makes reconstruction losses directly interpretable as
  reflectance error. This is the default.
* :class:`ChannelStandardizer` gives zero mean and unit variance per channel,
  which conditions optimisation better but requires the decoder to emit
  unbounded values and the metrics to invert the transform before reporting.

Every normaliser is invertible, because predictions must be returned to
reflectance space before they can be scored or plotted.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, runtime_checkable

import torch

from tinyearth.utils.logging import get_logger

__all__ = [
    "ChannelStandardizer",
    "ChannelStatistics",
    "IdentityNormalizer",
    "Normalizer",
    "build_normalizer",
    "compute_channel_statistics",
]

logger = get_logger(__name__)

_EPS = 1e-6


@runtime_checkable
class Normalizer(Protocol):
    """Invertible per-channel transform applied to ``[T, C, H, W]`` imagery."""

    def __call__(self, images: torch.Tensor) -> torch.Tensor:
        """Normalise imagery.

        Args:
            images: Reflectance, ``[..., C, H, W]``.

        Returns:
            Normalised imagery of the same shape.
        """
        ...

    def invert(self, images: torch.Tensor) -> torch.Tensor:
        """Map normalised imagery back to reflectance.

        Args:
            images: Normalised imagery, ``[..., C, H, W]``.

        Returns:
            Reflectance of the same shape.
        """
        ...


@dataclass(frozen=True)
class IdentityNormalizer:
    """Pass imagery through unchanged.

    The correct default for Sentinel-2 reflectance, which is already bounded.
    """

    def __call__(self, images: torch.Tensor) -> torch.Tensor:
        """Return ``images`` unchanged."""
        return images

    def invert(self, images: torch.Tensor) -> torch.Tensor:
        """Return ``images`` unchanged."""
        return images


@dataclass(frozen=True)
class ChannelStatistics:
    """Per-channel mean and standard deviation.

    Attributes:
        mean: One value per channel.
        std: One value per channel. Values below :data:`_EPS` are rejected,
            since a constant channel cannot be standardised.
        n_pixels: Number of valid pixels the statistics were computed over.
            Recorded so a reader can judge whether the estimate is trustworthy.
    """

    mean: tuple[float, ...]
    std: tuple[float, ...]
    n_pixels: int = 0

    def __post_init__(self) -> None:
        """Validate shape agreement and positive standard deviations."""
        if len(self.mean) != len(self.std):
            raise ValueError(f"mean has {len(self.mean)} channels but std has {len(self.std)}.")
        if not self.mean:
            raise ValueError("Statistics must cover at least one channel.")
        for index, value in enumerate(self.std):
            if value < _EPS:
                raise ValueError(
                    f"std[{index}]={value} is too small to standardise with. "
                    "A near-constant channel should be dropped rather than scaled."
                )

    def save(self, path: Path | str) -> Path:
        """Write statistics to JSON.

        Args:
            path: Destination file. Parent directories are created.

        Returns:
            The path written.
        """
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(
            json.dumps(
                {"mean": list(self.mean), "std": list(self.std), "n_pixels": self.n_pixels},
                indent=2,
            ),
            encoding="utf-8",
        )
        return destination

    @classmethod
    def load(cls, path: Path | str) -> ChannelStatistics:
        """Read statistics from JSON.

        Args:
            path: Source file written by :meth:`save`.

        Returns:
            The loaded statistics.

        Raises:
            FileNotFoundError: If the file is missing.
        """
        source = Path(path)
        if not source.is_file():
            raise FileNotFoundError(
                f"No channel statistics at {source}. Generate them with "
                "`python scripts/compute_dataset_statistics.py`."
            )
        payload = json.loads(source.read_text(encoding="utf-8"))
        return cls(
            mean=tuple(float(value) for value in payload["mean"]),
            std=tuple(float(value) for value in payload["std"]),
            n_pixels=int(payload.get("n_pixels", 0)),
        )


class ChannelStandardizer:
    """Standardise each channel to zero mean and unit variance.

    Statistics must come from the **training** split only. Computing them over
    validation or test data leaks distributional information and inflates
    reported quality.

    Attributes:
        statistics: The statistics being applied.
    """

    def __init__(self, statistics: ChannelStatistics) -> None:
        """Create a standardiser.

        Args:
            statistics: Per-channel mean and standard deviation.
        """
        self.statistics = statistics
        # Shaped [C, 1, 1] so it broadcasts against [..., C, H, W].
        self._mean = torch.tensor(statistics.mean, dtype=torch.float32).view(-1, 1, 1)
        self._std = torch.tensor(statistics.std, dtype=torch.float32).view(-1, 1, 1)

    def _check_channels(self, images: torch.Tensor) -> None:
        """Fail loudly on a channel-count mismatch.

        Args:
            images: Imagery to check.

        Raises:
            ValueError: If the channel count disagrees with the statistics.
        """
        expected = self._mean.shape[0]
        actual = images.shape[-3]
        if actual != expected:
            raise ValueError(
                f"Statistics cover {expected} channels but imagery has {actual}. "
                "Recompute statistics for the channel subset in use."
            )

    def __call__(self, images: torch.Tensor) -> torch.Tensor:
        """Standardise imagery."""
        self._check_channels(images)
        return (images - self._mean.to(images.device)) / self._std.to(images.device)

    def invert(self, images: torch.Tensor) -> torch.Tensor:
        """Return standardised imagery to reflectance."""
        self._check_channels(images)
        return images * self._std.to(images.device) + self._mean.to(images.device)

    def __repr__(self) -> str:
        """Return a debugging representation."""
        return f"ChannelStandardizer(mean={self.statistics.mean}, std={self.statistics.std})"


def build_normalizer(
    kind: str,
    *,
    statistics: ChannelStatistics | None = None,
) -> Normalizer:
    """Construct a normaliser by name.

    Args:
        kind: ``"identity"`` or ``"standardize"``.
        statistics: Required for ``"standardize"``.

    Returns:
        The constructed normaliser.

    Raises:
        ValueError: If ``kind`` is unknown, or ``"standardize"`` is requested
            without statistics.
    """
    normalised = kind.strip().lower()
    if normalised == "identity":
        return IdentityNormalizer()
    if normalised in {"standardize", "standardise"}:
        if statistics is None:
            raise ValueError(
                "normalization='standardize' requires channel statistics. Run "
                "`python scripts/compute_dataset_statistics.py` and point "
                "data.normalization.statistics_path at the result."
            )
        return ChannelStandardizer(statistics)
    raise ValueError(f"Unknown normalization {kind!r}. Expected 'identity' or 'standardize'.")


def compute_channel_statistics(
    batches: Iterable[tuple[torch.Tensor, torch.Tensor | None]],
    n_channels: int,
) -> ChannelStatistics:
    """Accumulate per-channel statistics over masked imagery.

    Uses a streaming sum of values and squares so that arbitrarily large
    datasets fit in memory. Masked-out pixels are excluded, which matters:
    cloudy pixels are near-white and would otherwise bias every channel mean
    upwards.

    Args:
        batches: Pairs of ``(images, valid)``. ``images`` is ``[..., C, H, W]``;
            ``valid`` is a broadcastable mask where 1 means usable, or ``None``
            to count every pixel.
        n_channels: Expected channel count.

    Returns:
        The accumulated statistics.

    Raises:
        ValueError: If no valid pixels were seen, which usually means the mask
            convention is inverted somewhere upstream.
    """
    total = torch.zeros(n_channels, dtype=torch.float64)
    total_sq = torch.zeros(n_channels, dtype=torch.float64)
    count = torch.zeros(n_channels, dtype=torch.float64)

    for images, valid in batches:
        values = images.to(torch.float64)
        # Collapse everything except the channel axis.
        channel_first = values.movedim(-3, 0).reshape(n_channels, -1)

        if valid is None:
            weights = torch.ones_like(channel_first)
        else:
            mask = valid.to(torch.float64).expand_as(values)
            weights = mask.movedim(-3, 0).reshape(n_channels, -1)

        total += (channel_first * weights).sum(dim=1)
        total_sq += (channel_first.pow(2) * weights).sum(dim=1)
        count += weights.sum(dim=1)

    if float(count.min().item()) <= 0:
        raise ValueError(
            "No valid pixels were observed while computing statistics. Check "
            "that the cloud-mask convention is correct (1 must mean *usable* "
            "after inversion)."
        )

    mean = total / count
    variance = (total_sq / count) - mean.pow(2)
    std = variance.clamp_min(_EPS).sqrt()

    logger.info(
        "channel statistics over %.3g valid pixels: mean=%s std=%s",
        float(count.min().item()),
        [round(value, 5) for value in mean.tolist()],
        [round(value, 5) for value in std.tolist()],
    )

    return ChannelStatistics(
        mean=tuple(float(value) for value in mean.tolist()),
        std=tuple(float(value) for value in std.tolist()),
        n_pixels=int(count.min().item()),
    )


def statistics_from_sequence(
    means: Sequence[float],
    stds: Sequence[float],
) -> ChannelStatistics:
    """Build statistics from literal values, e.g. from a config file.

    Args:
        means: Per-channel means.
        stds: Per-channel standard deviations.

    Returns:
        The constructed statistics.
    """
    return ChannelStatistics(
        mean=tuple(float(value) for value in means),
        std=tuple(float(value) for value in stds),
    )
