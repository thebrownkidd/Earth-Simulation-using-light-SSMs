"""Spatial cropping of minicube windows.

EarthNet2021 minicubes are 128x128 pixels at 20 m resolution. Encoding and
decoding every one of those pixels dominates the cost of a training step --
measured on this project's CPU reference machine, a step at 128x128 costs about
18x a step at 32x32, tracking the 16x pixel ratio almost exactly.

That cost is spent on the parts of the model this project holds *fixed*. The
research question is about the temporal backbone, and the encoder and decoder
are identical across every architecture compared. Reducing spatial extent
therefore buys a large amount of compute at no cost to what is being measured,
which makes it the right place to economise -- unlike shortening the sequence,
which would attack the independent variable itself.

Cropping rather than downsampling
---------------------------------
Both shrink the tensor. Cropping is preferred here because it preserves the
native 20 m ground resolution, so the model sees real Sentinel-2 texture rather
than a blurred average, and because a random crop differs between epochs and so
acts as augmentation. The cost is field of view: a 32x32 crop covers 640 m
rather than 2.56 km.

Train on crops, predict on whole scenes
---------------------------------------
Nothing in the architecture is tied to the training crop size. The encoder and
decoder are fully convolutional, and the temporal backbone folds the latent grid
into the batch dimension, treating each location as an independent sequence --
so height and width never reach a shape-dependent parameter. A model trained on
32x32 crops therefore runs unchanged on a full 128x128 scene, which is how the
qualitative figures are produced.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

import torch

__all__ = ["CropMode", "SpatialCrop"]


class CropMode(StrEnum):
    """How a crop origin is chosen.

    Attributes:
        RANDOM: Uniformly random origin, redrawn per sample. Used for training,
            where it doubles as augmentation.
        CENTER: The centred origin. Used for validation and test, where a crop
            that moved between epochs would make the metric noisy for reasons
            unrelated to the model.
    """

    RANDOM = "random"
    CENTER = "center"


@dataclass(frozen=True)
class SpatialCrop:
    """A square crop applied identically to every tensor of one sample.

    The same origin must be used for imagery and masks, and for history and
    target alike: a mask cropped at a different origin from the imagery it
    describes would silently mislabel which pixels are cloudy, and a target
    cropped elsewhere than its history would ask the model to forecast a
    different place than it observed.

    Attributes:
        size: Side length in pixels.
        mode: How the origin is chosen.

    Raises:
        ValueError: If ``size`` is not positive.
    """

    size: int
    mode: CropMode = CropMode.RANDOM

    def __post_init__(self) -> None:
        """Validate the crop size.

        Raises:
            ValueError: If ``size`` is not positive.
        """
        if self.size < 1:
            raise ValueError(f"crop size must be >= 1, got {self.size}.")

    def origin(self, height: int, width: int) -> tuple[int, int]:
        """Choose a crop origin for a frame of the given size.

        Random origins are drawn from the ambient :mod:`torch` RNG. Under a
        :class:`~torch.utils.data.DataLoader` each worker process is seeded
        deterministically from the run's base seed, so this stays reproducible
        across runs while still differing between workers and epochs.

        Args:
            height: Source height in pixels.
            width: Source width in pixels.

        Returns:
            The ``(top, left)`` corner.

        Raises:
            ValueError: If the crop does not fit inside the source.
        """
        if self.size > height or self.size > width:
            raise ValueError(
                f"crop size {self.size} does not fit in a {height}x{width} frame. "
                "Lower data.crop_size, or remove it to use whole scenes."
            )

        span_y = height - self.size
        span_x = width - self.size
        if self.mode is CropMode.CENTER:
            return span_y // 2, span_x // 2

        top = int(torch.randint(span_y + 1, ())) if span_y else 0
        left = int(torch.randint(span_x + 1, ())) if span_x else 0
        return top, left

    def apply(self, tensor: torch.Tensor, top: int, left: int) -> torch.Tensor:
        """Crop the trailing two dimensions of a tensor.

        Args:
            tensor: Any tensor whose last two dimensions are spatial, such as
                ``[T, C, H, W]`` imagery or ``[T, 1, H, W]`` masks.
            top: Row of the crop's top-left corner.
            left: Column of the crop's top-left corner.

        Returns:
            The cropped view.
        """
        return tensor[..., top : top + self.size, left : left + self.size]

    @classmethod
    def for_evaluation(cls, size: int) -> SpatialCrop:
        """Return a centred crop of ``size``, for validation and test splits."""
        return cls(size=size, mode=CropMode.CENTER)
