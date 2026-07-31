"""The sample contract shared by every TinyEarth dataset.

Every dataset in this project yields the same structure, so that swapping a
data source never requires touching a model, a loss or the training loop.

Shape convention
----------------
Tensors are ``[T, C, H, W]`` -- time-major, channels second. This matches
:class:`torch.nn.Conv3d` and the ``(B, T, C, H, W)`` batch layout used
throughout, and avoids a permute in the encoder.

``target`` keeps its leading time axis even when the forecast horizon is 1.
The alternative -- returning ``[C, H, W]`` for ``horizon == 1`` and
``[K, C, H, W]`` otherwise -- makes tensor *rank* depend on a config value,
which forces every loss, metric and collate function downstream to branch. A
horizon-1 target is simply ``[1, C, H, W]``.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, NotRequired, TypedDict

import torch

__all__ = ["Batch", "Sample", "SampleMetadata", "collate_samples"]


@dataclass(frozen=True)
class SampleMetadata:
    """Provenance for a single training sample.

    Enough to locate the exact source frames again, which is what makes a
    surprising prediction investigable rather than merely noted.

    Attributes:
        cube_id: Identifier of the source minicube, without file extension.
        split: Split the sample was drawn from.
        source: Absolute path of the source file, as a string.
        start_index: Index of the first history frame within the source cube.
        history_length: Number of history frames.
        horizon: Number of forecast frames.
        valid_fraction: Fraction of target pixels that are cloud-free, in
            ``[0, 1]``. ``1.0`` when cloud masking is disabled.
        extra: Additional dataset-specific fields.
    """

    cube_id: str
    split: str
    source: str
    start_index: int
    history_length: int
    horizon: int
    valid_fraction: float = 1.0
    extra: dict[str, Any] = field(default_factory=dict)


class Sample(TypedDict):
    """One dataset item.

    Keys:
        images: History frames, ``[T, C, H, W]``.
        target: Frames to forecast, ``[K, C, H, W]``.
        metadata: Provenance for this sample.
        images_mask: Validity mask for ``images``, ``[T, 1, H, W]``, where 1
            means the pixel is cloud-free and usable. Present only when cloud
            masking is enabled.
        target_mask: Validity mask for ``target``, ``[K, 1, H, W]``. Present
            only when cloud masking is enabled.
    """

    images: torch.Tensor
    target: torch.Tensor
    metadata: SampleMetadata
    images_mask: NotRequired[torch.Tensor]
    target_mask: NotRequired[torch.Tensor]


class Batch(TypedDict):
    """A collated batch of samples.

    Tensors gain a leading batch axis; metadata becomes a list, since it holds
    strings and is not stackable.

    Keys:
        images: ``[B, T, C, H, W]``.
        target: ``[B, K, C, H, W]``.
        metadata: Per-sample provenance, length ``B``.
        images_mask: ``[B, T, 1, H, W]``, when masking is enabled.
        target_mask: ``[B, K, 1, H, W]``, when masking is enabled.
    """

    images: torch.Tensor
    target: torch.Tensor
    metadata: list[SampleMetadata]
    images_mask: NotRequired[torch.Tensor]
    target_mask: NotRequired[torch.Tensor]


def collate_samples(samples: Sequence[Sample]) -> Batch:
    """Collate samples into a batch.

    PyTorch's default collate cannot handle :class:`SampleMetadata`, and
    converting metadata to tensors would lose the identifiers that make a
    sample traceable. This keeps metadata as a plain list.

    Args:
        samples: Samples to collate. Must be non-empty, and must agree on
            which optional mask keys are present.

    Returns:
        The collated batch.

    Raises:
        ValueError: If ``samples`` is empty, or if mask keys are inconsistent
            across samples -- which would otherwise produce a batch that
            silently drops masks for some items.
    """
    if not samples:
        raise ValueError("Cannot collate an empty sequence of samples.")

    has_masks = "images_mask" in samples[0]
    if any(("images_mask" in sample) != has_masks for sample in samples):
        raise ValueError(
            "Inconsistent mask keys across samples: some carry 'images_mask' and "
            "some do not. All samples in a batch must come from a dataset with "
            "the same cloud-masking setting."
        )

    batch: Batch = {
        "images": torch.stack([sample["images"] for sample in samples]),
        "target": torch.stack([sample["target"] for sample in samples]),
        "metadata": [sample["metadata"] for sample in samples],
    }
    if has_masks:
        batch["images_mask"] = torch.stack([sample["images_mask"] for sample in samples])
        batch["target_mask"] = torch.stack([sample["target_mask"] for sample in samples])
    return batch
