"""DataLoader construction.

Centralised so that the reproducibility requirements are met by default rather
than by remembering. Two are easy to get wrong and near-invisible when missed:

* **``generator``** controls shuffle order. Left to the global RNG, epoch order
  depends on how many random numbers the *model* happened to draw -- so adding a
  dropout layer silently changes the data order, and an "identical" rerun is not.
* **``worker_init_fn``** reseeds ``random`` and ``numpy`` per worker. PyTorch
  seeds each worker's ``torch`` RNG but leaves the other two identical, so N
  workers apply the same augmentations and effective diversity drops N-fold.

Both are wired in :func:`build_dataloader`; see ``docs/reproducibility.md``.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch.utils.data import DataLoader, Dataset

from tinyearth.datasets.types import Batch, Sample, collate_samples
from tinyearth.utils.logging import get_logger
from tinyearth.utils.seed import seeded_generator, worker_init_fn

__all__ = ["LoaderSettings", "build_dataloader"]

logger = get_logger(__name__)


@dataclass(frozen=True)
class LoaderSettings:
    """DataLoader configuration.

    Attributes:
        batch_size: Samples per batch.
        num_workers: Worker processes. ``0`` loads in the main process, which
            is the right choice on Windows for small datasets and for debugging,
            since worker startup there is expensive.
        shuffle: Shuffle sample order. Should be ``False`` for evaluation so
            that per-sample results line up across runs.
        drop_last: Drop a trailing partial batch. Useful for training stability
            and for throughput measurement, where a short final batch skews the
            per-batch timing.
        pin_memory: Page-lock host memory for faster host-to-device copies.
            Only meaningful with CUDA.
        persistent_workers: Keep workers alive between epochs. Saves startup
            cost; requires ``num_workers > 0``.
        prefetch_factor: Batches prefetched per worker. Requires
            ``num_workers > 0``.

    Raises:
        ValueError: If a field is out of range, or if worker-only options are
            set with ``num_workers == 0``.
    """

    batch_size: int = 8
    num_workers: int = 0
    shuffle: bool = True
    drop_last: bool = False
    pin_memory: bool = False
    persistent_workers: bool = False
    prefetch_factor: int | None = None

    def __post_init__(self) -> None:
        """Validate the settings."""
        if self.batch_size < 1:
            raise ValueError(f"batch_size must be >= 1, got {self.batch_size}.")
        if self.num_workers < 0:
            raise ValueError(f"num_workers must be >= 0, got {self.num_workers}.")
        if self.num_workers == 0:
            if self.persistent_workers:
                raise ValueError("persistent_workers requires num_workers > 0.")
            if self.prefetch_factor is not None:
                raise ValueError("prefetch_factor requires num_workers > 0.")
        if self.prefetch_factor is not None and self.prefetch_factor < 1:
            raise ValueError(f"prefetch_factor must be >= 1, got {self.prefetch_factor}.")


def build_dataloader(
    dataset: Dataset[Sample],
    settings: LoaderSettings,
    *,
    seed: int = 42,
    device: torch.device | None = None,
) -> DataLoader[Sample]:
    """Build a reproducible :class:`~torch.utils.data.DataLoader`.

    Args:
        dataset: Dataset yielding :class:`~tinyearth.datasets.types.Sample`.
        settings: Loader configuration.
        seed: Seed for the shuffle generator. Vary it per epoch only via
            :meth:`~torch.utils.data.DataLoader.__iter__`; the generator itself
            stays fixed so a rerun reproduces the same epoch order.
        device: Target device. When given, ``pin_memory`` is enabled only if the
            device is CUDA -- pinning without a GPU costs memory and buys nothing.

    Returns:
        The configured loader.
    """
    pin_memory = settings.pin_memory
    if device is not None and device.type != "cuda" and pin_memory:
        logger.debug("disabling pin_memory: device is %s, not cuda", device.type)
        pin_memory = False

    extra: dict[str, object] = {}
    if settings.num_workers > 0:
        extra["persistent_workers"] = settings.persistent_workers
        if settings.prefetch_factor is not None:
            extra["prefetch_factor"] = settings.prefetch_factor

    return DataLoader(
        dataset,
        batch_size=settings.batch_size,
        shuffle=settings.shuffle,
        drop_last=settings.drop_last,
        num_workers=settings.num_workers,
        pin_memory=pin_memory,
        collate_fn=collate_samples,
        generator=seeded_generator(seed),
        worker_init_fn=worker_init_fn,
        **extra,  # type: ignore[arg-type]
    )


def describe_batch(batch: Batch) -> str:
    """Summarise a batch's shapes, for logging and debugging.

    Args:
        batch: A collated batch.

    Returns:
        A one-line description.
    """
    parts = [
        f"images={tuple(batch['images'].shape)}",
        f"target={tuple(batch['target'].shape)}",
    ]
    if "target_mask" in batch:
        valid = float(batch["target_mask"].mean().item())
        parts.append(f"target_valid={valid:.3f}")
    return " ".join(parts)
