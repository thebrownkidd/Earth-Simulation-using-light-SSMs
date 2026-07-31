"""``tinyearth-data``: build the data pipeline and report on it.

The Phase 2 entry point and acceptance check. It constructs the configured
dataset, draws a batch, and reports shapes, cloud statistics and split sizes --
without needing a model.

It also verifies reproducibility directly: two independently constructed
loaders with the same seed must yield bit-identical first batches. That check
is cheap and catches the class of bug where shuffling silently depends on
global RNG state.

Example:
    ```bash
    tinyearth-data                                  # synthetic, default config
    tinyearth-data +experiment=data_smoke           # the Phase 2 experiment
    tinyearth-data data=earthnet2021                # the real dataset
    tinyearth-data data.history_length=8 data.horizon=4
    ```
"""

from __future__ import annotations

from typing import cast

import torch
from omegaconf import DictConfig
from torch.utils.data import DataLoader

from tinyearth.bootstrap import RunContext, initialise_run
from tinyearth.cli._hydra import run_with_hydra
from tinyearth.config.resolution import to_dataclass
from tinyearth.datasets.factory import build_datamodule, resolve_dataset_root
from tinyearth.datasets.loaders import describe_batch
from tinyearth.datasets.splits import Split
from tinyearth.datasets.types import Batch, Sample
from tinyearth.utils.logging import log_section

__all__ = ["main"]


def _first_batch(loader: DataLoader[Sample]) -> Batch:
    """Draw the first batch from a loader.

    Args:
        loader: A dataloader over samples.

    Returns:
        The first collated batch.
    """
    # DataLoader.__iter__ is typed as yielding Any, so the cast records what
    # collate_samples actually guarantees.
    return cast("Batch", next(iter(loader)))


def _tensor_fields(batch: Batch) -> list[tuple[str, torch.Tensor]]:
    """Return the tensor entries present in ``batch``, in a stable order.

    Listing the keys explicitly rather than indexing by a loop variable keeps
    the :class:`~tinyearth.datasets.types.Batch` TypedDict statically checkable.

    Args:
        batch: A collated batch.

    Returns:
        ``(label, tensor)`` pairs; mask entries are omitted when masking is off.
    """
    fields: list[tuple[str, torch.Tensor]] = [
        ("images", batch["images"]),
        ("target", batch["target"]),
    ]
    if "images_mask" in batch:
        fields.append(("images_mask", batch["images_mask"]))
    if "target_mask" in batch:
        fields.append(("target_mask", batch["target_mask"]))
    return fields


def report_data(context: RunContext) -> None:
    """Build the configured pipeline and log a full report.

    Args:
        context: An initialised run context.

    Raises:
        ValueError: If the composed config carries no ``data`` group.
    """
    logger = context.logger
    cfg = to_dataclass(context.cfg)
    if cfg.data is None:
        raise ValueError(
            "No data group in the config. Compose one with `data=synthetic` or "
            "`data=earthnet2021`."
        )

    log_section(logger, "Dataset")
    root = resolve_dataset_root(cfg.data, context.paths)
    logger.info("  %-18s %s", "name", cfg.data.name)
    logger.info("  %-18s %s", "root", root)
    logger.info("  %-18s %s", "split", cfg.data.split)
    logger.info("  %-18s %d -> %d", "history -> horizon", cfg.data.history_length, cfg.data.horizon)
    logger.info("  %-18s %s", "cloud_masking", cfg.data.cloud_masking)
    logger.info("  %-18s %s", "mask_policy", cfg.data.mask_policy)
    logger.info("  %-18s %s", "normalization", cfg.data.normalization.kind)

    train = build_datamodule(cfg.data, context.paths, split=Split.TRAIN, seed=context.seed)
    val = build_datamodule(cfg.data, context.paths, split=Split.VAL, seed=context.seed)

    log_section(logger, "Splits")
    for bundle in (train, val):
        logger.info(
            "  %-18s %4d cubes  %6d windows  %5d batches",
            bundle.split.value,
            len(bundle.dataset.cubes),
            len(bundle.dataset),
            len(bundle.loader),
        )
    overlap = set(train.dataset.cubes) & set(val.dataset.cubes)
    logger.info("  %-18s %d", "train/val overlap", len(overlap))
    if overlap:
        logger.error("train and validation share cubes -- the partition is broken")

    log_section(logger, "Batch")
    batch = _first_batch(train.loader)
    logger.info("  %s", describe_batch(batch))
    for label, tensor in _tensor_fields(batch):
        logger.info(
            "  %-18s %-24s min=%.4f max=%.4f mean=%.4f",
            label,
            str(tuple(tensor.shape)),
            float(tensor.min()),
            float(tensor.max()),
            float(tensor.mean()),
        )
    first = batch["metadata"][0]
    logger.info("  %-18s %s", "cube_id[0]", first.cube_id)
    logger.info("  %-18s %d", "start_index[0]", first.start_index)
    logger.info("  %-18s %.4f", "valid_fraction[0]", first.valid_fraction)

    log_section(logger, "Reproducibility")
    repeat = build_datamodule(cfg.data, context.paths, split=Split.TRAIN, seed=context.seed)
    other = _first_batch(repeat.loader)
    identical = torch.equal(batch["images"], other["images"]) and torch.equal(
        batch["target"], other["target"]
    )
    logger.info("  %-18s %s", "same seed -> same batch", identical)
    if not identical:
        logger.error("two loaders with the same seed produced different batches")

    stats = train.dataset.cache_statistics()
    logger.info(
        "  %-18s hits=%d misses=%d size=%d",
        "cube cache",
        stats["hits"],
        stats["misses"],
        stats["size"],
    )


def main() -> None:
    """Console-script entry point for ``tinyearth-data``."""

    def entrypoint(cfg: DictConfig) -> None:
        """Initialise the run and report on the data pipeline."""
        context = initialise_run(cfg)
        report_data(context)
        context.logger.info("data pipeline OK")

    run_with_hydra(entrypoint)


if __name__ == "__main__":
    main()
