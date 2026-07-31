"""Building datasets and dataloaders from configuration.

The single place that translates a validated
:class:`~tinyearth.config.schema.DataConfig` into live objects. Keeping the
translation here means the dataset classes stay independent of Hydra and remain
usable from a notebook or a plain script.

Two conventions are applied that the config alone cannot express:

* **Root resolution.** Real data resolves under ``paths.data``; synthetic data
  resolves under ``paths.cache``, because generated cubes are derived artefacts
  and should not sit in the directory a user populated by hand.
* **Shuffling is forced off for evaluation splits.** Shuffled evaluation makes
  per-sample results non-comparable across runs, and it is easy to leave
  ``shuffle: true`` in an inherited config by accident.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from tinyearth.config.resolution import ResolvedPaths
from tinyearth.config.schema import DataConfig
from tinyearth.datasets.earthnet2021 import EarthNet2021Dataset
from tinyearth.datasets.loaders import LoaderSettings, build_dataloader
from tinyearth.datasets.masking import MaskPolicy
from tinyearth.datasets.normalization import ChannelStatistics, Normalizer, build_normalizer
from tinyearth.datasets.splits import Split
from tinyearth.datasets.synthetic import SyntheticEarthNet2021, SyntheticSpec
from tinyearth.datasets.types import Sample
from tinyearth.utils.logging import get_logger

__all__ = ["DataBundle", "build_datamodule", "build_dataset", "resolve_dataset_root"]

logger = get_logger(__name__)

_SYNTHETIC = "synthetic"
_EARTHNET = "earthnet2021"


@dataclass(frozen=True)
class DataBundle:
    """A dataset paired with its loader.

    Attributes:
        dataset: The constructed dataset.
        loader: A dataloader over it.
        split: Which split this covers.
    """

    dataset: EarthNet2021Dataset
    loader: DataLoader[Sample]
    split: Split

    def __len__(self) -> int:
        """Number of samples in the dataset."""
        return len(self.dataset)


def resolve_dataset_root(cfg: DataConfig, paths: ResolvedPaths) -> Path:
    """Resolve the dataset root to an absolute path.

    Args:
        cfg: Data configuration.
        paths: Resolved project paths.

    Returns:
        An absolute path. Absolute values in ``cfg.root`` are used as given;
        relative values resolve under ``paths.cache`` for the synthetic dataset
        and ``paths.data`` otherwise.
    """
    root = Path(cfg.root).expanduser()
    if root.is_absolute():
        return root
    base = paths.cache if cfg.name == _SYNTHETIC else paths.data
    return base / root


def _build_normalizer(cfg: DataConfig, paths: ResolvedPaths) -> Normalizer:
    """Construct the configured normaliser, loading statistics if needed."""
    statistics = None
    if cfg.normalization.statistics_path:
        path = Path(cfg.normalization.statistics_path).expanduser()
        if not path.is_absolute():
            path = paths.cache / path
        statistics = ChannelStatistics.load(path)
    return build_normalizer(cfg.normalization.kind, statistics=statistics)


def build_dataset(
    cfg: DataConfig,
    paths: ResolvedPaths,
    *,
    split: Split | str | None = None,
) -> EarthNet2021Dataset:
    """Construct the dataset described by ``cfg``.

    Args:
        cfg: Data configuration.
        paths: Resolved project paths.
        split: Override the configured split, for building train and validation
            datasets from one config.

    Returns:
        The constructed dataset.

    Raises:
        ValueError: If ``cfg.name`` is not a known dataset.
    """
    requested = cfg.split if split is None else split
    resolved_split = Split(requested) if isinstance(requested, str) else requested

    root = resolve_dataset_root(cfg, paths)
    channels = tuple(cfg.channels) if cfg.channels else None

    shared: dict[str, object] = {
        "history_length": cfg.history_length,
        "horizon": cfg.horizon,
        "stride": cfg.stride,
        "channels": channels,
        "cloud_masking": cfg.cloud_masking,
        "mask_policy": MaskPolicy(cfg.mask_policy),
        "min_valid_fraction": cfg.min_valid_fraction,
        "normalizer": _build_normalizer(cfg, paths),
        "val_fraction": cfg.val_fraction,
        "split_salt": cfg.split_salt,
        "cache_size": cfg.cache_size,
        "nan_is_invalid": cfg.nan_is_invalid,
        "max_cubes": cfg.max_cubes,
    }

    if cfg.name == _SYNTHETIC:
        spec = SyntheticSpec(
            n_cubes=cfg.synthetic.n_cubes,
            n_frames=cfg.synthetic.n_frames,
            size=cfg.synthetic.size,
            cloud_fraction=cfg.synthetic.cloud_fraction,
            nan_fraction=cfg.synthetic.nan_fraction,
            seed=cfg.synthetic.seed,
        )
        dataset = SyntheticEarthNet2021(root, resolved_split, spec=spec, **shared)
        logger.warning("%s", dataset.describe())
        return dataset

    if cfg.name == _EARTHNET:
        return EarthNet2021Dataset(
            root,
            resolved_split,
            expected_frames=cfg.expected_frames,
            context_frames=cfg.context_frames,
            **shared,  # type: ignore[arg-type]
        )

    raise ValueError(f"Unknown dataset {cfg.name!r}. Available: {_EARTHNET!r}, {_SYNTHETIC!r}.")


def build_datamodule(
    cfg: DataConfig,
    paths: ResolvedPaths,
    *,
    split: Split | str | None = None,
    seed: int = 42,
    device: torch.device | None = None,
) -> DataBundle:
    """Construct a dataset and its dataloader together.

    Args:
        cfg: Data configuration.
        paths: Resolved project paths.
        split: Override the configured split.
        seed: Seed for the shuffle generator.
        device: Target device; disables ``pin_memory`` when not CUDA.

    Returns:
        The dataset, its loader, and the split they cover.
    """
    dataset = build_dataset(cfg, paths, split=split)

    shuffle = cfg.loader.shuffle and dataset.split is Split.TRAIN
    if cfg.loader.shuffle and not shuffle:
        logger.info(
            "forcing shuffle=False for split %r so per-sample results stay comparable",
            dataset.split.value,
        )

    settings = LoaderSettings(
        batch_size=cfg.loader.batch_size,
        num_workers=cfg.loader.num_workers,
        shuffle=shuffle,
        drop_last=cfg.loader.drop_last,
        pin_memory=cfg.loader.pin_memory,
        persistent_workers=cfg.loader.persistent_workers,
        prefetch_factor=cfg.loader.prefetch_factor,
    )
    loader = build_dataloader(dataset, settings, seed=seed, device=device)
    return DataBundle(dataset=dataset, loader=loader, split=dataset.split)
