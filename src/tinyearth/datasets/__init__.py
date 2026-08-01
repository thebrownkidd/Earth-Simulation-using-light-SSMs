"""Data pipeline for Earth observation sequences.

Every dataset yields the same contract, so swapping a data source never
requires touching a model, a loss or the training loop::

    {
        "images":      Tensor[T, C, H, W],   # history frames
        "target":      Tensor[K, C, H, W],   # frames to forecast
        "metadata":    SampleMetadata,       # provenance
        "images_mask": Tensor[T, 1, H, W],   # optional, 1 = cloud-free
        "target_mask": Tensor[K, 1, H, W],   # optional
    }

Registered datasets:

``earthnet2021``
    The real dataset. Requires a download; see ``docs/datasets.md``. TinyEarth
    does not redistribute it.
``synthetic``
    Generated cubes in the identical on-disk format, for tests, CI and the
    example notebook. Results from it are meaningless.

Example:
    >>> from tinyearth.datasets import DATASETS, LoaderSettings, build_dataloader
    >>> dataset = DATASETS.build("synthetic", root="/tmp/tiny", history_length=4)
    >>> loader = build_dataloader(dataset, LoaderSettings(batch_size=2))
"""

from __future__ import annotations

from torch.utils.data import Dataset

from tinyearth.datasets.crops import CropMode, SpatialCrop
from tinyearth.datasets.earthnet2021 import EarthNet2021Dataset, IndexEntry
from tinyearth.datasets.factory import (
    DataBundle,
    build_datamodule,
    build_dataset,
    resolve_dataset_root,
)
from tinyearth.datasets.loaders import LoaderSettings, build_dataloader, describe_batch
from tinyearth.datasets.masking import MaskPolicy, apply_mask_policy, passes_validity_threshold
from tinyearth.datasets.minicube import (
    BAND_NAMES,
    RGB_INDICES,
    Minicube,
    MinicubeFormatError,
    read_minicube,
)
from tinyearth.datasets.normalization import (
    ChannelStandardizer,
    ChannelStatistics,
    IdentityNormalizer,
    Normalizer,
    build_normalizer,
    compute_channel_statistics,
)
from tinyearth.datasets.splits import (
    TEST_SPLITS,
    Split,
    SplitNotFoundError,
    assign_partition,
    summarise_partition,
)
from tinyearth.datasets.synthetic import (
    SyntheticEarthNet2021,
    SyntheticSpec,
    write_synthetic_dataset,
)
from tinyearth.datasets.types import Batch, Sample, SampleMetadata, collate_samples
from tinyearth.datasets.windows import Window, WindowMode, WindowSpec, generate_windows
from tinyearth.utils.registry import Registry

__all__ = [
    "BAND_NAMES",
    "DATASETS",
    "RGB_INDICES",
    "TEST_SPLITS",
    "Batch",
    "ChannelStandardizer",
    "ChannelStatistics",
    "CropMode",
    "DataBundle",
    "EarthNet2021Dataset",
    "IdentityNormalizer",
    "IndexEntry",
    "LoaderSettings",
    "MaskPolicy",
    "Minicube",
    "MinicubeFormatError",
    "Normalizer",
    "Sample",
    "SampleMetadata",
    "SpatialCrop",
    "Split",
    "SplitNotFoundError",
    "SyntheticEarthNet2021",
    "SyntheticSpec",
    "Window",
    "WindowMode",
    "WindowSpec",
    "apply_mask_policy",
    "assign_partition",
    "build_dataloader",
    "build_datamodule",
    "build_dataset",
    "build_normalizer",
    "collate_samples",
    "compute_channel_statistics",
    "describe_batch",
    "generate_windows",
    "passes_validity_threshold",
    "read_minicube",
    "resolve_dataset_root",
    "summarise_partition",
    "write_synthetic_dataset",
]

DATASETS: Registry[Dataset[Sample]] = Registry("dataset")
"""Datasets selectable by name from a config.

Populated below rather than by decorator, because the classes are also part of
the public API and should not have their definitions coupled to registration.
"""

DATASETS.register("earthnet2021")(EarthNet2021Dataset)
DATASETS.register("synthetic")(SyntheticEarthNet2021)
