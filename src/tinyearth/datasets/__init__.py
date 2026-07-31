"""Data pipeline for Earth observation sequences.

Phase 2 populates this subpackage with the EarthNet2021 dataset class,
temporal sequence generation, normalisation, optional cloud masking and
the train/val/test splits. Datasets yield::

    {"images": Tensor[T, C, H, W], "target": Tensor[C, H, W], "metadata": ...}

No dataset is redistributed by this repository; see ``docs/datasets.md``
(added in Phase 2) for download instructions.
"""

from __future__ import annotations

__all__: list[str] = []
