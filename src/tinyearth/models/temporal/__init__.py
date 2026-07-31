"""Temporal backbones -- the only component that varies across experiments.

Phase 3 adds ConvLSTM and temporal-transformer baselines for correctness
reference. Phase 4 adds the State Space Model backbones that the project's
research question is actually about, along with the parameter-count tiers
(tiny ~2M, small ~5M, base ~10M, large ~20M) that scale within this
component.
"""

from __future__ import annotations

__all__: list[str] = []
