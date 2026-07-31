"""Reconstruction losses.

Phase 3 starts with L1 alone. SSIM, SAM and multi-scale reconstruction
follow. The loss system is registry-backed so that adding a loss is a
new module plus a config entry, never an edit to the training loop.
"""

from __future__ import annotations

__all__: list[str] = []
