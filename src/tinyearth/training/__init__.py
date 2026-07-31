"""Training loop, optimisation and experiment tracking.

Phase 3 adds the trainer. The design constraint is that no training logic
lives in an entry-point script: scripts compose a config and call in here.
Metric tracking backends (TensorBoard, optional Weights & Biases) are
defined here, separately from the console logging in
:mod:`tinyearth.utils.logging`.
"""

from __future__ import annotations

__all__: list[str] = []
