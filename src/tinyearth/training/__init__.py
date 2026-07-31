"""Training loop, optimisation and experiment tracking.

The design constraint is that **no training logic lives in an entry-point
script**: scripts compose a config and call in here. That keeps the loop
testable without spawning a subprocess.

Metric tracking (:mod:`tinyearth.training.tracking`) is deliberately separate
from the console diagnostics in :mod:`tinyearth.utils.logging`, so a tracking
backend can be swapped without touching diagnostics.
"""

from __future__ import annotations

from tinyearth.training.optim import (
    build_optimizer,
    build_scheduler,
    current_lr,
    parameter_groups,
)
from tinyearth.training.tracking import (
    JsonTracker,
    MetricTracker,
    MultiTracker,
    NullTracker,
    TensorBoardTracker,
    build_tracker,
)
from tinyearth.training.trainer import EpochResult, Trainer, TrainingResult

__all__ = [
    "EpochResult",
    "JsonTracker",
    "MetricTracker",
    "MultiTracker",
    "NullTracker",
    "TensorBoardTracker",
    "Trainer",
    "TrainingResult",
    "build_optimizer",
    "build_scheduler",
    "build_tracker",
    "current_lr",
    "parameter_groups",
]
