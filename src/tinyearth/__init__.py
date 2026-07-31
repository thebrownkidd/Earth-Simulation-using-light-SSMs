"""TinyEarth: parameter-efficient State Space Models for Earth observation forecasting.

TinyEarth investigates a single research question: *how small can a State Space
Model become while remaining competitive at Earth observation forecasting?* The
codebase is organised so that only the **temporal backbone** varies across
experiments; encoders, decoders, losses and the training loop are held fixed.

Subpackages:
    config: Hydra structured-config schemas and helpers.
    utils: Determinism, logging, device and registry utilities.
    datasets: Data pipeline (Phase 2).
    models: Encoders, temporal backbones, decoders and losses (Phases 3-4).
    training: Training loop and optimisation (Phase 3).
    evaluation: Forecast-quality and efficiency metrics (Phase 3).
    cli: Console entry points.
"""

from __future__ import annotations

__version__ = "0.1.0"

__all__ = ["__version__"]
