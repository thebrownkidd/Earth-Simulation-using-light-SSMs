"""Model components: encoders, temporal backbones, decoders and losses.

The architecture is fixed::

    image encoder  ->  temporal backbone  ->  decoder  ->  forecast

**Only the temporal backbone varies across experiments.** That control is what
makes the efficiency comparison meaningful, and the interfaces in
:mod:`tinyearth.models.base` are what enforce it.

Phase 3 provides the ConvLSTM and temporal-transformer baselines, whose purpose
is correctness and a reference point rather than benchmark numbers. Phase 4 adds
the State Space Model backbones and the parameter-count tiers (tiny ~2M, small
~5M, base ~10M, large ~20M) that scale within that component.

Example:
    >>> from tinyearth.models import TEMPORAL_BACKBONES
    >>> sorted(TEMPORAL_BACKBONES.keys())
    ['convlstm', 'transformer']
"""

from __future__ import annotations

from tinyearth.models.base import Decoder, Encoder, TemporalBackbone
from tinyearth.models.decoders import DECODERS, CNNDecoder
from tinyearth.models.encoders import ENCODERS, CNNEncoder
from tinyearth.models.factory import build_backbone, build_forecaster, build_loss
from tinyearth.models.forecaster import Forecaster, ParameterBreakdown, count_parameters
from tinyearth.models.losses import (
    LOSSES,
    CharbonnierLoss,
    CompositeLoss,
    ForecastLoss,
    L1Loss,
    L2Loss,
)
from tinyearth.models.temporal import (
    TEMPORAL_BACKBONES,
    ConvLSTMBackbone,
    TemporalTransformerBackbone,
)

__all__ = [
    "DECODERS",
    "ENCODERS",
    "LOSSES",
    "TEMPORAL_BACKBONES",
    "CNNDecoder",
    "CNNEncoder",
    "CharbonnierLoss",
    "CompositeLoss",
    "ConvLSTMBackbone",
    "Decoder",
    "Encoder",
    "ForecastLoss",
    "Forecaster",
    "L1Loss",
    "L2Loss",
    "ParameterBreakdown",
    "TemporalBackbone",
    "TemporalTransformerBackbone",
    "build_backbone",
    "build_forecaster",
    "build_loss",
    "count_parameters",
]
